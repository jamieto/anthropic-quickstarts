"""
Agentic sampling loop that calls the Claude API and local implementation of anthropic-defined computer use tools.
"""

import json
import os
import platform
from collections.abc import Callable
from datetime import datetime
from enum import StrEnum
from typing import Any, Optional, cast

import aiomysql
import httpx
from anthropic import (
    Anthropic,
    AnthropicBedrock,
    AnthropicVertex,
    APIError,
    APIResponseValidationError,
    APIStatusError,
)
from anthropic.types.beta import (
    BetaCacheControlEphemeralParam,
    BetaContentBlockParam,
    BetaImageBlockParam,
    BetaMessage,
    BetaMessageParam,
    BetaTextBlock,
    BetaTextBlockParam,
    BetaToolResultBlockParam,
    BetaToolUseBlockParam,
)

from .tools import (
    TOOL_GROUPS_BY_VERSION,
    ToolCollection,
    ToolResult,
    ToolVersion,
)

PROMPT_CACHING_BETA_FLAG = "prompt-caching-2024-07-31"


class APIProvider(StrEnum):
    ANTHROPIC = "anthropic"
    BEDROCK = "bedrock"
    VERTEX = "vertex"


# This system prompt is optimized for the Docker environment in this repository and
# specific tool combinations enabled.
# We encourage modifying this system prompt to ensure the model has context for the
# environment it is running in, and to provide any additional information that may be
# helpful for the task at hand.
SYSTEM_PROMPT = f"""<SYSTEM_CAPABILITY>
* You are utilising an Ubuntu virtual machine using {platform.machine()} architecture with internet access.
* You can feel free to install Ubuntu applications with your bash tool. Use curl instead of wget.
* To open firefox, please just click on the firefox icon.  Note, firefox-esr is what is installed on your system.
* Using bash tool you can start GUI applications, but you need to set export DISPLAY=:1 and use a subshell. For example "(DISPLAY=:1 xterm &)". GUI apps run with bash tool will appear within your desktop environment, but they may take some time to appear. Take a screenshot to confirm it did.
* When using your bash tool with commands that are expected to output very large quantities of text, redirect into a tmp file and use str_replace_based_edit_tool or `grep -n -B <lines before> -A <lines after> <query> <filename>` to confirm output.
* When viewing a page it can be helpful to zoom out so that you can see everything on the page.  Either that, or make sure you scroll down to see everything before deciding something isn't available.
* When using your computer function calls, they take a while to run and send back to you.  Where possible/feasible, try to chain multiple of these calls all into one function calls request.
* The current date is {datetime.today().strftime('%A, %B %-d, %Y')}.
* Always look into the the /home/computeruse/chat-tools directory and go through the list of txt file as it contains information of the tool and the URL to access the tool. You will read through the content of each tools to determine if the tool can be used to solve the task on hand. Tools inside this directory are accessible via the URL provided in the txt file. You must open the URL in Firefox to access the tool.
* When you're given a file name to work with, you will look into the /home/computeruse/files directory to find it and work with it.
* If you need any credentials for any service, you will first look into the /home/computeruse/credentials directory and find the appropriate json file. Each of the json files will have the necessary credentials and the file name will be the service name.
* If you were asked to run or execute any tools, you will first look into the /home/computeruse/tools directory to find it. Each of the tools will have its own Readme file. Only when you cannot find a tool that is suitable for the task at hand, you will then create your own tool.
* If you were tasked to create a new tool, you must make sure all tools created are stored in the /home/computeruse/tools directory with their own directory. Each tool directory will have the necessary files to run the tool and it must have a Readme.md file. The Readme.md file will have the necessary information about the tool and how to run it. The tool's direction name will be the tool's name. Naming convention for the tool's directory is lowercase with underscores.
</SYSTEM_CAPABILITY>

<IMPORTANT>
* When using Firefox, if a startup wizard appears, IGNORE IT.  Do not even click "skip this step".  Instead, click on the address bar where it says "Search or enter address", and enter the appropriate search term or URL there.
* If the item you are looking at is a pdf, if after taking a single screenshot of the pdf it seems that you want to read the entire document instead of trying to continue to read the pdf from your screenshots + navigation, determine the URL, use curl to download the pdf, install and use pdftotext to convert it to a text file, and then read that text file directly with your str_replace_based_edit_tool.
* When inputting text into a text field, you must wait for 2 seconds and then you must confirm that the text has been entered correctly by checking the text field after you have entered the text. Make sure to check for any extra spaces or characters that may have been added. Make sure to check for any typos. Make sure to check for any missing characters. Before submitting, ALWAYS take a screenshot of the text field to confirm that the text has been entered correctly. Only proceed to the next step if the text has been entered correctly.
</IMPORTANT>"""


class ConversationStore:
    def __init__(self, pool: aiomysql.Pool):
        self.pool = pool

    @staticmethod
    async def create(
        host: Optional[str] = None,
        port: Optional[int] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        db: Optional[str] = None,
    ):
        """
        Create a ConversationStore instance with either provided parameters or environment variables.
        """
        connection_params = {
            "host": host or os.getenv("DB_HOST", "localhost"),
            "port": port or int(os.getenv("DB_PORT", "3306")),
            "user": user or os.getenv("DB_USER", "root"),
            "password": password or os.getenv("DB_PASSWORD", ""),
            "db": db or os.getenv("DB_NAME", "multiai"),
            "autocommit": True,
        }

        try:
            pool = await aiomysql.create_pool(**connection_params)
            return ConversationStore(pool)
        except Exception as e:
            raise ConnectionError(f"Failed to connect to MySQL: {str(e)}") from e

    async def create_conversation(self, model: str, conv_type: str, status: str) -> int:
        """Create a new conversation record and return its ID"""
        async with self.pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO computer_use_chats (
                        name, type, status, status_updated_at, created_at, updated_at
                    ) VALUES (%s, %s, %s, %s, %s, %s)
                """,
                    (
                        model,
                        conv_type,
                        status,
                        datetime.utcnow(),
                        datetime.utcnow(),
                        datetime.utcnow(),
                    ),
                )
                await conn.commit()
                # Get the auto-generated ID
                conversation_id = cur.lastrowid
                return conversation_id

    async def store_message(
        self,
        conversation_id: int,
        role: str,
        content: str,
        raw_content: str,
        tool_id: Optional[str] = None,
        is_error: bool = False,
        image_data: Optional[str] = None,
    ):
        """Store a message associated with a conversation"""
        async with self.pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO computer_use_chat_messages (
                        computer_use_chat_id, role, content, tool_id,
                        is_error, timestamp, image_data, raw_content, created_at, updated_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                    (
                        conversation_id,
                        role,
                        content,
                        tool_id,
                        is_error,
                        datetime.utcnow(),
                        image_data,
                        raw_content,
                        datetime.utcnow(),
                        datetime.utcnow(),
                    ),
                )
                await conn.commit()

    async def mark_completed(self, conversation_id: int):
        async with self.pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    UPDATE computer_use_chats
                    SET completed_at = %s
                    WHERE id = %s AND type = 'single'
                """,
                    (datetime.utcnow(), conversation_id),
                )
                await conn.commit()


async def _update_status(
    conversation_store: ConversationStore,
    conversation_id: int,
    status: str,
    message: Optional[str] = None,
):
    """A helper to update the status and related fields in the database."""
    now = datetime.utcnow()
    pool = conversation_store.pool

    async with pool.acquire() as conn:
        async with conn.cursor() as cur:
            # The base query updates the core status fields
            query = """
                UPDATE computer_use_chats
                SET status = %s, status_updated_at = %s, status_message = %s
            """
            params = [status, now, message]

            # If the new status is 'completed', also set the completed_at timestamp
            if status == "completed":
                query += ", completed_at = %s"
                params.append(now)

            query += " WHERE id = %s"
            params.append(conversation_id)

            await cur.execute(query, tuple(params))
            await conn.commit()


async def sampling_loop(
    *,
    model: str,
    provider: APIProvider,
    system_prompt_suffix: str,
    messages: list[BetaMessageParam],
    output_callback: Callable[[BetaContentBlockParam], None],
    tool_output_callback: Callable[[ToolResult, str], None],
    api_response_callback: Callable[
        [httpx.Request, httpx.Response | object | None, Exception | None], None
    ],
    api_key: str,
    only_n_most_recent_images: int | None = None,
    max_tokens: int = 4096,
    conversation_store: Optional[ConversationStore] = None,
    current_conversation_id: Optional[int] = None,
    conversation_type: str = "continuous",
    tool_version: ToolVersion,
    thinking_budget: int | None = None,
    token_efficient_tools_beta: bool = False,
):
    """
    Agentic sampling loop for the assistant/tool interaction of computer use.
    """
    tool_group = TOOL_GROUPS_BY_VERSION[tool_version]
    tool_collection = ToolCollection(*(ToolCls() for ToolCls in tool_group.tools))
    system = BetaTextBlockParam(
        type="text",
        text=f"{SYSTEM_PROMPT}{' ' + system_prompt_suffix if system_prompt_suffix else ''}",
    )

    # Create conversation store if not provided
    if conversation_store is None:
        # try:
        conversation_store = await ConversationStore.create()
    # except ConnectionError as e:
    #     return messages, None

    if current_conversation_id is None:
        current_conversation_id = await conversation_store.create_conversation(
            model=model,
            conv_type=conversation_type,
            status="running",
        )

    while True:
        # Check for pause/stop signals at the start of each iteration
        try:
            async with conversation_store.pool.acquire() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(
                        "SELECT status FROM computer_use_chats WHERE id = %s",
                        (current_conversation_id,),
                    )
                    result = await cur.fetchone()
                    current_status = result[0] if result else "failed"

            if current_status == "stopping":
                await _update_status(
                    conversation_store,
                    current_conversation_id,
                    "cancelled",
                    "Task cancelled by user.",
                )
                break  # Exit the loop gracefully

            if current_status == "pausing":
                await _update_status(
                    conversation_store,
                    current_conversation_id,
                    "paused",
                    "Task paused by user.",
                )
                break  # Exit the loop gracefully

        except Exception as e:
            await _update_status(
                conversation_store,
                current_conversation_id,
                "failed",
                f"Internal error during status check: {e}",
            )
            break

        enable_prompt_caching = False
        betas = [tool_group.beta_flag] if tool_group.beta_flag else []
        if token_efficient_tools_beta:
            betas.append("token-efficient-tools-2025-02-19")
        image_truncation_threshold = only_n_most_recent_images or 0
        if provider == APIProvider.ANTHROPIC:
            client = Anthropic(api_key=api_key, max_retries=4)
            enable_prompt_caching = True
        elif provider == APIProvider.VERTEX:
            client = AnthropicVertex()
        elif provider == APIProvider.BEDROCK:
            client = AnthropicBedrock()

        if enable_prompt_caching:
            betas.append(PROMPT_CACHING_BETA_FLAG)
            _inject_prompt_caching(messages)
            # Because cached reads are 10% of the price, we don't think it's
            # ever sensible to break the cache by truncating images
            only_n_most_recent_images = 0
            # Use type ignore to bypass TypedDict check until SDK types are updated
            system["cache_control"] = {"type": "ephemeral"}  # type: ignore

        if only_n_most_recent_images:
            _maybe_filter_to_n_most_recent_images(
                messages,
                only_n_most_recent_images,
                min_removal_threshold=image_truncation_threshold,
            )
        extra_body = {}
        if thinking_budget:
            # Ensure we only send the required fields for thinking
            extra_body = {
                "thinking": {"type": "enabled", "budget_tokens": thinking_budget}
            }

        # Store the last user message into db
        last_user_message = messages[-1]
        if last_user_message["role"] == "user":
            if current_conversation_id is not None:
                await conversation_store.store_message(
                    conversation_id=current_conversation_id,
                    role="user",
                    content=json.dumps(last_user_message["content"]),
                    raw_content=json.dumps(last_user_message),
                    tool_id="user-input",
                )

        # Call the API
        # we use raw_response to provide debug information to streamlit. Your
        # implementation may be able call the SDK directly with:
        # `response = client.messages.create(...)` instead.
        try:
            raw_response = client.beta.messages.with_raw_response.create(
                max_tokens=max_tokens if max_tokens is not None else 4096,
                messages=messages,
                model=model,
                system=[system],
                tools=tool_collection.to_params(),
                betas=betas,
                extra_body=extra_body,
            )
            # except (APIStatusError, APIResponseValidationError) as e:
            #     api_response_callback(e.request, e.response, e)
            #     return messages, current_conversation_id
            # except APIError as e:
            #     api_response_callback(e.request, e.body, e)
            #     return messages, current_conversation_id

            api_response_callback(
                raw_response.http_response.request, raw_response.http_response, None
            )

            response = raw_response.parse()

            response_params = _response_to_params(response)
            response_message: BetaMessageParam = {
                "role": "assistant",
                "content": response_params,
            }
            messages.append(response_message)

            if current_conversation_id is not None:
                await conversation_store.store_message(
                    conversation_id=current_conversation_id,
                    role="assistant",
                    content=json.dumps(response_params),
                    tool_id="response",
                    raw_content=json.dumps(response_message),
                )

            tool_result_content: list[BetaToolResultBlockParam] = []
            for content_block in response_params:
                output_callback(content_block)

                if (
                    isinstance(content_block, dict)
                    and content_block.get("type") == "tool_use"
                ):
                    # Type narrowing for tool use blocks
                    tool_use_block = cast(BetaToolUseBlockParam, content_block)
                    result = await tool_collection.run(
                        name=tool_use_block["name"],
                        tool_input=cast(
                            dict[str, Any], tool_use_block.get("input", {})
                        ),
                    )
                    tool_result_content.append(
                        _make_api_tool_result(result, tool_use_block["id"])
                    )
                    tool_output_callback(result, tool_use_block["id"])

            if not tool_result_content:
                if current_conversation_id is not None:
                    await _update_status(
                        conversation_store,
                        current_conversation_id,
                        "completed",
                        "Task finished successfully without further tool use.",
                    )
                    await conversation_store.mark_completed(current_conversation_id)
                    return messages, current_conversation_id

            result_message: BetaMessageParam = {
                "role": "user",
                "content": tool_result_content,
            }

            if current_conversation_id is not None:
                await conversation_store.store_message(
                    conversation_id=current_conversation_id,
                    role="tool",
                    content=json.dumps(tool_result_content),
                    tool_id="response",
                    raw_content=json.dumps(result_message),
                )

            messages.append(result_message)

        except (APIStatusError, APIResponseValidationError) as e:
            # MODIFIED: On API error, update status to 'failed'
            error_message = f"API Error: {e}"

            await _update_status(
                conversation_store, current_conversation_id, "failed", error_message
            )
            api_response_callback(e.request, e.response, e)
            return messages, current_conversation_id
        except APIError as e:
            # MODIFIED: On API error, update status to 'failed'
            error_message = f"API Error: {e}"

            await _update_status(
                conversation_store, current_conversation_id, "failed", error_message
            )
            api_response_callback(e.request, e.body, e)
            return messages, current_conversation_id
        except Exception as e:
            # MODIFIED: On any other unexpected error, update status to 'failed'
            error_message = f"An unexpected error occurred: {e}"

            await _update_status(
                conversation_store, current_conversation_id, "failed", error_message
            )
            raise  # Re-raise the exception after logging the status


def _maybe_filter_to_n_most_recent_images(
    messages: list[BetaMessageParam],
    images_to_keep: int,
    min_removal_threshold: int,
):
    """
    With the assumption that images are screenshots that are of diminishing value as
    the conversation progresses, remove all but the final `images_to_keep` tool_result
    images in place, with a chunk of min_removal_threshold to reduce the amount we
    break the implicit prompt cache.
    """
    if images_to_keep is None:
        return messages

    tool_result_blocks = cast(
        list[BetaToolResultBlockParam],
        [
            item
            for message in messages
            for item in (
                message["content"] if isinstance(message["content"], list) else []
            )
            if isinstance(item, dict) and item.get("type") == "tool_result"
        ],
    )

    total_images = sum(
        1
        for tool_result in tool_result_blocks
        for content in tool_result.get("content", [])
        if isinstance(content, dict) and content.get("type") == "image"
    )

    images_to_remove = total_images - images_to_keep
    # for better cache behavior, we want to remove in chunks
    images_to_remove -= images_to_remove % min_removal_threshold

    for tool_result in tool_result_blocks:
        if isinstance(tool_result.get("content"), list):
            new_content = []
            for content in tool_result.get("content", []):
                if isinstance(content, dict) and content.get("type") == "image":
                    if images_to_remove > 0:
                        images_to_remove -= 1
                        continue
                new_content.append(content)
            tool_result["content"] = new_content


def _response_to_params(
    response: BetaMessage,
) -> list[BetaContentBlockParam]:
    res: list[BetaContentBlockParam] = []
    for block in response.content:
        if isinstance(block, BetaTextBlock):
            if block.text:
                res.append(BetaTextBlockParam(type="text", text=block.text))
            elif getattr(block, "type", None) == "thinking":
                # Handle thinking blocks - include signature field
                thinking_block = {
                    "type": "thinking",
                    "thinking": getattr(block, "thinking", None),
                }
                if hasattr(block, "signature"):
                    thinking_block["signature"] = getattr(block, "signature", None)
                res.append(cast(BetaContentBlockParam, thinking_block))
        else:
            # Handle tool use blocks normally
            res.append(cast(BetaToolUseBlockParam, block.model_dump()))
    return res


def _inject_prompt_caching(
    messages: list[BetaMessageParam],
):
    """
    Set cache breakpoints for the 3 most recent turns
    one cache breakpoint is left for tools/system prompt, to be shared across sessions
    """

    breakpoints_remaining = 3
    for message in reversed(messages):
        if message["role"] == "user" and isinstance(
            content := message["content"], list
        ):
            if breakpoints_remaining:
                breakpoints_remaining -= 1
                # Use type ignore to bypass TypedDict check until SDK types are updated
                content[-1]["cache_control"] = BetaCacheControlEphemeralParam(  # type: ignore
                    {"type": "ephemeral"}
                )
            else:
                if isinstance(content[-1], dict) and "cache_control" in content[-1]:
                    del content[-1]["cache_control"]  # type: ignore
                # we'll only every have one extra turn per loop
                break


def _make_api_tool_result(
    result: ToolResult, tool_use_id: str
) -> BetaToolResultBlockParam:
    """Convert an agent ToolResult to an API ToolResultBlockParam."""
    tool_result_content: list[BetaTextBlockParam | BetaImageBlockParam] | str = []
    is_error = False
    if result.error:
        is_error = True
        tool_result_content = _maybe_prepend_system_tool_result(result, result.error)
    else:
        if result.output:
            tool_result_content.append(
                {
                    "type": "text",
                    "text": _maybe_prepend_system_tool_result(result, result.output),
                }
            )
        if result.base64_image:
            tool_result_content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": result.base64_image,
                    },
                }
            )
    return {
        "type": "tool_result",
        "content": tool_result_content,
        "tool_use_id": tool_use_id,
        "is_error": is_error,
    }


def _maybe_prepend_system_tool_result(result: ToolResult, result_text: str):
    if result.system:
        result_text = f"<system>{result.system}</system>\n{result_text}"
    return result_text
