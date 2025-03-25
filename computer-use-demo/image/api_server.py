import asyncio
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from computer_use_demo.loop import (
    APIProvider,
    BetaContentBlockParam,
    BetaMessageParam,
    ConversationStore,
    ToolResult,
    sampling_loop,
)

PROVIDER_TO_DEFAULT_MODEL_NAME: dict[APIProvider, str] = {
    APIProvider.ANTHROPIC: "claude-3-7-sonnet-20250219",
    APIProvider.BEDROCK: "anthropic.claude-3-5-sonnet-20241022-v2:0",
    APIProvider.VERTEX: "claude-3-5-sonnet-v2@20241022",
}

app = FastAPI()


class MessageRequest(BaseModel):
    message: str
    tools: Optional[str] = None
    conversation_id: Optional[int] = None
    model: Optional[str] = None
    system_prompt_suffix: Optional[str] = None
    provider: Optional[str] = "anthropic"
    only_n_most_recent_images: Optional[int] = 10
    max_tokens: Optional[int] = 4096


class MessageResponse(BaseModel):
    conversation_id: int
    messages: List[Dict[str, Any]]
    completed: bool


# Global store for messages
message_store: Dict[int, List[BetaMessageParam]] = {}


def message_callback(content: BetaContentBlockParam):
    """Callback for message outputs"""
    # We'll collect messages through the messages list


def tool_callback(result: ToolResult, tool_id: str):
    """Callback for tool outputs"""
    # We'll collect tool results through the messages list


def api_callback(request, response, error):
    """Callback for API responses"""
    if error:
        logging.error(f"API Error: {error}")


@app.post("/message", response_model=MessageResponse)
async def process_message(request: MessageRequest):
    try:
        # Get API key from environment
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not set")

        # Get or create message history
        messages = (
            message_store.get(request.conversation_id, [])
            if request.conversation_id
            else []
        )

        # Add the new message
        messages.append(
            {
                "role": "user",
                "content": [{"type": "text", "text": request.message}],
            }
        )

        # Set up provider
        provider = APIProvider.ANTHROPIC
        model = PROVIDER_TO_DEFAULT_MODEL_NAME[provider]

        # Create conversation store
        conversation_store = await ConversationStore.create()

        current_conversation_id = await conversation_store.create_conversation(
            model=model,
            conv_type="single",
        )

        # Run the sampling loop
        asyncio.create_task(
            sampling_loop(
                model=model,
                provider=provider,
                system_prompt_suffix=request.system_prompt_suffix or "",
                messages=messages,
                output_callback=message_callback,
                tool_output_callback=tool_callback,
                api_response_callback=api_callback,
                api_key=api_key,
                only_n_most_recent_images=1,
                max_tokens=8192,
                conversation_store=conversation_store,
                current_conversation_id=current_conversation_id,
                conversation_type="single",
                tool_version="computer_use_20250124",
                thinking_budget=4096,
                token_efficient_tools_beta=False,
            )
        )

        return MessageResponse(
            conversation_id=current_conversation_id, messages=[], completed=True
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: int):
    """Get message history for a conversation"""
    if conversation_id not in message_store:
        raise HTTPException(status_code=404, detail="Conversation not found")

    return {"messages": message_store[conversation_id]}


@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
