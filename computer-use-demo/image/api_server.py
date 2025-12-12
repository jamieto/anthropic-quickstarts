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

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Get environment variables
CHAT_ID = os.getenv("CHAT_ID")
SESSION_ID = os.getenv("SESSION_ID")
AGENT_ID = os.getenv("AGENT_ID")

logger.info(f"API Server starting with CHAT_ID={CHAT_ID}, SESSION_ID={SESSION_ID}, AGENT_ID={AGENT_ID}")

PROVIDER_TO_DEFAULT_MODEL_NAME: dict[APIProvider, str] = {
    APIProvider.ANTHROPIC: "claude-sonnet-4-5-20250929",
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
    use_extended_context: bool = False
    parent_chat_id: Optional[int] = None
    agent_name: Optional[str] = None
    cleanup_on_complete: bool = True
    session_id: Optional[str] = None
    spawn_id: Optional[int] = None


class MessageResponse(BaseModel):
    conversation_id: int
    chat_id: Optional[str] = None
    session_id: Optional[str] = None
    agent_name: Optional[str] = None
    messages: List[Dict[str, Any]]
    completed: bool


message_store: Dict[int, List[BetaMessageParam]] = {}


def message_callback(content: BetaContentBlockParam):
    logger.debug(f"Message callback: {type(content)}")


def tool_callback(result: ToolResult, tool_id: str):
    logger.debug(f"Tool callback: tool_id={tool_id}, error={result.error}")


def api_callback(request, response, error):
    if error:
        logger.error(f"API Callback Error: {error}")
    else:
        logger.debug(f"API Callback: response received")


async def run_sampling_loop_with_logging(
    conversation_id: int,
    **kwargs
):
    """Wrapper to catch and log any errors from sampling_loop."""
    logger.info(f"[Conv {conversation_id}] Starting sampling loop...")
    try:
        result = await sampling_loop(**kwargs)
        logger.info(f"[Conv {conversation_id}] Sampling loop completed successfully")
        return result
    except Exception as e:
        logger.exception(f"[Conv {conversation_id}] Sampling loop FAILED with error: {e}")
        raise


@app.post("/message", response_model=MessageResponse)
async def process_message(request: MessageRequest):
    logger.info(f"=== /message endpoint called ===")
    logger.info(f"Message: {request.message[:100]}...")
    logger.info(f"CHAT_ID={CHAT_ID}, SESSION_ID={SESSION_ID}, AGENT_ID={AGENT_ID}")
    
    try:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            logger.error("ANTHROPIC_API_KEY not set!")
            raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not set")
        
        logger.debug(f"API key found (length={len(api_key)})")

        messages = (
            message_store.get(request.conversation_id, [])
            if request.conversation_id
            else []
        )

        messages.append(
            {
                "role": "user",
                "content": [{"type": "text", "text": request.message}],
            }
        )

        provider = APIProvider.ANTHROPIC
        model = request.model or PROVIDER_TO_DEFAULT_MODEL_NAME[provider]
        logger.info(f"Using model: {model}")

        # Create conversation store
        logger.debug("Creating ConversationStore...")
        try:
            conversation_store = await ConversationStore.create()
            logger.debug("ConversationStore created successfully")
        except Exception as e:
            logger.exception(f"Failed to create ConversationStore: {e}")
            raise HTTPException(status_code=500, detail=f"Database connection failed: {e}")

        agent_name = request.agent_name or AGENT_ID or "main-orchestrator"
        
        logger.info(f"Creating conversation record: chat_id={CHAT_ID}, session_id={SESSION_ID}, agent_name={agent_name}")

        try:
            current_conversation_id = await conversation_store.create_conversation(
                model=model,
                conv_type="single",
                status="running",
                chat_id=int(CHAT_ID) if CHAT_ID else None,
                session_id=SESSION_ID,
                parent_chat_id=request.parent_chat_id,
                agent_name=agent_name,
            )
            logger.info(f"Conversation created with ID: {current_conversation_id}")
        except Exception as e:
            logger.exception(f"Failed to create conversation: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to create conversation: {e}")

        # Start the sampling loop as a background task
        logger.info(f"Spawning sampling_loop as background task...")

        # Determine if this is a sub-agent or main orchestrator
        is_sub_agent = request.parent_chat_id is not None

        # Main orchestrator should NEVER self-cleanup - it waits for more prompts
        # Sub-agents can self-cleanup based on the request
        should_cleanup = request.cleanup_on_complete if is_sub_agent else False
        
        task = asyncio.create_task(
            run_sampling_loop_with_logging(
                conversation_id=current_conversation_id,
                model=model,
                provider=provider,
                system_prompt_suffix=request.system_prompt_suffix or "",
                messages=messages,
                output_callback=message_callback,
                tool_output_callback=tool_callback,
                api_response_callback=api_callback,
                api_key=api_key,
                only_n_most_recent_images=90,
                max_tokens=32768,
                conversation_store=conversation_store,
                current_conversation_id=current_conversation_id,
                conversation_type="single",
                tool_version="computer_use_20250124",
                thinking_budget=4096,
                token_efficient_tools_beta=False,
                use_extended_context=request.use_extended_context,
                agent_name=agent_name,
                cleanup_on_complete=should_cleanup,
                session_id=request.session_id or SESSION_ID,
                spawn_id=request.spawn_id,
            )
        )
        
        # Add callback to log when task completes or fails
        def task_done_callback(t):
            if t.exception():
                logger.error(f"Background task failed: {t.exception()}")
                asyncio.create_task(_mark_failed(current_conversation_id, str(t.exception())))
            else:
                logger.info(f"Background task completed")

        async def _mark_failed(conversation_id: int, error_message: str):
            try:
                conversation_store = await ConversationStore.create()
                await conversation_store.update_status(conversation_id, "failed", error_message)
            except Exception as e:
                logger.error(f"Failed to mark conversation {conversation_id} as failed: {e}")
        
        task.add_done_callback(task_done_callback)
        
        logger.info(f"Background task created, returning response immediately")

        return MessageResponse(
            conversation_id=current_conversation_id,
            chat_id=CHAT_ID,
            session_id=SESSION_ID,
            agent_name=agent_name,
            messages=[],
            completed=False,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Unexpected error in /message endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: int):
    logger.debug(f"GET /conversations/{conversation_id}")
    try:
        conversation_store = await ConversationStore.create()
        conversation = await conversation_store.get_conversation(conversation_id)
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return conversation
    except Exception as e:
        logger.exception(f"Error getting conversation: {e}")
        raise


@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

@app.get("/debug/db")
async def debug_db():
    """Test database connectivity."""
    try:
        logger.info("Testing database connection...")
        conversation_store = await ConversationStore.create()
        
        async with conversation_store.pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT 1")
                result = await cur.fetchone()
                
        logger.info("Database connection successful")
        return {"status": "ok", "result": result}
    except Exception as e:
        logger.exception(f"Database connection failed: {e}")
        return {"status": "error", "error": str(e)}


@app.get("/debug/env")
async def debug_env():
    """Check environment variables."""
    return {
        "CHAT_ID": os.getenv("CHAT_ID"),
        "SESSION_ID": os.getenv("SESSION_ID"),
        "AGENT_ID": os.getenv("AGENT_ID"),
        "DB_HOST": os.getenv("DB_HOST"),
        "DB_PORT": os.getenv("DB_PORT"),
        "DB_NAME": os.getenv("DB_NAME"),
        "DB_USER": os.getenv("DB_USER"),
        "ANTHROPIC_API_KEY": "set" if os.getenv("ANTHROPIC_API_KEY") else "NOT SET",
    }

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting API server on port 8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)