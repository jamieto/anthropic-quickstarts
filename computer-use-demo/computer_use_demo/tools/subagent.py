"""
SubAgent Tool - Allows the agent to spawn other agents via the broker.
"""

import os
import uuid
import asyncio
from typing import Optional
from venv import logger
import httpx

from .base import BaseAnthropicTool, ToolResult


class SubAgentTool(BaseAnthropicTool):
    """
    Allows the main agent to spawn sub-agents by calling the broker.
    Each sub-agent gets its own VM/pod but shares the same /home/computeruse/project/ workspace.
    """
    
    name = "spawn_subagent"
    api_type = "custom"
    
    def __init__(self):
        self.broker_url = os.getenv("BROKER_URL", "http://broker:8001")
        self.broker_token = os.getenv("BROKER_TOKEN")
        self.chat_id = os.getenv("CHAT_ID")
        self.user_id = os.getenv("USER_ID")
        self.my_session_id = os.getenv("SESSION_ID")  # This pod's session ID
        self.my_agent_id = os.getenv("AGENT_ID")
        
        # Set by sampling_loop when tool is initialized
        # This is the conversation_id in computer_use_chats table
        self.my_conversation_id: Optional[int] = None
    
    def to_params(self) -> dict:
        return {
            "name": self.name,
            "description": """Spawn a specialized sub-agent to work on a specific task.

The sub-agent:
- Runs in its own isolated environment with full computer use capabilities
- Shares the same /home/computeruse/project/ workspace (can see and modify files you create)
- Has its own context window (won't pollute your context)
- Can use bash, file editing, browser, screenshots - everything you can

Use this when you need:
- Specialized expertise (security review, data analysis, UI design)
- Parallel work (spawn multiple agents for different parts)
- Deep focus on a subtask without losing your main context
- Another perspective or approach

You define WHO they are (system_prompt) and WHAT they do (task).""",
            "input_schema": {
                "type": "object",
                "properties": {
                    "agent_name": {
                        "type": "string",
                        "description": "Short identifier for this agent (e.g., 'security-reviewer', 'data-analyst', 'frontend-dev'). Used for tracking and logging."
                    },
                    "display_name": {
                        "type": "string",
                        "description": "Human-readable name for this agent (e.g., 'Specification Writer', 'Frontend Developer'). Shown in UI."
                    },
                    "system_prompt": {
                        "type": "string", 
                        "description": """The sub-agent's identity and expertise. Define:
- WHO they are (role, expertise, background)
- HOW they work (methodology, approach, standards)
- WHAT they focus on (specific areas, constraints)

Be specific - this shapes their entire behavior."""
                    },
                    "task": {
                        "type": "string",
                        "description": """The specific task to complete. Include:
- Clear objective
- Expected deliverable
- Any context they need
- Where to put results (usually /home/computeruse/project/)

They can see /home/computeruse/project/ and /home/computeruse/uploads/ just like you."""
                    },
                    "wait_for_completion": {
                        "type": "boolean",
                        "description": "If true (default), wait for sub-agent to finish and return their result. If false, spawn and continue immediately (for parallel work).",
                        "default": True
                    },
                    "timeout_minutes": {
                        "type": "integer",
                        "description": "Max time to wait for completion (default: 30 minutes)",
                        "default": 30
                    }
                },
                "required": ["agent_name", "system_prompt", "task"]
            }
        }
    
    async def __call__(
        self,
        agent_name: str,
        display_name: str,
        system_prompt: str,
        task: str,
        wait_for_completion: bool = True,
        timeout_minutes: int = 30,
        **kwargs
    ) -> ToolResult:
        """Spawn a sub-agent via the broker."""
        
        unique_id = uuid.uuid4().hex[:8]
        session_id = f"chat-{self.chat_id}-sub-{agent_name}-{unique_id}"
        sub_agent_id = f"sub-{agent_name}-{unique_id}"
        sub_conversation_id = None  # Track this for cleanup
        
        logger.info(f"[SubAgentTool] Spawning sub-agent: {agent_name}")
        logger.info(f"[SubAgentTool] Session ID: {session_id}")
        
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                
                # 1. Create session via broker
                logger.info(f"[SubAgentTool] Creating session via broker...")
                response = await client.post(
                    f"{self.broker_url}/sessions",
                    json={
                        "session_id": session_id,
                        "chat_id": self.chat_id,
                        "user_id": self.user_id,
                        "agent_id": sub_agent_id,
                        "agent_name": agent_name,
                        "parent_session_id": self.my_session_id,
                    },
                    headers={"X-Broker-Token": self.broker_token}
                )
                
                if response.status_code == 409:
                    return ToolResult(error=f"Agent '{agent_name}' session already exists")
                    
                if response.status_code != 201:
                    return ToolResult(
                        error=f"Failed to create sub-agent session: {response.status_code} - {response.text}"
                    )
                
                session = response.json()
                service_name = session["service_name"]
                internal_api_url = f"http://{service_name}.default.svc.cluster.local:8000"
                
                logger.info(f"[SubAgentTool] Internal API URL: {internal_api_url}")
                
                # 2. Wait for pod to be ready
                logger.info(f"[SubAgentTool] Waiting for pod to be ready...")
                ready = await self._wait_for_ready(session_id, timeout_seconds=120)
                if not ready:
                    # CLEANUP: Pod failed to start
                    await self._cleanup_session(session_id, None, "Pod failed to start")
                    return ToolResult(error=f"Sub-agent '{agent_name}' pod failed to start within timeout")
                
                logger.info(f"[SubAgentTool] Pod is ready!")
                
                # 3. Wait for API to be healthy
                logger.info(f"[SubAgentTool] Waiting for API health check...")
                api_healthy = await self._wait_for_api_health(internal_api_url, timeout_seconds=60)
                if not api_healthy:
                    # CLEANUP: API failed health check
                    await self._cleanup_session(session_id, None, "API failed to become healthy")
                    return ToolResult(error=f"Sub-agent '{agent_name}' API failed to become healthy")
                
                logger.info(f"[SubAgentTool] API is healthy!")
                
                # 4. Send task to sub-agent
                logger.info(f"[SubAgentTool] Sending task to sub-agent...")
                
                task_response = await client.post(
                    f"{internal_api_url}/message",
                    json={
                        "message": task,
                        "system_prompt_suffix": system_prompt,
                        "max_tokens": 8192,
                        "parent_chat_id": self.my_conversation_id,
                        "agent_name": display_name,
                    },
                    timeout=60
                )
                
                if not task_response.is_success:
                    # CLEANUP: Failed to send task
                    await self._cleanup_session(session_id, None, f"Failed to send task: {task_response.text}")
                    return ToolResult(
                        error=f"Failed to send task to sub-agent: {task_response.text}"
                    )
                
                result_data = task_response.json()
                sub_conversation_id = result_data.get("conversation_id")
                logger.info(f"[SubAgentTool] Sub-agent conversation ID: {sub_conversation_id}")
                
                if not wait_for_completion:
                    # NOT WAITING: Don't cleanup - sub-agent is running independently
                    logger.info(f"[SubAgentTool] Not waiting for completion, returning immediately")
                    return ToolResult(
                        output=f"""Sub-agent '{agent_name}' spawned and working.

    Session ID: {session_id}
    Conversation ID: {sub_conversation_id}
    VNC URL: {session.get('vnc_url', 'N/A')}

    The sub-agent is now working independently. Since you chose not to wait:
    - They share your /home/computeruse/project/ workspace - check there for their work
    - You can continue with other tasks in parallel""",
                        system=f"Sub-agent spawned: session={session_id}, conversation={sub_conversation_id}"
                    )
                
                # 5. Poll for completion
                logger.info(f"[SubAgentTool] Polling for completion (timeout: {timeout_minutes} min)...")
                final_result = await self._poll_for_completion(
                    conversation_id=sub_conversation_id,
                    timeout_minutes=timeout_minutes
                )
                
                # 6. Handle timeout vs completion
                if final_result is None:
                    # CLEANUP: Timeout
                    logger.warning(f"[SubAgentTool] Sub-agent timed out")
                    await self._cleanup_session(session_id, sub_conversation_id, f"Timed out after {timeout_minutes} minutes")
                    return ToolResult(
                        output=f"Sub-agent '{agent_name}' timed out after {timeout_minutes} minutes. Check /home/computeruse/project/ for partial results.",
                        system=f"Sub-agent {agent_name} timed out"
                    )
                
                # CLEANUP: Success - just delete pod, status already set by sub-agent
                logger.info(f"[SubAgentTool] Sub-agent completed")
                await self._cleanup_session(session_id)
                
                return ToolResult(
                    output=f"""Sub-agent '{agent_name}' completed their task.

    === SUB-AGENT RESULT ===
    {final_result}
    ========================

    Check /home/computeruse/project/ for any files they created or modified.""",
                    system=f"Sub-agent {agent_name} completed"
                )
                
        except httpx.TimeoutException as e:
            # CLEANUP: HTTP timeout
            logger.exception(f"[SubAgentTool] Timeout: {e}")
            await self._cleanup_session(session_id, sub_conversation_id, f"HTTP timeout: {e}")
            return ToolResult(
                error=f"Sub-agent '{agent_name}' request timed out"
            )
        except Exception as e:
            # CLEANUP: Unexpected error
            logger.exception(f"[SubAgentTool] Error: {e}")
            await self._cleanup_session(session_id, sub_conversation_id, f"Failed: {str(e)}")
            return ToolResult(
                error=f"Sub-agent '{agent_name}' failed: {str(e)}"
            )
    
    async def _wait_for_ready(self, session_id: str, timeout_seconds: int = 120) -> bool:
        """Poll broker until pod is ready."""
        import time
        start = time.time()
        
        async with httpx.AsyncClient() as client:
            while (time.time() - start) < timeout_seconds:
                try:
                    response = await client.get(
                        f"{self.broker_url}/sessions/{session_id}/status",
                        headers={"X-Broker-Token": self.broker_token},
                        timeout=10
                    )
                    
                    if response.is_success:
                        status = response.json()
                        if status.get("status") == "ready" and status.get("ready"):
                            return True
                        if status.get("status") == "failed":
                            return False
                            
                except Exception:
                    pass
                
                await asyncio.sleep(3)
        
        return False
    
    async def _wait_for_api_health(self, api_url: str, timeout_seconds: int = 60) -> bool:
        """Poll sub-agent's API until healthy."""
        import time
        start = time.time()
        
        async with httpx.AsyncClient() as client:
            while (time.time() - start) < timeout_seconds:
                try:
                    response = await client.get(
                        f"{api_url}/health",
                        timeout=5
                    )
                    if response.is_success:
                        return True
                except Exception:
                    pass
                
                await asyncio.sleep(2)
        
        return False
    
    async def _poll_for_completion(
        self, 
        conversation_id: int, 
        timeout_minutes: int
    ) -> Optional[str]:
        """Poll broker for conversation completion status."""
        import time
        start = time.time()
        timeout_seconds = timeout_minutes * 60
        
        async with httpx.AsyncClient() as client:
            while (time.time() - start) < timeout_seconds:
                try:
                    response = await client.get(
                        f"{self.broker_url}/conversations/{conversation_id}",
                        headers={"X-Broker-Token": self.broker_token},
                        timeout=10
                    )
                    
                    if response.is_success:
                        data = response.json()
                        status = data.get("status")
                        logger.debug(f"[SubAgentTool] Conversation {conversation_id} status: {status}")
                        
                        if status == "completed":
                            return data.get("status_message") or "Task completed successfully."
                        
                        if status in ("failed", "cancelled"):
                            return f"Sub-agent {status}: {data.get('status_message', 'Unknown error')}"
                    else:
                        logger.warning(f"[SubAgentTool] Poll response: {response.status_code}")
                            
                except Exception as e:
                    logger.debug(f"[SubAgentTool] Poll error: {e}")
                
                await asyncio.sleep(5)
        
        # Timeout - return special message, caller will handle cleanup
        return None  # Return None to indicate timeout

    async def _cleanup_session(self, session_id: str, conversation_id: Optional[int] = None, reason: str = "cleaned up"):
        """Delete the sub-agent session and update its status."""
        
        # Update conversation status if we have the ID
        if conversation_id:
            try:
                async with httpx.AsyncClient() as client:
                    # Call broker to update status
                    await client.patch(
                        f"{self.broker_url}/conversations/{conversation_id}/status",
                        json={"status": "cancelled", "status_message": reason},
                        headers={"X-Broker-Token": self.broker_token},
                        timeout=10
                    )
                    logger.info(f"[SubAgentTool] Updated conversation {conversation_id} status to cancelled")
            except Exception as e:
                logger.warning(f"[SubAgentTool] Failed to update conversation status: {e}")
        
        # Delete the pod
        try:
            logger.info(f"[SubAgentTool] Deleting session {session_id}")
            async with httpx.AsyncClient() as client:
                await client.delete(
                    f"{self.broker_url}/sessions/{session_id}",
                    headers={"X-Broker-Token": self.broker_token},
                    timeout=10
                )
        except Exception as e:
            logger.warning(f"[SubAgentTool] Cleanup error: {e}")