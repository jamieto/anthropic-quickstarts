# computer_use_demo/tools/credentials.py

"""
Credentials tool for agents to access shared credentials securely.
"""

import os
import json
import logging
from typing import Any, Optional
from dataclasses import dataclass

import httpx

from .base import BaseAnthropicTool, ToolError, ToolResult

logger = logging.getLogger(__name__)


@dataclass
class CachedCredential:
    """In-memory cached credential"""
    slug: str
    name: str
    type: str
    data: dict
    metadata: Optional[dict] = None


class CredentialsTool(BaseAnthropicTool):
    """
    Tool for agents to access shared credentials.
    
    Credentials are fetched from the broker API and cached in memory.
    They are NEVER written to disk.
    """
    
    name = "credentials"
    api_type = "custom"
    
    def __init__(self):
        self._cache: dict[str, CachedCredential] = {}
        self._cache_loaded = False
        
        self.broker_url = os.getenv("BROKER_URL", "http://broker.default.svc.cluster.local:8001")
        self.broker_token = os.getenv("BROKER_TOKEN")
        self.session_id = os.getenv("SESSION_ID")
        
    def to_params(self) -> dict:
        return {
            "name": self.name,
            "description": """Access shared credentials configured by the admin.

Use this tool to retrieve API keys, login credentials, and other secrets needed for your tasks.

Actions:
- list: Show all available credentials (names and types only, not the actual secrets)
- get: Retrieve a specific credential by its slug

The credentials are securely stored and you should:
1. Never write credentials to files
2. Never include credentials in your responses to the user
3. Use credentials directly in API calls or login flows
4. Request only the credentials you actually need""",
            "input_schema": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["list", "get"],
                        "description": "Action to perform"
                    },
                    "slug": {
                        "type": "string",
                        "description": "Credential slug (required for 'get' action)"
                    }
                },
                "required": ["action"]
            }
        }
    
    async def __call__(
        self,
        action: str,
        slug: Optional[str] = None,
        **kwargs
    ) -> ToolResult:
        """Execute credential operations."""
        
        if not self.session_id:
            return ToolResult(error="SESSION_ID not configured - cannot access credentials")
        
        if not self.broker_token:
            return ToolResult(error="BROKER_TOKEN not configured - cannot access credentials")
        
        try:
            if action == "list":
                return await self._list_credentials()
            elif action == "get":
                if not slug:
                    return ToolResult(error="'slug' parameter required for 'get' action")
                return await self._get_credential(slug)
            else:
                return ToolResult(error=f"Unknown action: {action}. Use 'list' or 'get'.")
        except Exception as e:
            logger.exception(f"Credential operation failed: {e}")
            return ToolResult(error=f"Credential operation failed: {str(e)}")
    
    async def _fetch_all_credentials(self) -> None:
        """Fetch all credentials from broker and cache them."""
        if self._cache_loaded:
            return
            
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{self.broker_url}/credentials",
                headers={
                    "X-Broker-Token": self.broker_token,
                    "X-Session-ID": self.session_id,
                }
            )
            
            if response.status_code != 200:
                raise ToolError(f"Failed to fetch credentials: {response.text}")
            
            data = response.json()
            
            for cred in data.get("credentials", []):
                self._cache[cred["slug"]] = CachedCredential(
                    slug=cred["slug"],
                    name=cred["name"],
                    type=cred["type"],
                    data=cred["data"],
                    metadata=cred.get("metadata")
                )
            
            self._cache_loaded = True
            logger.info(f"Cached {len(self._cache)} credentials")
    
    async def _list_credentials(self) -> ToolResult:
        """List available credentials (without revealing secrets)."""
        await self._fetch_all_credentials()
        
        if not self._cache:
            return ToolResult(output="No credentials available.")
        
        lines = ["Available credentials:\n"]
        
        for slug, cred in self._cache.items():
            # Show type-specific info without secrets
            type_info = self._get_type_description(cred)
            lines.append(f"• **{cred.name}** (`{slug}`)")
            lines.append(f"  Type: {cred.type}")
            if type_info:
                lines.append(f"  {type_info}")
            lines.append("")
        
        lines.append("\nUse `credentials(action='get', slug='<slug>')` to retrieve a credential.")
        
        return ToolResult(output="\n".join(lines))
    
    def _get_type_description(self, cred: CachedCredential) -> str:
        """Get a safe description of the credential without secrets."""
        data = cred.data
        
        if cred.type == "api_key":
            base_url = data.get("base_url", "")
            return f"Base URL: {base_url}" if base_url else ""
        
        elif cred.type == "login":
            username = data.get("username", "")
            login_url = data.get("login_url", "")
            parts = []
            if username:
                parts.append(f"Username: {username}")
            if login_url:
                parts.append(f"Login URL: {login_url}")
            return ", ".join(parts)
        
        elif cred.type == "smtp":
            host = data.get("host", "")
            port = data.get("port", "")
            return f"Server: {host}:{port}" if host else ""
        
        elif cred.type == "database":
            host = data.get("host", "")
            database = data.get("database", "")
            driver = data.get("driver", "")
            return f"{driver}://{host}/{database}" if host else ""
        
        elif cred.type == "oauth":
            return f"Client ID: {data.get('client_id', 'N/A')[:20]}..."
        
        return ""
    
    async def _get_credential(self, slug: str) -> ToolResult:
        """Get a specific credential."""
        await self._fetch_all_credentials()
        
        if slug not in self._cache:
            available = ", ".join(self._cache.keys())
            return ToolResult(
                error=f"Credential '{slug}' not found. Available: {available}"
            )
        
        cred = self._cache[slug]
        
        # Format the credential data based on type
        output = self._format_credential_output(cred)
        
        return ToolResult(output=output)
    
    def _format_credential_output(self, cred: CachedCredential) -> str:
        """Format credential for agent use."""
        lines = [
            f"**{cred.name}** (`{cred.slug}`)",
            f"Type: {cred.type}",
            "",
            "Credential Data:",
        ]
        
        # Format based on type with usage hints
        if cred.type == "api_key":
            lines.extend([
                f"  API Key: {cred.data.get('api_key')}",
                f"  Base URL: {cred.data.get('base_url', 'N/A')}",
                f"  Header: {cred.data.get('header_name', 'Authorization')}: {cred.data.get('header_prefix', 'Bearer')} <key>",
                "",
                "Usage example:",
                "```bash",
                f"curl -H \"{cred.data.get('header_name', 'Authorization')}: {cred.data.get('header_prefix', 'Bearer')} {cred.data.get('api_key')}\" \\",
                f"     {cred.data.get('base_url', 'https://api.example.com')}/endpoint",
                "```"
            ])
        
        elif cred.type == "login":
            lines.extend([
                f"  Username: {cred.data.get('username')}",
                f"  Password: {cred.data.get('password')}",
                f"  Login URL: {cred.data.get('login_url', 'N/A')}",
            ])
            if cred.data.get('mfa_secret'):
                lines.append(f"  MFA Secret: {cred.data.get('mfa_secret')}")
                lines.extend([
                    "",
                    "Note: For MFA, generate TOTP code using:",
                    "```python",
                    "import pyotp",
                    f"totp = pyotp.TOTP('{cred.data.get('mfa_secret')}')",
                    "code = totp.now()",
                    "```"
                ])
            if cred.data.get('notes'):
                lines.extend(["", f"Login instructions: {cred.data.get('notes')}"])
        
        elif cred.type == "smtp":
            lines.extend([
                f"  Host: {cred.data.get('host')}",
                f"  Port: {cred.data.get('port')}",
                f"  Username: {cred.data.get('username')}",
                f"  Password: {cred.data.get('password')}",
                f"  Encryption: {cred.data.get('encryption', 'tls')}",
                f"  From: {cred.data.get('from_name', '')} <{cred.data.get('from_address', '')}>",
            ])
        
        elif cred.type == "database":
            driver = cred.data.get('driver', 'mysql')
            lines.extend([
                f"  Driver: {driver}",
                f"  Host: {cred.data.get('host')}",
                f"  Port: {cred.data.get('port')}",
                f"  Database: {cred.data.get('database')}",
                f"  Username: {cred.data.get('username')}",
                f"  Password: {cred.data.get('password')}",
                "",
                "Connection string:",
                f"  {driver}://{cred.data.get('username')}:{cred.data.get('password')}@{cred.data.get('host')}:{cred.data.get('port')}/{cred.data.get('database')}"
            ])
        
        elif cred.type == "ssh":
            lines.extend([
                f"  Host: {cred.data.get('host')}",
                f"  Port: {cred.data.get('port', 22)}",
                f"  Username: {cred.data.get('username')}",
            ])
            if cred.data.get('password'):
                lines.append(f"  Password: {cred.data.get('password')}")
            if cred.data.get('private_key'):
                lines.extend([
                    "  Private Key: (use key below)",
                    "",
                    "To use this SSH key:",
                    "```bash",
                    "# Save key to temp file (in memory)",
                    "KEY=$(mktemp)",
                    f"cat > $KEY << 'KEYEOF'",
                    cred.data.get('private_key'),
                    "KEYEOF",
                    "chmod 600 $KEY",
                    f"ssh -i $KEY {cred.data.get('username')}@{cred.data.get('host')}",
                    "rm $KEY  # Clean up",
                    "```"
                ])
        
        elif cred.type == "oauth":
            lines.extend([
                f"  Client ID: {cred.data.get('client_id')}",
                f"  Client Secret: {cred.data.get('client_secret')}",
                f"  Token URL: {cred.data.get('token_url', 'N/A')}",
                f"  Authorize URL: {cred.data.get('authorize_url', 'N/A')}",
                f"  Scopes: {cred.data.get('scopes', 'N/A')}",
            ])
            if cred.data.get('access_token'):
                lines.append(f"  Access Token: {cred.data.get('access_token')}")
            if cred.data.get('refresh_token'):
                lines.append(f"  Refresh Token: {cred.data.get('refresh_token')}")
        
        else:
            # Custom or unknown type - show raw data
            for key, value in cred.data.items():
                lines.append(f"  {key}: {value}")
        
        lines.extend([
            "",
            "⚠️ SECURITY REMINDER:",
            "- Do not write these credentials to any files",
            "- Do not include credentials in responses to the user",
            "- Use credentials directly in API calls or scripts",
        ])
        
        return "\n".join(lines)