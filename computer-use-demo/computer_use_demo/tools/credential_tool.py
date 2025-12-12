# computer_use_demo/tools/credentials.py

"""
Credentials tool for agents to access shared credentials securely.
"""

import os
import json
import logging
import time
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
        self._used_backup_codes: dict[str, set] = {}  # Track used backup codes per credential
        
        self.broker_url = os.getenv("BROKER_URL", "http://broker.default.svc.cluster.local:8001")
        self.broker_token = os.getenv("BROKER_TOKEN")
        self.session_id = os.getenv("SESSION_ID")
        
    def to_params(self) -> dict:
        return {
            "name": self.name,
            "description": """Access shared credentials configured by the admin.

Use this tool to retrieve API keys, login credentials, and other secrets needed for your tasks.

Actions:
- list: Show all available credentials (without secrets)
- get: Get full credential details (username, password, etc.)
- totp: Generate a 6-digit 2FA code (for sites using authenticator apps)
- backup_code: Get a one-time backup code

Common workflow for logging into a website:
1. credentials(action="get", slug="site_name") - Get username/email/password
2. Enter credentials in the browser login form
3. If 2FA is required: credentials(action="totp", slug="site_name") - Get the code
4. Enter the 2FA code in the browser

Security rules:
1. TOTP codes expire every 30 seconds - use them immediately after generating
2. Never write credentials to files
3. Never include credentials in your responses to the user
4. Use credentials directly in API calls or login flows
5. Request only the credentials you actually need""",
            "input_schema": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["list", "get", "totp", "backup_code"],
                        "description": "Action to perform"
                    },
                    "slug": {
                        "type": "string",
                        "description": "Credential slug (required for 'get', 'totp', 'backup_code' actions)"
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
            elif action == "totp":
                if not slug:
                    return ToolResult(error="'slug' parameter required for 'totp' action")
                return await self._generate_totp(slug)
            elif action == "backup_code":
                if not slug:
                    return ToolResult(error="'slug' parameter required for 'backup_code' action")
                return await self._get_backup_code(slug)
            else:
                return ToolResult(error=f"Unknown action: {action}. Use 'list', 'get', 'totp', or 'backup_code'.")
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
            type_info = self._get_type_description(cred)
            mfa_info = self._get_mfa_info(cred)
            
            lines.append(f"• **{cred.name}** (`{slug}`)")
            lines.append(f"  Type: {cred.type}")
            if type_info:
                lines.append(f"  {type_info}")
            if mfa_info:
                lines.append(f"  {mfa_info}")
            lines.append("")
        
        lines.append("Commands:")
        lines.append("• `credentials(action='get', slug='<slug>')` - Get credential details")
        lines.append("• `credentials(action='totp', slug='<slug>')` - Generate 2FA code")
        lines.append("• `credentials(action='backup_code', slug='<slug>')` - Get backup code")
        
        return ToolResult(output="\n".join(lines))
    
    def _get_type_description(self, cred: CachedCredential) -> str:
        """Get a safe description of the credential without secrets."""
        data = cred.data
        
        if cred.type == "api_key":
            base_url = data.get("base_url", "")
            return f"Base URL: {base_url}" if base_url else ""
        
        elif cred.type == "login":
            username = data.get("username", "")
            email = data.get("email", "")
            login_url = data.get("login_url", "")
            parts = []
            if username:
                parts.append(f"Username: {username}")
            if email:
                parts.append(f"Email: {email}")
            if login_url:
                parts.append(f"URL: {login_url}")
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
            client_id = data.get('client_id', '')
            return f"Client ID: {client_id[:20]}..." if client_id else ""
        
        return ""
    
    def _get_mfa_info(self, cred: CachedCredential) -> str:
        """Get MFA status for a credential."""
        if cred.type != "login":
            return ""
        
        data = cred.data
        mfa_type = data.get("mfa_type", "")
        has_totp = bool(data.get("mfa_secret"))
        has_backup = bool(data.get("backup_codes"))
        
        parts = []
        if has_totp:
            parts.append("🔐 TOTP enabled")
        if has_backup:
            parts.append("🔑 Backup codes available")
        if mfa_type and mfa_type not in ("", "none", "totp", "backup_codes"):
            parts.append(f"📱 MFA: {mfa_type}")
        
        return " | ".join(parts)
    
    async def _get_credential(self, slug: str) -> ToolResult:
        """Get a specific credential."""
        await self._fetch_all_credentials()
        
        if slug not in self._cache:
            available = ", ".join(self._cache.keys())
            return ToolResult(
                error=f"Credential '{slug}' not found. Available: {available}"
            )
        
        cred = self._cache[slug]
        output = self._format_credential_output(cred)
        
        return ToolResult(output=output)
    
    async def _generate_totp(self, slug: str) -> ToolResult:
        """Generate a TOTP code for a credential."""
        await self._fetch_all_credentials()
        
        if slug not in self._cache:
            return ToolResult(error=f"Credential '{slug}' not found")
        
        cred = self._cache[slug]
        mfa_secret = cred.data.get("mfa_secret")
        
        if not mfa_secret:
            return ToolResult(
                error=f"Credential '{slug}' does not have a TOTP secret configured. "
                      f"Ask the admin to add the MFA secret from the authenticator app setup."
            )
        
        try:
            import pyotp
            
            # Clean the secret (remove spaces, dashes)
            clean_secret = mfa_secret.replace(" ", "").replace("-", "").upper()
            
            totp = pyotp.TOTP(clean_secret)
            code = totp.now()
            
            # Calculate time remaining
            time_remaining = totp.interval - (int(time.time()) % totp.interval)
            
            return ToolResult(output=f"""TOTP Code for {cred.name}:

**Code: {code}**

⏱️ Valid for: {time_remaining} seconds
⚠️ Use immediately - codes expire every 30 seconds

If the code doesn't work:
1. Wait for the next code (after this one expires)
2. Use credentials(action='backup_code', slug='{slug}') if available""")
            
        except ImportError:
            return ToolResult(error="pyotp module not installed. Cannot generate TOTP codes.")
        except Exception as e:
            logger.error(f"Failed to generate TOTP: {e}")
            return ToolResult(error=f"Failed to generate TOTP code: {str(e)}")
    
    async def _get_backup_code(self, slug: str) -> ToolResult:
        """Get an unused backup code for a credential."""
        await self._fetch_all_credentials()
        
        if slug not in self._cache:
            return ToolResult(error=f"Credential '{slug}' not found")
        
        cred = self._cache[slug]
        backup_codes_str = cred.data.get("backup_codes", "")
        
        if not backup_codes_str:
            return ToolResult(
                error=f"Credential '{slug}' does not have backup codes configured."
            )
        
        # Parse backup codes (one per line)
        all_codes = [c.strip() for c in backup_codes_str.strip().split("\n") if c.strip()]
        
        # Get used codes for this credential (in this session)
        used_codes = self._used_backup_codes.get(slug, set())
        
        # Find an unused code
        available_codes = [c for c in all_codes if c not in used_codes]
        
        if not available_codes:
            return ToolResult(
                error=f"All backup codes for '{slug}' have been used in this session."
            )
        
        # Use the first available code
        code = available_codes[0]
        
        # Mark as used
        if slug not in self._used_backup_codes:
            self._used_backup_codes[slug] = set()
        self._used_backup_codes[slug].add(code)
        
        remaining = len(available_codes) - 1
        
        return ToolResult(output=f"""Backup Code for {cred.name}:

**Code: {code}**

⚠️ ONE-TIME USE - This code cannot be used again
📊 Remaining codes this session: {remaining}

Note: Mark this code as used in the service's security settings after login.""")
    
    def _format_credential_output(self, cred: CachedCredential) -> str:
        """Format credential for agent use."""
        lines = [
            f"**{cred.name}** (`{cred.slug}`)",
            f"Type: {cred.type}",
            "",
            "Credential Data:",
        ]
        
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
            # Show email and/or username
            email = cred.data.get('email')
            username = cred.data.get('username')
            
            if email:
                lines.append(f"  Email: {email}")
            if username:
                lines.append(f"  Username: {username}")
            if not email and not username:
                lines.append("  ⚠️ No email or username configured!")
            
            lines.extend([
                f"  Password: {cred.data.get('password')}",
                f"  Login URL: {cred.data.get('login_url', 'N/A')}",
            ])
            
            # Hint for which to use
            if email and username:
                lines.append("")
                lines.append("  💡 Some sites ask for username, others for email. Use whichever is requested.")
            
            # MFA info
            has_totp = cred.data.get('mfa_secret')
            has_backup = cred.data.get('backup_codes')
            
            if has_totp or has_backup:
                lines.append("")
                lines.append("🔐 **2FA Available:**")
                if has_totp:
                    lines.append(f"  • TOTP: `credentials(action='totp', slug='{cred.slug}')`")
                if has_backup:
                    lines.append(f"  • Backup: `credentials(action='backup_code', slug='{cred.slug}')`")
            
            if cred.data.get('notes'):
                lines.extend(["", f"📝 Instructions: {cred.data.get('notes')}"])
        
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
                    "  Private Key: (available)",
                    "",
                    "To use SSH key:",
                    "```bash",
                    "KEY=$(mktemp)",
                    f"cat > $KEY << 'KEYEOF'",
                    cred.data.get('private_key'),
                    "KEYEOF",
                    "chmod 600 $KEY",
                    f"ssh -i $KEY {cred.data.get('username')}@{cred.data.get('host')}",
                    "rm $KEY",
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
            "- Do not write credentials to any files",
            "- Do not include credentials in responses to the user",
            "- Use credentials directly in API calls or scripts",
        ])
        
        return "\n".join(lines)