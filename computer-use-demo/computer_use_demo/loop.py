"""
Agentic sampling loop that calls the Claude API and local implementation of anthropic-defined computer use tools.
"""

import json
import os
import platform
import time
from collections.abc import Callable
from datetime import datetime
from enum import StrEnum
from typing import Any, Optional, cast

import logging

logger = logging.getLogger(__name__)

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

from .tools.subagent import SubAgentTool
from .tools.credential_tool import CredentialsTool

PROMPT_CACHING_BETA_FLAG = "prompt-caching-2024-07-31"


class APIProvider(StrEnum):
    ANTHROPIC = "anthropic"
    BEDROCK = "bedrock"
    VERTEX = "vertex"

S3_BUCKET = os.getenv('S3_BUCKET', 'your-default-bucket')
CHAT_ID = os.getenv('CHAT_ID', 'default-chat-id')
PROJECT_STATUS_PATH = "/home/computeruse/project/project_status.md"
PROJECT_LOG_PATH = "/home/computeruse/project/project_log.md"

def _load_project_memory() -> str:
    """Load current project context - status first, then recent log entries."""
    memory = ""
    
    # Always load status (small, critical)
    if os.path.exists(PROJECT_STATUS_PATH):
        try:
            with open(PROJECT_STATUS_PATH, "r") as f:
                content = f.read()
                memory += f"""
<PROJECT_STATUS>
{content}
</PROJECT_STATUS>
"""
        except Exception as e:
            logger.error(f"Failed to read project status: {e}")
    
    # Load log (may be larger, but important for context)
    if os.path.exists(PROJECT_LOG_PATH):
        try:
            with open(PROJECT_LOG_PATH, "r") as f:
                content = f.read()
                # Optional: truncate to last N entries if file gets huge
                memory += f"""
<PROJECT_LOG>
{content}
</PROJECT_LOG>
"""
        except Exception as e:
            logger.error(f"Failed to read project log: {e}")
    
    return memory

def _ensure_project_dir():
    """Ensure the project directory exists."""
    project_dir = os.path.dirname(PROJECT_STATUS_PATH)
    if not os.path.exists(project_dir):
        os.makedirs(project_dir, exist_ok=True)

def _get_deployment_info() -> str:
    """Get deployment info from environment variables."""
    app_url = os.getenv('APP_URL')
    app_port = os.getenv('APP_PORT', '3000')
    public_url = os.getenv('PUBLIC_URL')
    
    if not app_url:
        return ""
    
    return f"""
<USER_ACCESSIBLE_ENDPOINT>
**MAKING THINGS ACCESSIBLE TO THE USER:**

You can make any HTTP-based service accessible to the user via their browser!

**Your Public Endpoint:** {app_url}
**Port to bind:** {app_port}
**Base Path:** /webapp/

**How it works:**
1. Start any HTTP server on port {app_port} (bind to 0.0.0.0)
2. It becomes instantly accessible at {app_url}
3. User can access it directly in their browser

**CRITICAL: BASE PATH HANDLING**

Your app is served at `/webapp/` - NOT at the root `/`. This means:
- User visits: `{app_url}/page` 
- Your server receives: `/page` (prefix stripped)
- But ALL links/assets in your HTML must account for the `/webapp/` base path!

**THE PROBLEM:**
```html
<!-- WRONG - These will break! -->
<a href="/about">About</a>           → goes to domain.com/about (404!)
<img src="/images/logo.png">         → goes to domain.com/images/logo.png (404!)

<!-- CORRECT - Use relative paths -->
<a href="./about">About</a>          → goes to domain.com/webapp/about
<img src="./images/logo.png">        → goes to domain.com/webapp/images/logo.png
```

**SOLUTION 1: Use HTML Base Tag (Recommended for static sites)**
Add this to your HTML `<head>`:
```html
<head>
    <base href="/webapp/">
    <!-- Now all relative URLs resolve from /webapp/ -->
</head>
```

**SOLUTION 2: Use Relative Paths**
Always use `./` prefix for links and assets:
```html
<a href="./about">About</a>
<link rel="stylesheet" href="./css/style.css">
<script src="./js/app.js"></script>
<img src="./images/logo.png">
```

**SOLUTION 3: Framework-Specific Configuration**

**React (Create React App / Vite):**
```javascript
// vite.config.js
export default defineConfig({{
  base: '/webapp/',
}})

// Or for Create React App, set in package.json:
// "homepage": "/webapp/"
```

**Vue:**
```javascript
// vite.config.js
export default defineConfig({{
  base: '/webapp/',
}})
```

**Next.js:**
```javascript
// next.config.js
module.exports = {{
  basePath: '/webapp',
  assetPrefix: '/webapp/',
}}
```

**Svelte/SvelteKit:**
```javascript
// svelte.config.js
export default {{
  kit: {{
    paths: {{
      base: '/webapp'
    }}
  }}
}}
```

**Flask:**
```python
from flask import Flask
app = Flask(__name__)
app.config['APPLICATION_ROOT'] = '/webapp'

# Or use url_for() which handles this automatically
```

**Express.js:**
```javascript
const express = require('express');
const app = express();

// Mount everything under /webapp awareness
app.use((req, res, next) => {{
  res.locals.basePath = '/webapp';
  next();
}});
```

**SOLUTION 4: JavaScript Router Configuration**

**React Router:**
```javascript
<BrowserRouter basename="/webapp">
  <Routes>
    <Route path="/" element={{<Home />}} />
    <Route path="/about" element={{<About />}} />
  </Routes>
</BrowserRouter>
```

**Vue Router:**
```javascript
const router = createRouter({{
  history: createWebHistory('/webapp/'),
  routes: [...]
}})
```

**IMPORTANT RULES:**
1. ALWAYS use relative paths (`./`) or configure base path
2. NEVER use absolute paths starting with `/` for internal links
3. For static HTML sites, ALWAYS add `<base href="/webapp/">`
4. For frameworks, ALWAYS configure the base path in the config
5. Test navigation after deployment - click through all links!

**CHECKLIST BEFORE DEPLOYMENT:**
- [ ] HTML has `<base href="/webapp/">` OR framework base path configured
- [ ] All `<a href>` links use relative paths or respect base
- [ ] All `<img src>` use relative paths
- [ ] All `<link href>` use relative paths  
- [ ] All `<script src>` use relative paths
- [ ] CSS `url()` references use relative paths
- [ ] JavaScript fetch/API calls use relative paths
- [ ] Router configured with base path (if using SPA)

**QUICK TEST:**
After starting your server, mentally trace each link:
- Does `./about` resolve to `/webapp/about`? ✅
- Does `/about` resolve to `/about`? ❌ (This is wrong!)

**COMMON USE CASES:**

**Static Website / HTML Files:**
```bash
# Add base tag to all HTML files first!
cd /home/computeruse/project/website
python3 -m http.server {app_port} --bind 0.0.0.0 &
echo "View at: {app_url}"
```

**React/Vue/Vite Dev Server:**
```bash
cd /home/computeruse/project/frontend
# Make sure vite.config.js has base: '/webapp/'
npm run dev -- --host 0.0.0.0 --port {app_port} &
echo "Dev server at: {app_url}"
```

**Express/Node.js:**
```bash
cd /home/computeruse/project/app
PORT={app_port} node server.js &
echo "Server running at: {app_url}"
```

**DEBUGGING:**
If links are broken after deployment:
1. Open browser dev tools (F12)
2. Check Network tab for 404 errors
3. Look at the failed URL - does it have `/webapp/`?
4. Fix the source link to use relative path or add base tag
</USER_ACCESSIBLE_ENDPOINT>
"""

# This system prompt is optimized for the Docker environment in this repository and
# specific tool combinations enabled.
# We encourage modifying this system prompt to ensure the model has context for the
# environment it is running in, and to provide any additional information that may be
# helpful for the task at hand.
SYSTEM_PROMPT = """<SYSTEM_CAPABILITY>
* You are utilising an Ubuntu virtual machine using {platform.machine()} architecture with internet access.
* You can feel free to install Ubuntu applications with your bash tool. Use curl instead of wget.
* To open firefox, please just click on the firefox icon.  Note, firefox-esr is what is installed on your system.
* Using bash tool you can start GUI applications, but you need to set export DISPLAY=:1 and use a subshell. For example "(DISPLAY=:1 xterm &)". GUI apps run with bash tool will appear within your desktop environment, but they may take some time to appear. Take a screenshot to confirm it did.
* When using your bash tool with commands that are expected to output very large quantities of text, redirect into a tmp file and use str_replace_based_edit_tool or `grep -n -B <lines before> -A <lines after> <query> <filename>` to confirm output.
* When viewing a page it can be helpful to zoom out so that you can see everything on the page.  Either that, or make sure you scroll down to see everything before deciding something isn't available.
* When using your computer function calls, they take a while to run and send back to you.  Where possible/feasible, try to chain multiple of these calls all into one function calls request.
* The current date is {datetime.today().strftime("%A, %B %-d, %Y")}.
</SYSTEM_CAPABILITY>

<DIRECTORY_STRUCTURE>
Your workspace has six directories with clear purposes:

**/home/computeruse/uploads/** - USER'S INPUT FILES
   - Files the user uploaded for this project
   - When user says "I uploaded a file" or "use the file I gave you" or there are files in /uploads/, check here
   - All team members can access these (read-only)
   - Examples: requirements, data files, images, specifications, reference documents
   - Your starting materials

**/home/computeruse/work/** - YOUR PERSONAL WORKSPACE
   - Your private work-in-progress area
   - Other team members CANNOT see this
   - Persists across your work sessions
   - Use for: research notes, drafts, experiments, personal planning
   - When ready, move polished work to /project/

**/home/computeruse/project/** - TEAM'S SHARED WORKSPACE
   - All team members can see and contribute
   - This is where the actual work happens - both the deliverable AND supporting work
   - Organize in subdirectories that make sense for your task
   
   **Typical organization:**
   - The actual deliverable (website, report, design, whatever you're building)
   - Supporting work (research, planning, documentation, notes)
   - Shared resources (data, images, templates specific to this project)
   
   **Examples:**
   - Web project: /project/webapp/, /project/research/, /project/planning/
   - Report: /project/report.pdf, /project/data_analysis/, /project/sources/
   - Design: /project/designs/, /project/drafts/, /project/client_feedback/
   
   **This is collaborative:**
   - Build upon what other team members created
   - Keep it organized so others can understand your work
   - The deliverable being built here is what the user will receive

**/home/computeruse/library/** - USER'S PERSONAL LIBRARY
   - User's files available across ALL their projects (not just this one)
   - READ-ONLY: you can use but not modify
   - Examples: company logos, brand guidelines, reusable templates
   - Check here if user mentions "my logo", "our brand", "my template" or similar
   - Less common than /uploads/ (which is project-specific)

**/home/computeruse/tools/** - SHARED TOOLS
   - Pre-installed tools available to everyone
   - Explore before creating new tools: ls /home/computeruse/tools/
   - You CAN create new tools in /tools/your_tool_name/
   - You CANNOT modify existing tools (create a new one instead)
   - Each tool has its own subdirectory with README.md

**/home/computeruse/tmp/** - TEMPORARY SCRATCH SPACE
   - Ephemeral - cleared when session ends
   - Use for: downloads, temporary extractions, quick tests, packaging
   - Don't save important work here or work that can be used later as evidence or reference

**WORKFLOW:**
1. User uploads files → /home/computeruse/uploads/
2. You work privately → /home/computeruse/work/ (drafts, experiments)
3. You share with team → /home/computeruse/project/ (the actual deliverable + supporting work)
4. When complete → package from /home/computeruse/project/ and upload to S3

**VISIBILITY:**
- /home/computeruse/uploads/ - All team members can READ
- /home/computeruse/work/ - Only YOU can access
- /home/computeruse/project/ - All team members can READ and WRITE
- /home/computeruse/library/ - All team members can READ
- /home/computeruse/tools/ - All team members can READ and CREATE new tools
- /home/computeruse/tmp/ - Only YOU can access
</DIRECTORY_STRUCTURE>

<CREDENTIALS_ACCESS>
**SHARED CREDENTIALS:**

The admin has configured shared credentials that you can use for various services.
Use the `credentials` tool to access them:

**List available credentials:**
```
credentials(action="list")
```

**Get a specific credential:**
```
credentials(action="get", slug="openai_api")
```

**SECURITY RULES:**
1. NEVER write credentials to files - use them directly in memory
2. NEVER include credentials in your responses to the user
3. NEVER echo or print credentials to stdout in a way user can see
4. Use credentials directly in API calls, scripts, or login flows
5. If a credential doesn't exist, ask the user to have admin add it

**Common credential types:**
- `api_key`: API keys for services (OpenAI, Stripe, etc.)
- `login`: Username/password for websites
- `oauth`: OAuth client credentials
- `smtp`: Email sending credentials
- `database`: Database connection credentials
- `ssh`: SSH access credentials

**Example workflow:**
1. List credentials to see what's available
2. Get the specific credential you need
3. Use it directly in your API call or script
4. Never save the credential to disk
</CREDENTIALS_ACCESS>

<TOOL_SPECIFICATIONS>
You have 3 tools available:

**1. bash** - Execute shell commands
   - Use for: running commands, installing packages, system operations
   - Example: {"command": "ls -la /home/computeruse/project/"}
   - Timeout: 120 seconds. If timeout occurs, restart with {"restart": true}

**2. str_replace_editor** - Create, view, and edit files
   - **create**: Create new file (REQUIRES file_text with complete content)
   - **view**: View file or directory contents
   - **str_replace**: Replace text (old_str must exist exactly once)
   - **insert**: Insert text at line number (requires insert_line and new_str)
   - **undo_edit**: Undo last edit
   - All paths must be absolute (start with /)

**3. computer** - GUI interaction
   - screenshot, mouse_move, left_click, right_click, type, key, scroll

**TOOL SELECTION:**
- Creating/editing files → str_replace_editor (NOT bash heredoc)
- Running commands → bash
- GUI interaction → computer
</TOOL_SPECIFICATIONS>

<TOOL_SELECTION_STRATEGY>
WHEN TO USE EACH TOOL:

BASH TOOL - Use for:
- Running commands (ls, grep, find, etc.)
- Installing packages (apt-get, pip)
- Running scripts that already exist
- Quick one-liners
- System operations
DO NOT USE FOR: Creating large files (use str_replace_editor instead)

STR_REPLACE_EDITOR TOOL - Use for:
- Creating new files with content
- Editing existing files
- Viewing file contents
- ANY file creation task
BEST FOR: Files of any size, especially large files

COMPUTER TOOL - Use for:
- GUI interactions
- Taking screenshots
- Mouse/keyboard control
- Browser automation

CRITICAL: For creating files with content, ALWAYS use str_replace_editor tool, 
NOT bash with heredoc. The str_replace_editor tool is specifically designed 
for this and has no size limits.
</TOOL_SELECTION_STRATEGY>

<TOOL_MANAGEMENT>
**FINDING EXISTING TOOLS:**
* All tools are in /home/computeruse/tools/
* Before creating a new tool, explore existing ones: ls /home/computeruse/tools/
* Each tool has README.md explaining its purpose and how to use it
* Example: cat /home/computeruse/tools/pdf_converter/README.md

**CREATING NEW TOOLS:**
* Create in /home/computeruse/tools/your_tool_name/
* Use lowercase_with_underscores naming
* Must include README.md with: purpose, usage examples, inputs, outputs, requirements
* Make it reusable - other agents might use it

**TOOL RULES:**
1. NEVER modify or delete existing tools
2. If an existing tool almost works but needs changes, create a NEW tool
3. Use descriptive, specific names (pdf_to_text_converter, not converter)
4. Document your tool well in README.md
</TOOL_MANAGEMENT>

<FILE_DELIVERY>
**DELIVERING FILES TO THE USER:**

**CRITICAL: All completed work and deliverables MUST be uploaded to S3 for user download.**

Whenever you create files that the user needs to access, download, or that represent completed project deliverables, you MUST upload them to S3. This includes:
- Reports, documents, PDFs
- Websites, web applications
- Images, videos, media files
- Data files, spreadsheets, databases
- Code projects, applications
- Any packaged deliverables
- Any file the user explicitly requests
- Any completed project output

**Step 1: Package your deliverable**
```bash
# Navigate to your completed work
cd /home/computeruse/project/

# For a single file:
cp final_report.pdf /tmp/deliverable.pdf

# For multiple files/directories - ALWAYS ZIP THEM TOGETHER:
zip -r /tmp/deliverable.zip your_deliverable_folder/

# For websites/webapps:
tar -czf /tmp/website.tar.gz webapp/

# For code projects:
zip -r /tmp/project_name.zip project_folder/ -x "*/node_modules/*" "*/.git/*" "*/venv/*"
```

**Step 2: Upload to S3 (REQUIRED)**
```bash
# Upload your packaged file using environment variables
aws s3 cp /tmp/deliverable.zip \
  s3://""" + S3_BUCKET + """/outputs/""" + CHAT_ID + """/deliverable.zip --acl public-read

# Or for single files:
aws s3 cp /tmp/final_report.pdf \
  s3://""" + S3_BUCKET + """/outputs/""" + CHAT_ID + """/final_report.pdf --acl public-read
```

**Step 3: Provide the download URL to the user**
The download URL format is:
```
https://""" + S3_BUCKET + """.s3.amazonaws.com/outputs/""" + CHAT_ID + """/filename.ext
```

You can construct it in bash:
```bash
echo "Download your file here: https://""" + S3_BUCKET + """.s3.amazonaws.com/outputs/""" + CHAT_ID + """/deliverable.zip"
```

**ENVIRONMENT VARIABLES:**
* `$S3_BUCKET` - The S3 bucket name (automatically set)
* `$CHAT_ID` - The unique chat/session identifier (automatically set)
* These are pre-configured - just use them in your commands
* Verify they're set: `echo $S3_BUCKET` and `echo $CHAT_ID`

**AWS CREDENTIALS - ALREADY CONFIGURED:**
* AWS credentials are automatically configured via IRSA (IAM Roles for Service Accounts)
* Simply use the AWS CLI - authentication is handled automatically
* NEVER run `aws configure`
* NEVER create or modify AWS access keys or secret keys
* NEVER set AWS credential environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
* NEVER create or modify ~/.aws/ directories or config files
* Just use: `aws s3 cp <file> s3://$S3_BUCKET/outputs/$CHAT_ID/<filename> --acl public-read`

**WHEN TO UPLOAD TO S3:**
✓ User explicitly asks for a downloadable file
✓ User says they cannot access the VM
✓ Your task is complete and produces deliverable files
✓ You've created any final output: reports, websites, apps, images, videos, data files
✓ User needs to download or access the completed work
✓ Project is finished and ready for delivery
✓ ANY time you create files intended for the user

**PACKAGING BEST PRACTICES:**
- Use descriptive filenames: `marketing_campaign_q4.zip` not `output.zip`
- Include a README.md if the deliverable is complex or contains multiple files
- Test that all files are included before uploading
- For websites/apps, ensure all dependencies and assets are included
- Remove unnecessary files: node_modules, .git, venv, __pycache__, .DS_Store
- Verify the package can be extracted and used independently

**EXAMPLE WORKFLOW:**
```bash
# 1. Complete your work in /project/
cd /home/computeruse/project/

# 2. Package it appropriately
zip -r /tmp/website_final.zip webapp/ -x "*/node_modules/*"

# 3. Upload to S3 using environment variables
aws s3 cp /tmp/website_final.zip \
  s3://""" + S3_BUCKET + """/outputs/""" + CHAT_ID + """/website_final.zip --acl public-read

# 4. Construct and display the download URL
echo "Your website is ready! Download it here:"
echo "https://""" + S3_BUCKET + """.s3.amazonaws.com/outputs/""" + CHAT_ID + """/website_final.zip"
```

**MULTIPLE DELIVERABLES:**
If you have multiple separate deliverables (report + data + images), **ALWAYS combine them into a single ZIP file**:
```bash
# Create a temporary directory for all deliverables
mkdir -p /tmp/deliverables

# Copy all files to the directory
cp /home/computeruse/project/report.pdf /tmp/deliverables/
cp /home/computeruse/project/data.xlsx /tmp/deliverables/
cp -r /home/computeruse/project/images/ /tmp/deliverables/

# Optionally add a README explaining the contents
cat > /tmp/deliverables/README.txt << 'EOF'
PROJECT DELIVERABLES
===================
- report.pdf: Final project report
- data.xlsx: Analysis data and results
- images/: All project images and graphics
EOF

# Zip everything together
cd /tmp
zip -r project_deliverables.zip deliverables/

# Upload the single ZIP file using environment variables
aws s3 cp /tmp/project_deliverables.zip \
  s3://""" + S3_BUCKET + """/outputs/""" + CHAT_ID + """/project_deliverables.zip --acl public-read

# Display the download URL
echo "All deliverables are ready! Download here:"
echo "https://""" + S3_BUCKET + """.s3.amazonaws.com/outputs/""" + CHAT_ID + """/project_deliverables.zip"
```

Then provide the single download link to the user. This makes it easier for the user to download everything at once.
</FILE_DELIVERY>

<COLLABORATION>
**WORKING WITH YOUR TEAM:**

* **Shared workspace:** /home/computeruse/project/
* **Your private space:** /home/computeruse/work/
* **Team visibility:** Others see everything in /home/computeruse/project/, nothing in /home/computeruse/work/

**Best practices:**
- Check /home/computeruse/project/ to see what team members have already done
- Build upon their work rather than duplicating
- Organize clearly so others can understand and use your work
- Add README files or comments when helpful
- Don't overwrite others' work - communicate through file organization

**Example collaboration:**
Agent 1 creates: /home/computeruse/project/webapp/structure/
Agent 2 adds: /home/computeruse/project/webapp/backend/
Agent 3 adds: /home/computeruse/project/webapp/frontend/
Agent 4 polishes: /home/computeruse/project/webapp/ (final touches)
Final: Package /home/computeruse/project/webapp/ and upload to S3
</COLLABORATION>

<SYSTEM_ACCESS>
* You have full sudo access without requiring a password
* Use sudo for system-level operations: installing packages, fixing permissions, system configuration
* Example: sudo apt-get install <package>, sudo chown -R $USER:$USER <path>
* Make sure to use sudo responsibly and only when necessary
</SYSTEM_ACCESS>

<IMPORTANT_BEHAVIORAL_NOTES>
**Firefox Usage:**
* When Firefox shows a startup wizard, IGNORE it - don't click anything
* Click directly on the address bar where it says "Search or enter address"
* Type your URL or search term there and press Enter

**Reading PDFs:**
* If you need to read an entire PDF (not just view it):
```bash
  curl -o document.pdf https://example.com/document.pdf
  sudo apt-get install -y poppler-utils
  pdftotext document.pdf document.txt
  cat document.txt  # or use str_replace_based_edit_tool
```
* This is much faster and more accurate than taking many screenshots

**Text Input in Forms:**
* After entering text in any form field:
  1. Enter the text
  2. Wait 2 seconds
  3. Take a screenshot to verify text was entered correctly
  4. Check for: typos, extra spaces, missing characters
  5. Only proceed if the text is exactly correct
* This prevents form submission errors and data loss
</IMPORTANT_BEHAVIORAL_NOTES>"""

SYSTEM_PROMPT += f"""

<PROJECT_MEMORY>
You coordinate with other agents using TWO files in /home/computeruse/project/:

## 1. STATUS FILE: {PROJECT_STATUS_PATH}
**Purpose:** Current project state - what exists RIGHT NOW
**Operation:** OVERWRITE (not append)
**When to update:** When you START work and when you FINISH

**Format:**
```markdown
# Project Status
**Last Updated:** [YYYY-MM-DD HH:MM] by [Your Agent Name]
**State:** [Not Started | In Progress | Blocked | Complete]

## Current Deliverables
- [Component]: [Status] ([key files])

## Active Work
[What you're doing right now, or "None"]

## Blocked
[Any blockers, or "None"]

## Next Up
[What should happen next]
```

## 2. LOG FILE: {PROJECT_LOG_PATH}
**Purpose:** Historical record of completed work
**Operation:** APPEND ONLY (never modify previous entries)
**When to update:** When you COMPLETE a task

**Format:**
```markdown
## [Agent-Name] Task Description - YYYY-MM-DD HH:MM
**Delivered:** [1-2 sentence summary of what you built/did]
**Files:** [Full paths to key files created/modified]
**Decisions:** [Key choices you made and why]
**Handoff:** [What the next agent needs to know]

---
```

## YOUR WORKFLOW:
1. **ON START:** Read both files → Update STATUS (claim your work in "Active Work")
2. **DURING WORK:** Focus on your task (no logging needed)
3. **ON FINISH:** Update STATUS (new state, clear Active Work) → Append to LOG (your summary)

## CRITICAL RULES:
- ALWAYS update both files before you finish
- STATUS: Overwrite entire file with current state
- LOG: Only append, never modify existing entries
- Include FULL PATHS to files you create
- Explain your DECISIONS so future agents understand your choices
</PROJECT_MEMORY>
"""


# === COMPLETION PROMPT (used to force logging before exit) ===
LOG_COMPLETION_MESSAGE = """MANDATORY: Update project memory before finishing.

You must update BOTH files now:

**1. STATUS FILE ({status_path})** - OVERWRITE with current state:
```markdown
# Project Status
**Last Updated:** {timestamp} by Agent
**State:** [Complete or current state]

## Current Deliverables
[List what exists now with file paths]

## Active Work
None

## Blocked
None

## Next Up
[What should happen next, or "Project complete"]
```

**2. LOG FILE ({log_path})** - APPEND your entry:
```markdown
## [Agent] Task Description - {timestamp}
**Delivered:** [What you accomplished]
**Files:** [Key files with full paths]
**Decisions:** [Why you made key choices]
**Handoff:** [What next agent needs to know]

---
```

Use str_replace_editor to make these updates NOW."""

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
            "host": host or os.getenv("AGENT_DB_HOST", "localhost"),
            "port": port or int(os.getenv("AGENT_DB_PORT", "3306")),
            "user": user or os.getenv("AGENT_DB_USER", "root"),
            "password": password or os.getenv("AGENT_DB_PASSWORD", ""),
            "db": db or os.getenv("AGENT_DB_NAME", "multiai"),
            "autocommit": True,
        }

        try:
            pool = await aiomysql.create_pool(**connection_params)
            return ConversationStore(pool)
        except Exception as e:
            raise ConnectionError(f"Failed to connect to MySQL: {str(e)}") from e
            
    async def create_conversation(
        self,
        model: str,
        conv_type: str,
        status: str,
        # NEW parameters
        chat_id: Optional[int] = None,           # Laravel chat ID
        session_id: Optional[str] = None,        # K8s session ID
        parent_chat_id: Optional[int] = None,    # Parent computer_use_chats.id
        agent_name: Optional[str] = None,        # Agent identifier
    ) -> int:
        """Create a new conversation record."""
        async with self.pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO computer_use_chats (
                        name, type, status, status_updated_at,
                        chat_id, session_id, parent_chat_id, agent_name,
                        created_at, updated_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        model,
                        conv_type,
                        status,
                        datetime.utcnow(),
                        chat_id,
                        session_id,
                        parent_chat_id,
                        agent_name,
                        datetime.utcnow(),
                        datetime.utcnow(),
                    ),
                )
                await conn.commit()
                return cur.lastrowid
            
    async def get_conversation(self, conversation_id: int) -> Optional[dict]:
        """Get a single conversation by ID."""
        async with self.pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                await cur.execute(
                    """
                    SELECT 
                        id,
                        chat_id,
                        session_id,
                        agent_name,
                        parent_chat_id,
                        name as model,
                        type,
                        status,
                        status_message,
                        created_at,
                        completed_at
                    FROM computer_use_chats
                    WHERE id = %s
                    """,
                    (conversation_id,)
                )
                return await cur.fetchone()

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

    async def update_status(
        self,
        conversation_id: int,
        status: str,
        message: Optional[str] = None,
    ) -> None:
        """Update the status and related fields in the database."""
        now = datetime.utcnow()

        async with self.pool.acquire() as conn:
            async with conn.cursor() as cur:
                query = """
                    UPDATE computer_use_chats
                    SET status = %s, status_updated_at = %s, status_message = %s
                """
                params: list = [status, now, message]

                if status == "completed":
                    query += ", completed_at = %s"
                    params.append(now)

                query += " WHERE id = %s"
                params.append(conversation_id)

                await cur.execute(query, tuple(params))
                await conn.commit()

    async def store_api_request(
        self,
        conversation_id: int,
        iteration: int,
        request_data: dict,
    ) -> int:
        """Store full API request for debugging."""
        async with self.pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO computer_use_api_requests (
                        computer_use_chat_id, iteration, request_data, created_at
                    ) VALUES (%s, %s, %s, %s)
                    """,
                    (
                        conversation_id,
                        iteration,
                        json.dumps(request_data, default=str),
                        datetime.utcnow(),
                    ),
                )
                await conn.commit()
                return cur.lastrowid

    async def update_api_request_response(
        self,
        api_request_id: int,
        response_data: dict,
    ):
        """Update API request row with response data."""
        async with self.pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    UPDATE computer_use_api_requests 
                    SET response_data = %s, response_at = %s
                    WHERE id = %s
                    """,
                    (
                        json.dumps(response_data, default=str),
                        datetime.utcnow(),
                        api_request_id,
                    ),
                )
                await conn.commit()

    async def create_spawn(
        self,
        parent_conversation_id: int,
        chat_id: int,
        user_id: int,
        agent_id: str,
        agent_name: str,
        display_name: Optional[str],
        parent_agent_id: str,
        session_id: str,
        parent_session_id: str,
        system_prompt: str,
        task: str,
        wait_for_completion: bool = True,
        cleanup_on_complete: bool = True,
    ) -> int:
        """Create a spawn record for a sub-agent."""
        async with self.pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute("""
                    INSERT INTO sub_agent_spawns (
                        parent_conversation_id,
                        chat_id,
                        user_id,
                        agent_id,
                        agent_name,
                        display_name,
                        parent_agent_id,
                        session_id,
                        parent_session_id,
                        system_prompt,
                        task,
                        wait_for_completion,
                        cleanup_on_complete,
                        status,
                        created_at,
                        updated_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    parent_conversation_id,
                    chat_id,
                    user_id,
                    agent_id,
                    agent_name,
                    display_name,
                    parent_agent_id,
                    session_id,
                    parent_session_id,
                    system_prompt,
                    task,
                    wait_for_completion,
                    cleanup_on_complete,
                    "spawning",
                    datetime.utcnow(),
                    datetime.utcnow(),
                ))
                await conn.commit()
                return cur.lastrowid
    
    async def update_spawn(
        self,
        spawn_id: int,
        child_conversation_id: Optional[int] = None,
        pod_name: Optional[str] = None,
        status: Optional[str] = None,
        result_summary: Optional[str] = None,
        error_message: Optional[str] = None,
        started_at: Optional[datetime] = None,
        completed_at: Optional[datetime] = None,
    ) -> None:
        """Update a spawn record."""
        updates = []
        values = []
        
        if child_conversation_id is not None:
            updates.append("child_conversation_id = %s")
            values.append(child_conversation_id)
        
        if pod_name is not None:
            updates.append("pod_name = %s")
            values.append(pod_name)
        
        if status is not None:
            updates.append("status = %s")
            values.append(status)
        
        if result_summary is not None:
            updates.append("result_summary = %s")
            values.append(result_summary[:5000])  # Truncate if needed
        
        if error_message is not None:
            updates.append("error_message = %s")
            values.append(error_message[:2000])
        
        if started_at is not None:
            updates.append("started_at = %s")
            values.append(started_at)
        
        if completed_at is not None:
            updates.append("completed_at = %s")
            values.append(completed_at)
        
        if not updates:
            return
        
        updates.append("updated_at = %s")
        values.append(datetime.utcnow())
        values.append(spawn_id)
        
        async with self.pool.acquire() as conn:
            async with conn.cursor() as cur:
                query = f"UPDATE sub_agent_spawns SET {', '.join(updates)} WHERE id = %s"
                await cur.execute(query, tuple(values))
                await conn.commit()
    
    async def get_spawn(self, spawn_id: int) -> Optional[dict]:
        """Get a spawn record."""
        async with self.pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                await cur.execute(
                    "SELECT * FROM sub_agent_spawns WHERE id = %s",
                    (spawn_id,)
                )
                return await cur.fetchone()
    
    async def get_spawns_for_chat(self, chat_id: int) -> list[dict]:
        """Get all spawns for a chat."""
        async with self.pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                await cur.execute(
                    "SELECT * FROM sub_agent_spawns WHERE chat_id = %s ORDER BY created_at ASC",
                    (chat_id,)
                )
                return await cur.fetchall()

def _update_status_file_on_exit(exit_status: str, agent_name: str = "Agent"):
    """Update the status file when agent exits abnormally."""
    _ensure_project_dir()
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M")
    
    try:
        # Read existing status or create new
        existing_content = ""
        if os.path.exists(PROJECT_STATUS_PATH):
            with open(PROJECT_STATUS_PATH, "r") as f:
                existing_content = f.read()
        
        # If agent didn't complete normally, add a warning to status
        if exit_status not in ("completed",):
            warning = f"\n\n> ⚠️ **System Note:** {agent_name} exited with status '{exit_status}' at {timestamp}. Work may be incomplete.\n"
            
            with open(PROJECT_STATUS_PATH, "a") as f:
                f.write(warning)
                
    except Exception as e:
        logger.error(f"Failed to update status file on exit: {e}")


def _append_system_log_entry(exit_status: str, agent_name: str = "Agent"):
    """Append a system entry to the log file when agent exits."""
    _ensure_project_dir()
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M")
    
    try:
        if exit_status == "completed":
            # Agent completed normally - it should have logged already
            # Just add a small system note
            log_entry = f"\n> [SYSTEM] {agent_name} completed successfully at {timestamp}\n"
        elif exit_status in ("cancelled", "paused", "stopping"):
            log_entry = f"""
## [SYSTEM] {agent_name} Interrupted - {timestamp}
**Status:** {exit_status}
**Note:** Agent was interrupted by user. Check status file for current state.

---
"""
        else:
            # Failed or unknown
            log_entry = f"""
## [SYSTEM] {agent_name} Stopped Unexpectedly - {timestamp}
**Status:** {exit_status}
**Note:** Agent did not complete normally. Review recent work and status file.

---
"""
        
        with open(PROJECT_LOG_PATH, "a") as f:
            f.write(log_entry)
            
    except Exception as e:
        logger.error(f"Failed to append system log entry: {e}")


from typing import Optional

class HeartbeatTracker:
    def __init__(self, store: ConversationStore, conv_id: int, min_interval: int = 30):
        self.store = store
        self.conv_id = conv_id
        self.min_interval = min_interval
        self.last_beat: float = 0
    
    async def beat(self, phase: Optional[str] = None) -> None:  # Fixed
        now = time.time()
        if now - self.last_beat >= self.min_interval:
            await self._update(phase)
            self.last_beat = now
    
    async def _update(self, phase: Optional[str] = None) -> None:  # Fixed
        try:
            async with self.store.pool.acquire() as conn:
                async with conn.cursor() as cur:
                    # Only update status_message if phase is provided
                    if phase:
                        await cur.execute(
                            """UPDATE computer_use_chats 
                               SET last_heartbeat_at = %s, 
                                   updated_at = %s,
                                   status_message = %s 
                               WHERE id = %s""",
                            (datetime.utcnow(), datetime.utcnow(), phase, self.conv_id)
                        )
                    else:
                        await cur.execute(
                            """UPDATE computer_use_chats 
                               SET last_heartbeat_at = %s, 
                                   updated_at = %s
                               WHERE id = %s""",
                            (datetime.utcnow(), datetime.utcnow(), self.conv_id)
                        )
                    await conn.commit()
        except Exception as e:
            logger.warning(f"Heartbeat failed: {e}")

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
    max_tokens: int = 32768,
    conversation_store: Optional[ConversationStore] = None,
    current_conversation_id: Optional[int] = None,
    conversation_type: str = "continuous",
    tool_version: ToolVersion,
    thinking_budget: int | None = None,
    token_efficient_tools_beta: bool = False,
    use_extended_context: bool = False,
    agent_name: str = "Agent", 
    cleanup_on_complete: bool = True,
    session_id: Optional[str] = None,
    spawn_id: Optional[int] = None,
):
    """
    Agentic sampling loop for the assistant/tool interaction of computer use.
    """
    logger.info(f"[Loop] Starting sampling_loop for conversation {current_conversation_id}")
    logger.info(f"[Loop] Model: {model}, Provider: {provider}")
    logger.debug(f"[Loop] Tool version: {tool_version}")
    logger.info(f"[Loop] cleanup_on_complete={cleanup_on_complete}, session_id={session_id}")

    # Get broker info for self-cleanup
    broker_url = os.getenv("BROKER_URL", "http://broker:8001")
    broker_token = os.getenv("BROKER_TOKEN", "")

    _ensure_project_dir()

    tool_group = TOOL_GROUPS_BY_VERSION[tool_version]
    subagent_tool = SubAgentTool()
    tool_collection = ToolCollection(
        *(ToolCls() for ToolCls in tool_group.tools),
        subagent_tool,  # Include it during initialization
        CredentialsTool()  # Add CredentialsTool to the collection
    )
    logger.debug(f"[Loop] Tools loaded: {[t.name for t in tool_collection.tools]}")
    
    project_memory = _load_project_memory()
    deployment_info = _get_deployment_info()
    full_system_prompt = f"{SYSTEM_PROMPT}\n{project_memory}\n{deployment_info}\n{system_prompt_suffix}"

    system = BetaTextBlockParam(
        type="text",
        text=full_system_prompt,
    )

    # Create conversation store if not provided
    if conversation_store is None:
        # try:
        logger.debug(f"[Loop] Creating new ConversationStore")
        conversation_store = await ConversationStore.create()
    # except ConnectionError as e:
    #     return messages, None

    if current_conversation_id is None:
        logger.debug(f"[Loop] Creating new conversation")
        current_conversation_id = await conversation_store.create_conversation(
            model=model,
            conv_type=conversation_type,
            status="running",
            agent_name=agent_name,
        )

        logger.info(f"[Loop] Created conversation: {current_conversation_id}")

    heartbeat_tracker = HeartbeatTracker(conversation_store, current_conversation_id)
    await heartbeat_tracker.beat("starting")

    # Set the conversation ID on SubAgentTool so it can track hierarchy
    subagent_tool.my_conversation_id = current_conversation_id
    subagent_tool.conversation_store = conversation_store
    logger.debug(f"[Loop] Set subagent_tool.my_conversation_id = {current_conversation_id}")

    iteration = 0
    prompted_for_log = False  # Track if we've asked agent to log
    final_status = "running"  # Track exit status for finally block

    try:
        while True:
            iteration += 1
            logger.info(f"[Loop] === Iteration {iteration} ===")

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
                        logger.debug(f"[Loop] Current status from DB: {current_status}")

                if current_status == "stopping":
                    logger.info(f"[Loop] Received stop signal, exiting")
                    final_status = "cancelled"
                    await conversation_store.update_status(
                        current_conversation_id,
                        "cancelled",
                        "Task cancelled by user.",
                    )
                    break  # Exit the loop gracefully

                if current_status == "pausing":
                    logger.info(f"[Loop] Received pause signal, exiting")
                    final_status = "paused"
                    await conversation_store.update_status(
                        current_conversation_id,
                        "paused",
                        "Task paused by user.",
                    )
                    break  # Exit the loop gracefully

            except Exception as e:
                logger.exception(f"[Loop] Error checking status: {e}")
                final_status = "failed"
                await conversation_store.update_status(
                    current_conversation_id,
                    "failed",
                    f"Internal error during status check: {e}",
                )
                break

            enable_prompt_caching = False
            betas = [tool_group.beta_flag] if tool_group.beta_flag else []
            if token_efficient_tools_beta:
                betas.append("token-efficient-tools-2025-02-19")
            if use_extended_context:
                betas.append("context-1m-2025-08-07")
            image_truncation_threshold = only_n_most_recent_images or 0

            if provider == APIProvider.ANTHROPIC:
                client = Anthropic(api_key=api_key, max_retries=4, timeout=httpx.Timeout(300.0, connect=10.0))
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
                only_n_most_recent_images = 90
                # Use type ignore to bypass TypedDict check until SDK types are updated
                system["cache_control"] = {"type": "ephemeral"}  # type: ignore

            if only_n_most_recent_images:
                _maybe_filter_to_n_most_recent_images(
                    messages,
                    only_n_most_recent_images,
                    min_removal_threshold=image_truncation_threshold,
                )
            extra_body = {}
            # if thinking_budget:
            #     # Ensure we only send the required fields for thinking
            #     extra_body = {
            #         "thinking": {"type": "enabled", "budget_tokens": thinking_budget}
            #     }

            # Store the last user message into db
            last_user_message = messages[-1]
            if last_user_message["role"] == "user":
                if current_conversation_id is not None:

                    logger.debug(f"[Loop] Storing user message to DB")
                    try:
                        await conversation_store.store_message(
                            conversation_id=current_conversation_id,
                            role="user",
                            content=json.dumps(last_user_message["content"]),
                            raw_content=json.dumps(last_user_message),
                            tool_id="user-input",
                        )
                        logger.debug(f"[Loop] User message stored successfully")
                    except Exception as e:
                        logger.exception(f"[Loop] Failed to store user message: {e}")

            api_request_id: Optional[int] = None

            try:
                request_data = {
                    "model": model,
                    "system_prompt": full_system_prompt,
                    "system_prompt_length": len(full_system_prompt),
                    "messages": messages,
                    "messages_count": len(messages),
                    "tools": tool_collection.to_params(),
                    "betas": betas,
                    "extra_body": extra_body,
                    "max_tokens": max_tokens,
                }
                api_request_id = await conversation_store.store_api_request(
                    conversation_id=current_conversation_id,
                    iteration=iteration,
                    request_data=request_data,
                )
                logger.debug(f"[Loop] API request logged for iteration {iteration}")
            except Exception as e:
                logger.warning(f"[Loop] Failed to log API request: {e}")

            # Call the API
            logger.info(f"[Loop] Calling Claude API...")
            logger.debug(f"[Loop] Messages count: {len(messages)}")

            # we use raw_response to provide debug information to streamlit. Your
            # implementation may be able call the SDK directly with:
            # `response = client.messages.create(...)` instead.
            try:

                await heartbeat_tracker.beat("calling_api")

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

                logger.info(f"[Loop] API call successful")

                api_response_callback(
                    raw_response.http_response.request, raw_response.http_response, None
                )

                await heartbeat_tracker.beat("processing_response")

                response = raw_response.parse()

                try:
                    response_data = {
                        "id": response.id,
                        "model": response.model,
                        "stop_reason": response.stop_reason,
                        "stop_sequence": response.stop_sequence,
                        "content": [block.model_dump() for block in response.content],
                        "usage": {
                            "input_tokens": response.usage.input_tokens,
                            "output_tokens": response.usage.output_tokens,
                        } if response.usage else None,
                    }
                    await conversation_store.update_api_request_response(
                        api_request_id=api_request_id,
                        response_data=response_data,
                    )
                    logger.debug(f"[Loop] API response logged for iteration {iteration}")
                except Exception as e:
                    logger.warning(f"[Loop] Failed to log API response: {e}")

                logger.debug(f"[Loop] Response stop_reason: {response.stop_reason}")
                logger.debug(f"[Loop] Response content blocks: {len(response.content)}")

                response_params = _response_to_params(response)
                response_message: BetaMessageParam = {
                    "role": "assistant",
                    "content": response_params,
                }
                messages.append(response_message)

                if current_conversation_id is not None:
                    logger.debug(f"[Loop] Storing assistant message to DB")
                    try:
                        await conversation_store.store_message(
                            conversation_id=current_conversation_id,
                            role="assistant",
                            content=json.dumps(response_params),
                            tool_id="response",
                            raw_content=json.dumps(response_message),
                        )
                        logger.debug(f"[Loop] Assistant message stored successfully")
                    except Exception as e:
                        logger.exception(f"[Loop] Failed to store assistant message: {e}")

                tool_result_content: list[BetaToolResultBlockParam] = []
                for content_block in response_params:
                    output_callback(content_block)

                    if (
                        isinstance(content_block, dict)
                        and content_block.get("type") == "tool_use"
                    ):
                        # Type narrowing for tool use blocks
                        tool_use_block = cast(BetaToolUseBlockParam, content_block)
                        tool_name = tool_use_block.get("name", "unknown")
                        tool_input = cast(dict[str, Any], tool_use_block.get("input", {}))

                        await heartbeat_tracker.beat(f"executing_tool:{tool_name}")

                        logger.info(f"[Loop] Executing tool: {tool_use_block['name']}")
                        logger.debug(f"[Loop] Tool input: {tool_use_block.get('input', {})}")

                        try:
                            result = await tool_collection.run(
                                name=tool_use_block["name"],
                                tool_input=tool_input,
                            )

                            # --- NEW INTERRUPTION CHECK ---
                            # If the tool realized we are cancelled, it returns this string.
                            # We break the loop so the `finally` block runs and saves the log.
                            if result.output == "CANCELLED_BY_USER":
                                logger.info("[Loop] Tool detected cancellation signal. Exiting.")
                                # We set status so the finally block logs it correctly
                                final_status = "cancelled"
                                break 
                            # ------------------------------
                        except Exception as tool_error:
                            # Don't crash the loop - return error to Claude so it can retry
                            logger.error(f"[Loop] Tool execution error: {tool_error}")
                            result = ToolResult(error=f"Tool execution failed: {str(tool_error)}")
                        
                        tool_result_content.append(
                            _make_api_tool_result(result, tool_use_block["id"])
                        )
                        tool_output_callback(result, tool_use_block["id"])

                await heartbeat_tracker.beat("iteration_complete")

                if not tool_result_content:
                    logger.info(f"[Loop] No tool calls, task complete!")

                    if not prompted_for_log:
                        # First time seeing no tool calls - prompt agent to log
                        prompted_for_log = True
                        logger.info("[Loop] Prompting agent to update project memory before completion")
                        
                        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M")
                        log_prompt_text = LOG_COMPLETION_MESSAGE.format(
                            status_path=PROJECT_STATUS_PATH,
                            log_path=PROJECT_LOG_PATH,
                            timestamp=timestamp,
                        )
                        
                        log_prompt: BetaMessageParam = {
                            "role": "user",
                            "content": [{"type": "text", "text": log_prompt_text}]
                        }
                        messages.append(log_prompt)

                        if current_conversation_id is not None:
                            try:
                                await conversation_store.store_message(
                                    conversation_id=current_conversation_id,
                                    role="user",
                                    content=json.dumps(log_prompt["content"]),
                                    raw_content=json.dumps(log_prompt),
                                    tool_id="system-log-prompt",
                                )
                            except Exception as e:
                                logger.exception(f"[Loop] Failed to store log prompt: {e}")
                        
                        # Continue to next iteration to let agent respond
                        continue
                    
                    # Already prompted - agent is truly done
                    logger.info(f"[Loop] No tool calls after log prompt, task complete!")
                    final_status = "completed"

                    if current_conversation_id is not None:
                        await conversation_store.update_status(
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
                    logger.debug(f"[Loop] Storing tool results to DB")
                    try:
                        await conversation_store.store_message(
                            conversation_id=current_conversation_id,
                            role="tool",
                            content=json.dumps(tool_result_content),
                            tool_id="response",
                            raw_content=json.dumps(result_message),
                        )
                    except Exception as e:
                        logger.exception(f"[Loop] Failed to store tool results: {e}")

                messages.append(result_message)
                logger.debug(f"[Loop] Continuing to next iteration...")

            except (APIStatusError, APIResponseValidationError) as e:
                error_message = f"API Error: {e}"
                logger.exception(f"[Loop] {error_message}")
                await conversation_store.update_status(
                    current_conversation_id, "failed", error_message
                )
                api_response_callback(e.request, e.response, e)
                return messages, current_conversation_id
            except APIError as e:
                error_message = f"API Error: {e}"
                logger.exception(f"[Loop] {error_message}")
                await conversation_store.update_status(
                    current_conversation_id, "failed", error_message
                )
                api_response_callback(e.request, e.body, e)
                return messages, current_conversation_id
            except Exception as e:
                error_message = f"An unexpected error occurred: {e}"
                logger.exception(f"[Loop] {error_message}")
                try:
                    # Update status
                    await conversation_store.update_status(
                        current_conversation_id, "failed", error_message
                    )
                    
                    # Store the error as a message so user can see what happened
                    await conversation_store.store_message(
                        conversation_id=current_conversation_id,
                        role="system",
                        content=json.dumps([{"type": "error", "text": error_message}]),
                        raw_content=json.dumps({"role": "system", "content": error_message}),
                        tool_id="system-error",
                        is_error=True,
                    )
                except Exception as status_error:
                    logger.error(f"[Loop] Failed to update status/store error: {status_error}")
                
                return messages, current_conversation_id
    finally:
        # === ALWAYS UPDATE PROJECT MEMORY ON EXIT ===
        logger.info(f"[Loop] Finally block - exit status: {final_status}")
        
        # Update status file if abnormal exit
        _update_status_file_on_exit(final_status, agent_name)
        
        # Append to log file
        _append_system_log_entry(final_status, agent_name)

        # ═══════════════════════════════════════════════════════════════
        # Update spawn record if this is a sub-agent
        # ═══════════════════════════════════════════════════════════════
        if spawn_id and conversation_store:
            try:
                # Get the final response to use as result_summary
                final_summary = None
                if messages:
                    # Get the last assistant message as summary
                    for msg in reversed(messages):
                        if msg.get("role") == "assistant":
                            content = msg.get("content", [])
                            if isinstance(content, list):
                                for block in content:
                                    if isinstance(block, dict) and block.get("type") == "text":
                                        final_summary = block.get("text", "")[:5000]
                                        break
                            break
                
                # Map final_status to spawn status
                spawn_status = {
                    "completed": "completed",
                    "cancelled": "cancelled",
                    "paused": "cancelled",
                    "failed": "failed",
                }.get(final_status, "failed")
                
                await conversation_store.update_spawn(
                    spawn_id,
                    status=spawn_status,
                    result_summary=final_summary,
                    completed_at=datetime.utcnow(),
                )
                logger.info(f"[Loop] Updated spawn record {spawn_id} to {spawn_status}")
            except Exception as e:
                logger.error(f"[Loop] Failed to update spawn record: {e}")

        if cleanup_on_complete and session_id:
            logger.info(f"[Loop] Self-cleanup enabled, deleting session {session_id}")
            await _self_cleanup(broker_url, broker_token, session_id, final_status)
        else:
            logger.info(f"[Loop] Self-cleanup disabled or no session_id, pod will stay running")

async def _self_cleanup(broker_url: str, broker_token: str, session_id: str, reason: str):
    """
    Sub-agent cleans up its own pod by calling the broker.
    This is fire-and-forget - we're about to terminate anyway.
    """
    try:
        logger.info(f"[Loop] Requesting self-cleanup: {session_id}")
        
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.delete(
                f"{broker_url}/sessions/{session_id}",
                headers={"X-Broker-Token": broker_token}
            )
            
            if response.is_success:
                logger.info(f"[Loop] Self-cleanup successful")
            else:
                logger.warning(f"[Loop] Self-cleanup failed: {response.status_code} - {response.text}")
                
    except Exception as e:
        # Don't crash - we're cleaning up anyway
        logger.warning(f"[Loop] Self-cleanup error (non-fatal): {e}")

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
            res.append({
                "type": "tool_use",
                "id": block.id,
                "name": block.name,
                "input": block.input,
            })
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
