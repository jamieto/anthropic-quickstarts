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

PROMPT_CACHING_BETA_FLAG = "prompt-caching-2024-07-31"


class APIProvider(StrEnum):
    ANTHROPIC = "anthropic"
    BEDROCK = "bedrock"
    VERTEX = "vertex"

S3_BUCKET = os.getenv('S3_BUCKET', 'your-default-bucket')
CHAT_ID = os.getenv('CHAT_ID', 'default-chat-id')

# This system prompt is optimized for the Docker environment in this repository and
# specific tool combinations enabled.
# We encourage modifying this system prompt to ensure the model has context for the
# environment it is running in, and to provide any additional information that may be
# helpful for the task at hand.
SYSTEM_PROMPT = """<SYSTEM_CAPABILITY>
* You are utilising an Ubuntu virtual machine using {platform.machine()} architecture with internet access.
* You can install Ubuntu applications with your bash tool. Use curl instead of wget.
* To open Firefox, click on the Firefox icon. Note: firefox-esr is installed on your system.
* Using bash tool you can start GUI applications, but you need to set export DISPLAY=:1 and use a subshell. For example "(DISPLAY=:1 xterm &)". GUI apps may take time to appear. Take a screenshot to confirm.
* When using bash commands that output very large quantities of text, redirect into a tmp file and use str_replace_based_edit_tool or `grep -n -B <lines before> -A <lines after> <query> <filename>` to confirm output.
* When viewing a page it can be helpful to zoom out so that you can see everything on the page. Make sure you scroll down to see everything before deciding something isn't available.
* When using computer function calls, they take a while to run and send back to you. Where possible/feasible, try to chain multiple of these calls all into one function calls request.
* The current date is """ + datetime.today().strftime('%A, %B %-d, %Y') + """.
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
1. User uploads files → /uploads/
2. You work privately → /work/ (drafts, experiments)
3. You share with team → /project/ (the actual deliverable + supporting work)
4. When complete → package from /project/ and upload to S3

**VISIBILITY:**
- /uploads/ - All team members can READ
- /work/ - Only YOU can access
- /project/ - All team members can READ and WRITE
- /library/ - All team members can READ
- /tools/ - All team members can READ and CREATE new tools
- /tmp/ - Only YOU can access
</DIRECTORY_STRUCTURE>

<TOOL_SPECIFICATIONS>
YOU HAVE 3 TOOLS AVAILABLE. HERE IS EXACTLY HOW TO USE EACH ONE:

================================================================================
1. BASH TOOL - Execute shell commands
================================================================================

REQUIRED PARAMETERS:
- command (string): The bash command to execute

OPTIONAL PARAMETERS:
- restart (bool): Set to true to restart the bash session

CORRECT USAGE:
{
  "name": "bash",
  "input": {
    "command": "ls -la /home/computeruse/project/"
  }
}

{
  "name": "bash",
  "input": {
    "command": "cat /home/computeruse/uploads/data.csv | head -n 10"
  }
}

{
  "name": "bash",
  "input": {
    "command": "pip install pandas --break-system-packages"
  }
}

{
  "name": "bash",
  "input": {
    "restart": true
  }
}

WRONG - WILL FAIL:
{
  "name": "bash",
  "input": {}
}
Error: "no command provided."

 BASH TIMEOUT HANDLING:
- Bash commands have a 120-second timeout
- If a command times out, you MUST restart bash before continuing:
  {"name": "bash", "input": {"restart": true}}
- For large file creation, use str_replace_editor instead of bash heredoc
- If you must use bash heredoc, keep content under 5000 lines

================================================================================
2. STR_REPLACE_EDITOR TOOL - Create, view, and edit files
================================================================================

This tool has 5 commands. Each requires different parameters:

--- COMMAND: "create" ---
Creates a new file. MUST include complete file content. USE THIS FOR ALL FILE CREATION, especially large files.

PREFERRED OVER: bash heredoc for any file with content
ADVANTAGE: No size limits, no timeout issues, designed for this purpose
REQUIRED: command, path, file_text

Example:
{
  "name": "str_replace_editor",
  "input": {
    "command": "create",
    "path": "/home/computeruse/project/script.py",
    "file_text": "#!/usr/bin/env python3\n\nimport os\nimport sys\n\ndef main():\n    print('Hello World')\n\nif __name__ == '__main__':\n    main()\n"
  }
}

Example - Creating a large HTML file:
{
  "name": "str_replace_editor",
  "input": {
    "command": "create",
    "path": "/home/computeruse/project/presentation.html",
    "file_text": "<!DOCTYPE html>\n<html>\n[thousands of lines of HTML]\n</html>"
  }
}

COMMON MISTAKE - DO NOT DO THIS:
{
  "name": "str_replace_editor",
  "input": {
    "command": "create",
    "path": "/home/computeruse/project/script.py"
  }
}
Error: "Parameter `file_text` is required for command: create"
FIX: You MUST include file_text with complete file contents

--- COMMAND: "view" ---
View file contents or list directory contents.

REQUIRED: command, path
OPTIONAL: view_range (list of 2 integers [start_line, end_line])

Examples:
{
  "name": "str_replace_editor",
  "input": {
    "command": "view",
    "path": "/home/computeruse/project/script.py"
  }
}

{
  "name": "str_replace_editor",
  "input": {
    "command": "view",
    "path": "/home/computeruse/project/script.py",
    "view_range": [1, 50]
  }
}

--- COMMAND: "str_replace" ---
Replace exact text in a file. The old_str MUST appear exactly once.

REQUIRED: command, path, old_str
OPTIONAL: new_str (defaults to empty string for deletion)

Example:
{
  "name": "str_replace_editor",
  "input": {
    "command": "str_replace",
    "path": "/home/computeruse/project/script.py",
    "old_str": "print('Hello World')",
    "new_str": "print('Hello Universe')"
  }
}

--- COMMAND: "insert" ---
Insert text at a specific line number.

REQUIRED: command, path, insert_line, new_str

Example:
{
  "name": "str_replace_editor",
  "input": {
    "command": "insert",
    "path": "/home/computeruse/project/script.py",
    "insert_line": 5,
    "new_str": "# This is a new comment\n"
  }
}

--- COMMAND: "undo_edit" ---
Undo the last edit to a file.

REQUIRED: command, path

Example:
{
  "name": "str_replace_editor",
  "input": {
    "command": "undo_edit",
    "path": "/home/computeruse/project/script.py"
  }
}

IMPORTANT NOTES FOR STR_REPLACE_EDITOR:
- Paths MUST be absolute (start with /)
- For create: file_text is ALWAYS required, even for empty files
- For str_replace: old_str must appear exactly once in the file
- For large files, include complete content - do not truncate

================================================================================
3. COMPUTER TOOL - Control mouse, keyboard, take screenshots
================================================================================

Common actions:
- screenshot: Take a screenshot
- mouse_move: Move cursor to coordinate [x, y]
- left_click, right_click, double_click: Click actions
- type: Type text
- key: Press keyboard keys (e.g., "Return", "ctrl+c")
- scroll: Scroll with scroll_direction ("up"/"down") and scroll_amount

All coordinates are [x, y] format. See error messages for required parameters.

================================================================================
CRITICAL RULES - READ CAREFULLY
================================================================================

1. NEVER call a tool with empty input: {}
   Every tool call MUST have parameters.
   Check the examples above for correct format.

2. BASH TOOL MUST have "command" parameter
   Even for simple commands: {"command": "ls"}
   WRONG: {} or {"input": {}}

3. STR_REPLACE_EDITOR "create" MUST have "file_text" parameter
   Include complete file contents.
   WRONG: {"command": "create", "path": "..."} without file_text

4. READ ERROR MESSAGES AND FIX THEM
   - If you get "no command provided" -> add the command parameter
   - If you get "file_text is required" -> add file_text to your create call
   - If you get "old_str is required" -> add old_str to your str_replace call
   - DO NOT repeat the same mistake

5. PARAMETER TYPES MATTER
   - coordinate: [int, int] example: [500, 300]
   - view_range: [int, int] example: [1, 100]
   - scroll_amount: int example: 5
   - duration: float example: 1.5

6. ABSOLUTE PATHS REQUIRED FOR FILE OPERATIONS
   CORRECT: /home/computeruse/project/file.py
   WRONG: project/file.py (missing leading /)

7. FILE CREATION STRATEGY
   - For large files: Include complete content in file_text
   - If hitting token limits: Use bash with heredoc instead:
     {"command": "cat > /path/file.py << 'EOF'\n[complete content]\nEOF"}

================================================================================
ERROR RECOVERY
================================================================================

If you receive an error, follow these steps:
1. READ the error message carefully
2. CHECK the examples above for the correct format
3. IDENTIFY what parameter you forgot or got wrong
4. MAKE the correct call with ALL required parameters
5. DO NOT repeat the same call that just failed

Common error patterns and fixes:

Error: "no command provided"
Fix: Add the command parameter to bash tool
{
  "name": "bash",
  "input": {
    "command": "your_bash_command_here"
  }
}

Error: "Parameter `file_text` is required for command: create"
Fix: Add file_text parameter with complete file contents
{
  "name": "str_replace_editor",
  "input": {
    "command": "create",
    "path": "/absolute/path/to/file",
    "file_text": "complete file contents here"
  }
}

Error: "coordinate is required for mouse_move"
Fix: Add coordinate parameter as a list of two integers
{
  "name": "computer",
  "input": {
    "action": "mouse_move",
    "coordinate": [x, y]
  }
}

Error: "old_str is required for str_replace"
Fix: Add old_str parameter
{
  "name": "str_replace_editor",
  "input": {
    "command": "str_replace",
    "path": "/path/to/file",
    "old_str": "text to find",
    "new_str": "replacement text"
  }
}

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

* **Shared workspace:** /project/
* **Your private space:** /work/
* **Team visibility:** Others see everything in /project/, nothing in /work/

**Best practices:**
- Check /project/ to see what team members have already done
- Build upon their work rather than duplicating
- Organize clearly so others can understand and use your work
- Add README files or comments when helpful
- Don't overwrite others' work - communicate through file organization

**Example collaboration:**
Agent 1 creates: /project/webapp/structure/
Agent 2 adds: /project/webapp/backend/
Agent 3 adds: /project/webapp/frontend/
Agent 4 polishes: /project/webapp/ (final touches)
Final: Package /project/webapp/ and upload to S3
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
    use_extended_context: bool = False,
):
    """
    Agentic sampling loop for the assistant/tool interaction of computer use.
    """
    logger.info(f"[Loop] Starting sampling_loop for conversation {current_conversation_id}")
    logger.info(f"[Loop] Model: {model}, Provider: {provider}")
    logger.debug(f"[Loop] Tool version: {tool_version}")

    tool_group = TOOL_GROUPS_BY_VERSION[tool_version]
    tool_collection = ToolCollection(*(ToolCls() for ToolCls in tool_group.tools))
    logger.debug(f"[Loop] Tools loaded: {[t.name for t in tool_collection.tools]}")

    tool_group = TOOL_GROUPS_BY_VERSION[tool_version]
    subagent_tool = SubAgentTool()
    tool_collection = ToolCollection(
        *(ToolCls() for ToolCls in tool_group.tools),
        subagent_tool  # Include it during initialization
    )
    logger.debug(f"[Loop] SubAgentTool added")
    

    system = BetaTextBlockParam(
        type="text",
        text=f"{SYSTEM_PROMPT}{' ' + system_prompt_suffix if system_prompt_suffix else ''}",
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
        )

        logger.info(f"[Loop] Created conversation: {current_conversation_id}")

    # Set the conversation ID on SubAgentTool so it can track hierarchy
    subagent_tool.my_conversation_id = current_conversation_id
    logger.debug(f"[Loop] Set subagent_tool.my_conversation_id = {current_conversation_id}")

    iteration = 0

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
                await _update_status(
                    conversation_store,
                    current_conversation_id,
                    "cancelled",
                    "Task cancelled by user.",
                )
                break  # Exit the loop gracefully

            if current_status == "pausing":
                logger.info(f"[Loop] Received pause signal, exiting")
                await _update_status(
                    conversation_store,
                    current_conversation_id,
                    "paused",
                    "Task paused by user.",
                )
                break  # Exit the loop gracefully

        except Exception as e:
            logger.exception(f"[Loop] Error checking status: {e}")
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
        if use_extended_context:
            betas.append("context-1m-2025-08-07")
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
        if thinking_budget:
            # Ensure we only send the required fields for thinking
            extra_body = {
                "thinking": {"type": "enabled", "budget_tokens": thinking_budget}
            }

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

        # Call the API
        logger.info(f"[Loop] Calling Claude API...")
        logger.debug(f"[Loop] Messages count: {len(messages)}")

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

            logger.info(f"[Loop] API call successful")

            api_response_callback(
                raw_response.http_response.request, raw_response.http_response, None
            )

            response = raw_response.parse()
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
                    logger.info(f"[Loop] Executing tool: {tool_use_block['name']}")
                    logger.debug(f"[Loop] Tool input: {tool_use_block.get('input', {})}")

                    try:
                        result = await tool_collection.run(
                            name=tool_use_block["name"],
                            tool_input=cast(
                                dict[str, Any], tool_use_block.get("input", {})
                            ),
                        )
                    except Exception as tool_error:
                        # Don't crash the loop - return error to Claude so it can retry
                        logger.error(f"[Loop] Tool execution error: {tool_error}")
                        result = ToolResult(error=f"Tool execution failed: {str(tool_error)}")
                    
                    tool_result_content.append(
                        _make_api_tool_result(result, tool_use_block["id"])
                    )
                    tool_output_callback(result, tool_use_block["id"])

            if not tool_result_content:
                logger.info(f"[Loop] No tool calls, task complete!")
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
            await _update_status(
                conversation_store, current_conversation_id, "failed", error_message
            )
            api_response_callback(e.request, e.response, e)
            return messages, current_conversation_id
        except APIError as e:
            error_message = f"API Error: {e}"
            logger.exception(f"[Loop] {error_message}")
            await _update_status(
                conversation_store, current_conversation_id, "failed", error_message
            )
            api_response_callback(e.request, e.body, e)
            return messages, current_conversation_id
        except Exception as e:
            error_message = f"An unexpected error occurred: {e}"
            logger.exception(f"[Loop] {error_message}")
            try:
                # Update status
                await _update_status(
                    conversation_store, current_conversation_id, "failed", error_message
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
