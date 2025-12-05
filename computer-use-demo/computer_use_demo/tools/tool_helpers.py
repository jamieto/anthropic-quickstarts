# tools/tool_helpers.py
"""
Tool call validation and auto-fixing for Claude Computer Use.
Catches common mistakes before execution, auto-fixes where possible.
"""

from pathlib import Path
from typing import Optional, Tuple
from .base import ToolResult

import logging
logger = logging.getLogger(__name__)


def fix_and_validate_tool_call(
    tool_name: str, 
    tool_input: dict
) -> Tuple[dict, Optional[ToolResult]]:
    """
    Auto-fix and validate tool calls before execution.
    
    Returns:
        (fixed_input, None) - if valid, proceed with fixed_input
        (original_input, ToolResult with error) - if invalid, return error to Claude
    """
    
    # Step 1: Auto-fix what we can
    fixed_input, fixes_applied = auto_fix_tool_call(tool_name, tool_input.copy())
    
    if fixes_applied:
        logger.info(f"[ToolHelper] Auto-fixed tool call: {fixes_applied}")
    
    # Step 2: Validate (catch what we couldn't fix)
    validation_error = validate_tool_call(tool_name, fixed_input)
    
    if validation_error:
        return tool_input, validation_error
    
    return fixed_input, None


def auto_fix_tool_call(tool_name: str, tool_input: dict) -> Tuple[dict, list[str]]:
    """
    Auto-fix common mistakes in tool calls.
    
    Returns:
        (fixed_input, list of fixes applied)
    """
    fixes = []
    
    # =========================================================================
    # BASH TOOL FIXES
    # =========================================================================
    if tool_name == "bash":
        # Nothing we can really auto-fix for bash
        # If command is missing, we can't guess what they wanted
        pass
    
    # =========================================================================
    # STR_REPLACE_EDITOR FIXES
    # =========================================================================
    elif tool_name in ("str_replace_editor", "str_replace_based_edit_tool"):
        path = tool_input.get("path", "")
        command = tool_input.get("command")
        
        # Fix: Relative path → Absolute path
        if path and not path.startswith("/"):
            # Assume they meant /home/computeruse/project/
            tool_input["path"] = f"/home/computeruse/project/{path}"
            fixes.append(f"path: '{path}' → '{tool_input['path']}'")
            path = tool_input["path"]
        
        # Fix: str_replace with empty old_str on empty/new file → convert to insert
        if command == "str_replace":
            old_str = tool_input.get("old_str")
            new_str = tool_input.get("new_str", "")
            
            if old_str == "" and new_str:
                # Check if file exists and is empty (or doesn't exist)
                try:
                    file_path = Path(path)
                    file_is_empty = not file_path.exists() or file_path.read_text().strip() == ""
                    
                    if file_is_empty and not file_path.exists():
                        # File doesn't exist - convert to create
                        tool_input["command"] = "create"
                        tool_input["file_text"] = new_str
                        tool_input.pop("old_str", None)
                        tool_input.pop("new_str", None)
                        fixes.append("str_replace on non-existent file → create")
                    elif file_is_empty:
                        # File exists but empty - convert to insert at line 0
                        tool_input["command"] = "insert"
                        tool_input["insert_line"] = 0
                        tool_input.pop("old_str", None)
                        fixes.append("str_replace with empty old_str on empty file → insert at line 0")
                except Exception as e:
                    logger.warning(f"[ToolHelper] Could not check file for auto-fix: {e}")
        
        # Fix: insert with missing insert_line but has line number in wrong field
        if command == "insert":
            if tool_input.get("insert_line") is None:
                # Check if they put it in a wrong field name
                for wrong_name in ["line", "line_number", "at_line", "insertLine"]:
                    if wrong_name in tool_input:
                        tool_input["insert_line"] = tool_input.pop(wrong_name)
                        fixes.append(f"{wrong_name} → insert_line")
                        break
        
        # Fix: new_str vs file_text confusion for create
        if command == "create":
            if tool_input.get("file_text") is None and tool_input.get("new_str"):
                tool_input["file_text"] = tool_input.pop("new_str")
                fixes.append("new_str → file_text for create command")
            
            if tool_input.get("file_text") is None and tool_input.get("content"):
                tool_input["file_text"] = tool_input.pop("content")
                fixes.append("content → file_text for create command")
    
    # =========================================================================
    # COMPUTER TOOL FIXES
    # =========================================================================
    elif tool_name == "computer":
        action = tool_input.get("action")
        
        # Fix: coordinate as string → list
        coord = tool_input.get("coordinate")
        if isinstance(coord, str):
            try:
                # Try to parse "[100, 200]" or "100, 200" or "100 200"
                import re
                numbers = re.findall(r'\d+', coord)
                if len(numbers) >= 2:
                    tool_input["coordinate"] = [int(numbers[0]), int(numbers[1])]
                    fixes.append(f"coordinate string → list: {tool_input['coordinate']}")
            except:
                pass
        
        # Fix: scroll_amount as string → int
        scroll = tool_input.get("scroll_amount")
        if isinstance(scroll, str):
            try:
                tool_input["scroll_amount"] = int(scroll)
                fixes.append(f"scroll_amount string → int")
            except:
                pass
    
    return tool_input, fixes


def validate_tool_call(tool_name: str, tool_input: dict) -> Optional[ToolResult]:
    """
    Validate tool calls. Returns ToolResult with error if invalid, None if valid.
    Called AFTER auto-fix, so only catches truly invalid calls.
    """
    
    # =========================================================================
    # BASH VALIDATION
    # =========================================================================
    if tool_name == "bash":
        command = tool_input.get("command")
        restart = tool_input.get("restart")
        
        if not command and not restart:
            return ToolResult(
                error="MISSING PARAMETER: 'command'\n\n"
                      "The bash tool requires a command to execute.\n\n"
                      "CORRECT FORMAT:\n"
                      '{\n'
                      '  "command": "your_shell_command_here"\n'
                      '}\n\n'
                      "EXAMPLES:\n"
                      '  {"command": "ls -la /home/computeruse/project/"}\n'
                      '  {"command": "pip install pandas --break-system-packages"}\n'
                      '  {"restart": true}  // to restart bash session'
            )
    
    # =========================================================================
    # STR_REPLACE_EDITOR VALIDATION
    # =========================================================================
    elif tool_name in ("str_replace_editor", "str_replace_based_edit_tool"):
        command = tool_input.get("command")
        path = tool_input.get("path")
        
        # Validate path exists for all commands
        if not path:
            return ToolResult(
                error="MISSING PARAMETER: 'path'\n\n"
                      "All str_replace_editor commands require a 'path' parameter.\n"
                      "Path must be absolute (start with /)."
            )
        
        if not path.startswith("/"):
            return ToolResult(
                error=f"INVALID PATH: '{path}'\n\n"
                      f"Path must be absolute (start with /).\n\n"
                      f"Did you mean: /home/computeruse/project/{path}"
            )
        
        # Command-specific validation
        if command == "create":
            file_text = tool_input.get("file_text")
            
            if file_text is None:
                return ToolResult(
                    error="MISSING PARAMETER: 'file_text'\n\n"
                          "The 'create' command REQUIRES 'file_text' with the file content.\n\n"
                          "CORRECT FORMAT:\n"
                          '{\n'
                          '  "command": "create",\n'
                          f'  "path": "{path}",\n'
                          '  "file_text": "# Your content here\\n\\nComplete file content..."\n'
                          '}\n\n'
                          "You must provide the complete file content in 'file_text'."
                )
            
            # Note: We allow empty file_text ("") - sometimes that's intentional
            # But warn if it's just whitespace
            if file_text is not None and file_text.strip() == "" and file_text != "":
                return ToolResult(
                    error="INVALID: 'file_text' contains only whitespace.\n\n"
                          "If you want to create an empty file, use file_text: \"\"\n"
                          "Otherwise, provide actual content."
                )
        
        elif command == "str_replace":
            old_str = tool_input.get("old_str")
            
            if old_str is None:
                return ToolResult(
                    error="MISSING PARAMETER: 'old_str'\n\n"
                          "The 'str_replace' command requires 'old_str' (text to find and replace).\n\n"
                          "CORRECT FORMAT:\n"
                          '{\n'
                          '  "command": "str_replace",\n'
                          f'  "path": "{path}",\n'
                          '  "old_str": "exact text to find",\n'
                          '  "new_str": "replacement text"\n'
                          '}'
                )
            
            if old_str == "":
                return ToolResult(
                    error="INVALID: 'old_str' cannot be empty.\n\n"
                          "str_replace finds and replaces EXISTING text. "
                          "Empty string matches nothing.\n\n"
                          "WHAT YOU PROBABLY WANT:\n\n"
                          "To ADD content to an existing file, use 'insert':\n"
                          '{\n'
                          '  "command": "insert",\n'
                          f'  "path": "{path}",\n'
                          '  "insert_line": 0,\n'
                          '  "new_str": "content to add"\n'
                          '}\n\n'
                          "To CREATE a new file with content, use 'create':\n"
                          '{\n'
                          '  "command": "create",\n'
                          f'  "path": "{path}",\n'
                          '  "file_text": "complete file content"\n'
                          '}\n\n'
                          "To REPLACE specific text, provide the exact text to find in 'old_str'."
                )
        
        elif command == "insert":
            insert_line = tool_input.get("insert_line")
            new_str = tool_input.get("new_str")
            
            if insert_line is None:
                return ToolResult(
                    error="MISSING PARAMETER: 'insert_line'\n\n"
                          "The 'insert' command requires 'insert_line' (line number to insert at).\n\n"
                          "CORRECT FORMAT:\n"
                          '{\n'
                          '  "command": "insert",\n'
                          f'  "path": "{path}",\n'
                          '  "insert_line": 0,\n'
                          '  "new_str": "content to insert"\n'
                          '}\n\n'
                          "Use insert_line: 0 to insert at the beginning of the file."
                )
            
            if new_str is None:
                return ToolResult(
                    error="MISSING PARAMETER: 'new_str'\n\n"
                          "The 'insert' command requires 'new_str' (content to insert).\n\n"
                          "CORRECT FORMAT:\n"
                          '{\n'
                          '  "command": "insert",\n'
                          f'  "path": "{path}",\n'
                          f'  "insert_line": {insert_line},\n'
                          '  "new_str": "content to insert"\n'
                          '}'
                )
        
        elif command == "view":
            # view is usually fine, just needs path (already validated above)
            view_range = tool_input.get("view_range")
            if view_range is not None:
                if not isinstance(view_range, list) or len(view_range) != 2:
                    return ToolResult(
                        error="INVALID: 'view_range' must be a list of 2 integers [start, end].\n\n"
                              "EXAMPLE: {\"command\": \"view\", \"path\": \"...\", \"view_range\": [1, 50]}"
                    )
        
        elif command == "undo_edit":
            # undo_edit just needs path (already validated)
            pass
        
        elif command is None:
            return ToolResult(
                error="MISSING PARAMETER: 'command'\n\n"
                      "str_replace_editor requires a 'command' parameter.\n\n"
                      "AVAILABLE COMMANDS:\n"
                      "  - create: Create new file (requires: path, file_text)\n"
                      "  - view: View file contents (requires: path)\n"
                      "  - str_replace: Replace text (requires: path, old_str, new_str)\n"
                      "  - insert: Insert at line (requires: path, insert_line, new_str)\n"
                      "  - undo_edit: Undo last edit (requires: path)"
            )
        
        else:
            return ToolResult(
                error=f"UNKNOWN COMMAND: '{command}'\n\n"
                      "AVAILABLE COMMANDS:\n"
                      "  - create, view, str_replace, insert, undo_edit"
            )
    
    # =========================================================================
    # COMPUTER TOOL VALIDATION
    # =========================================================================
    elif tool_name == "computer":
        action = tool_input.get("action")
        
        if not action:
            return ToolResult(
                error="MISSING PARAMETER: 'action'\n\n"
                      "The computer tool requires an 'action' parameter.\n\n"
                      "COMMON ACTIONS:\n"
                      '  {"action": "screenshot"}\n'
                      '  {"action": "mouse_move", "coordinate": [500, 300]}\n'
                      '  {"action": "left_click"}\n'
                      '  {"action": "type", "text": "hello"}\n'
                      '  {"action": "key", "key": "Return"}'
            )
        
        # Validate coordinate for actions that need it
        coordinate_actions = ["mouse_move", "left_click_drag"]
        if action in coordinate_actions:
            coord = tool_input.get("coordinate")
            if coord is None:
                return ToolResult(
                    error=f"MISSING PARAMETER: 'coordinate'\n\n"
                          f"The '{action}' action requires a 'coordinate' parameter.\n\n"
                          "CORRECT FORMAT:\n"
                          '{\n'
                          f'  "action": "{action}",\n'
                          '  "coordinate": [x, y]\n'
                          '}\n\n'
                          "EXAMPLE: {\"action\": \"mouse_move\", \"coordinate\": [500, 300]}"
                )
            
            if not isinstance(coord, list) or len(coord) != 2:
                return ToolResult(
                    error=f"INVALID: 'coordinate' must be [x, y] list of 2 integers.\n\n"
                          f"Got: {coord}\n\n"
                          "CORRECT: {\"action\": \"mouse_move\", \"coordinate\": [500, 300]}"
                )
        
        # Validate text for type action
        if action == "type":
            text = tool_input.get("text")
            if text is None:
                return ToolResult(
                    error="MISSING PARAMETER: 'text'\n\n"
                          "The 'type' action requires a 'text' parameter.\n\n"
                          "CORRECT: {\"action\": \"type\", \"text\": \"your text here\"}"
                )
        
        # Validate key for key action
        if action == "key":
            key = tool_input.get("key")
            if key is None:
                return ToolResult(
                    error="MISSING PARAMETER: 'key'\n\n"
                          "The 'key' action requires a 'key' parameter.\n\n"
                          "EXAMPLES:\n"
                          '  {"action": "key", "key": "Return"}\n'
                          '  {"action": "key", "key": "ctrl+c"}\n'
                          '  {"action": "key", "key": "Alt+Tab"}'
                )
        
        # Validate scroll
        if action == "scroll":
            direction = tool_input.get("scroll_direction")  
            if direction is None:
                return ToolResult(
                    error="MISSING PARAMETER: 'scroll_direction'\n\n"
                          "The 'scroll' action requires 'scroll_direction'.\n\n"
                          "CORRECT: {\"action\": \"scroll\", \"scroll_direction\": \"down\", \"scroll_amount\": 3}"
                )
    
    # Valid!
    return None