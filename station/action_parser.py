# Copyright 2025 DualverseAI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# station/execute_action_parser.py
"""
Parses agent responses to extract action commands and associated YAML data.
Action commands are in the format /execute_action{...} (optionally enclosed in backticks)
and normally start a line. Inline action tokens followed immediately by their YAML block
or another standalone action, and terminal inline action chains, are normalized onto
action lines. Tolerates empty content within action braces.
YAML blocks are expected to be in the format ```yaml ... ```
Includes a pre-processing step to neutralize /execute_action{} strings within YAML blocks.
"""

import os
import re
import sys
import yaml 
from typing import List, Tuple, Optional, Dict, Any, NamedTuple

if __package__ in (None, ""):
    package_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if package_parent not in sys.path:
        sys.path.insert(0, package_parent)

from station import constants 

class ParsedActionInfo(NamedTuple):
    command: str # Will be an empty string if action braces were empty
    args: Optional[str]
    yaml_data: Optional[Dict[str, Any]]
    yaml_error: Optional[str] # Can indicate "Empty action command"

class ActionParser:
    def __init__(self):
        self.action_prefix = "/execute_action" 
        
        # MODIFIED: Regex to optionally match backticks and allow EMPTY content inside braces.
        #   ^[ \t]* # Start of line, optional leading whitespace
        #   (?:\`?)                   # Optional non-capturing group for a leading backtick
        #   <escaped_prefix>          # The action prefix itself
        #   \{                       # A literal opening brace
        #   ([^{}]*)                 # Group 1: The command and arguments (ZERO or more chars not braces)
        #   \}                       # A literal closing brace
        #   (?:\`?)                   # Optional non-capturing group for a trailing backtick
        self.action_pattern_compiled = re.compile(
            r"^[ \t]*(?:\`?)" + re.escape(self.action_prefix) + r"\{([^{}]*)\}(?:\`?)", # Changed + to *
            re.MULTILINE
        )
        
        self.yaml_block_pattern_compiled = re.compile(
            r"^([ \t]*)```yaml\s*\n(?P<yaml_content>.*?)^(?:\1|)```(?=\s*(?:\n|$))", # Allow same indentation or zero indentation for closing ```
            re.MULTILINE | re.DOTALL
        )
        
        self.embedded_action_pattern = re.compile(re.escape(self.action_prefix) + r"\{")
        self.neutralized_action_prefix = "_action{" 

        self.thought_block_pattern_compiled = constants.THOUGHT_BLOCK_PATTERN
        self.yaml_open_fence_pattern = re.compile(r"^([ \t]*)```yaml\b")
        self.yaml_close_fence_pattern = re.compile(r"^([ \t]*)```(.*)$")
        self.fence_line_pattern = re.compile(r"^[ \t]*```")
        self.inline_action_token_pattern = re.compile(
            r"`?" + re.escape(self.action_prefix) + r"\{[^{}]*\}`?"
        )
        self.terminal_inline_action_suffix_pattern = re.compile(
            r"\A(?:`?" + re.escape(self.action_prefix) + r"\{[^{}]*\}`?\s*)+"
            r"(?:[ \t]*```yaml\s*\n.*?^[ \t]*```\s*)?\Z",
            re.MULTILINE | re.DOTALL,
        )

    def _split_line_end(self, line: str) -> Tuple[str, str]:
        if line.endswith("\r\n"):
            return line[:-2], "\r\n"
        if line.endswith("\n"):
            return line[:-1], "\n"
        if line.endswith("\r"):
            return line[:-1], "\r"
        return line, ""

    def _normalize_yaml_block_closings(self, text: str) -> str:
        """
        Ensure closing ``` in YAML blocks is on its own line.
        If a closing fence has trailing text, move that text to the next line.
        """
        lines = text.splitlines(keepends=True)
        output_lines = []
        in_yaml_block = False
        yaml_open_indent = ""

        for line in lines:
            body, line_ending = self._split_line_end(line)
            if not in_yaml_block:
                output_lines.append(line)
                open_match = self.yaml_open_fence_pattern.match(body)
                if open_match:
                    in_yaml_block = True
                    yaml_open_indent = open_match.group(1)
                continue

            close_match = self.yaml_close_fence_pattern.match(body)
            if close_match:
                indent = close_match.group(1)
                trailing = close_match.group(2)
                if indent == yaml_open_indent or indent == "":
                    if trailing.strip():
                        default_ending = line_ending if line_ending else "\n"
                        output_lines.append(f"{indent}```{default_ending}")
                        if trailing.startswith(" "):
                            trailing = trailing[1:]
                        output_lines.append(f"{trailing}{line_ending or ''}")
                    else:
                        output_lines.append(line)
                    in_yaml_block = False
                else:
                    output_lines.append(line)
            else:
                output_lines.append(line)

        return "".join(output_lines)

    def _neutralize_actions_in_yaml_blocks(self, text: str) -> str:
        output_parts = []
        last_end = 0
        for match in self.yaml_block_pattern_compiled.finditer(text):
            output_parts.append(text[last_end:match.start(0)])
            yaml_block_full_text = match.group(0)
            yaml_content_text = match.group("yaml_content")

            if yaml_content_text is not None:
                neutralized_content = self.embedded_action_pattern.sub(
                    self.neutralized_action_prefix, yaml_content_text
                )
                content_start_in_block = match.start("yaml_content") - match.start(0)
                content_end_in_block = match.end("yaml_content") - match.start(0)
                prefix = yaml_block_full_text[:content_start_in_block]
                suffix = yaml_block_full_text[content_end_in_block:]
                modified_yaml_block = prefix + neutralized_content + suffix
                output_parts.append(modified_yaml_block)
            else:
                output_parts.append(yaml_block_full_text)
            last_end = match.end(0)
        
        output_parts.append(text[last_end:])
        return "".join(output_parts)

    def _is_start_of_action_line(self, text: str, action_start: int) -> bool:
        line_start = text.rfind("\n", 0, action_start) + 1
        line_prefix = text[line_start:action_start]
        return line_prefix.strip() == ""

    def _position_inside_fenced_block(self, text: str, position: int) -> bool:
        in_fence = False
        cursor = 0

        for line in text.splitlines(keepends=True):
            line_end = cursor + len(line)
            if position < line_end:
                return in_fence

            if self.fence_line_pattern.match(line):
                in_fence = not in_fence
            cursor = line_end

        return in_fence

    def _normalize_terminal_inline_actions(self, text: str) -> str:
        """
        Move recoverable inline actions and terminal action chains onto action-only lines.

        An action immediately followed by its YAML block or another standalone action
        is strong evidence that the model intended to issue it, even when it was glued
        to the preceding prose. Other inline tokens remain inert unless they form the
        existing terminal action-chain recovery case.
        """
        normalized_parts = []
        cursor = 0
        for match in self.inline_action_token_pattern.finditer(text):
            action_start = match.start(0)
            if self._is_start_of_action_line(text, action_start):
                continue
            if self._position_inside_fenced_block(text, action_start):
                continue

            token = match.group(0).strip().strip("`")
            command_text = token[len(self.action_prefix):].strip()
            if not command_text.startswith("{") or not command_text.endswith("}"):
                continue
            command_and_args = command_text[1:-1].strip()
            command = command_and_args.split(None, 1)[0].lower() if command_and_args else ""
            following_text = text[match.end(0):]
            yaml_follows = (
                command in constants.ACTIONS_EXPECTING_YAML
                and re.match(r"^[ \t]*\r?\n[ \t]*```yaml\b", following_text)
            )
            standalone_action_follows = re.match(
                r"^[ \t]*\r?\n(?:[ \t]*\r?\n)*[ \t]*`?"
                + re.escape(self.action_prefix)
                + r"\{",
                following_text,
            )
            if not yaml_follows and not standalone_action_follows:
                continue

            normalized_parts.append(text[cursor:action_start])
            normalized_parts.append("\n")
            normalized_parts.append(text[action_start:match.end(0)])
            cursor = match.end(0)

        if normalized_parts:
            normalized_parts.append(text[cursor:])
            text = "".join(normalized_parts)

        for match in self.inline_action_token_pattern.finditer(text):
            action_start = match.start(0)
            if self._is_start_of_action_line(text, action_start):
                continue
            if self._position_inside_fenced_block(text, action_start):
                continue

            suffix = text[action_start:]
            if not self.terminal_inline_action_suffix_pattern.fullmatch(suffix):
                continue

            prefix = text[:action_start].rstrip(" \t")
            normalized_suffix = re.sub(
                r"(?<!^)(?<!\n)[ \t]*(`?" + re.escape(self.action_prefix) + r"\{[^{}]*\}`?)",
                r"\n\1",
                suffix,
            )

            separator = "\n" if prefix else ""
            return prefix + separator + normalized_suffix.lstrip(" \t")

        return text
    
    def _remove_blocks(self, text: str, block_pattern: re.Pattern) -> str:
        """
        Completely removes blocks matching the pattern from the text.
        """
        return block_pattern.sub("", text)

    def _normalize_tags(self, data: Dict[str, Any]) -> None:
        if constants.TAGS_KEY in data: 
            tags_value = data[constants.TAGS_KEY]
            if isinstance(tags_value, str):
                data[constants.TAGS_KEY] = [tag.strip() for tag in tags_value.split(',') if tag.strip()]
            elif isinstance(tags_value, list):
                data[constants.TAGS_KEY] = [str(tag).strip() for tag in tags_value if str(tag).strip()]

    def parse(self, response_text: str) -> List[ParsedActionInfo]:
        processed_response_text = self._normalize_yaml_block_closings(response_text)
        processed_response_text = self._neutralize_actions_in_yaml_blocks(processed_response_text)
        processed_response_text = self._remove_blocks(
            processed_response_text, self.thought_block_pattern_compiled
        )
        processed_response_text = self._normalize_terminal_inline_actions(processed_response_text)

        parsed_actions: List[ParsedActionInfo] = []
        action_matches = list(self.action_pattern_compiled.finditer(processed_response_text))
        
        for i, current_action_match in enumerate(action_matches):
            command_and_args_str = current_action_match.group(1).strip() 
            
            command: str
            args: Optional[str]
            
            # MODIFICATION: Handle empty command_and_args_str (from /execute_action{})
            if not command_and_args_str: # Content within {} was empty or just whitespace
                command = "" 
                args = None
                action_error_message = f"Empty action command received: {current_action_match.group(0)}"
            else:
                parts = command_and_args_str.split(None, 1)
                command = parts[0].lower()
                args = parts[1] if len(parts) > 1 else None
                action_error_message = None # No error at this stage for non-empty commands
            
            yaml_data: Optional[Dict[str, Any]] = None
            yaml_error: Optional[str] = None # Specific to YAML parsing

            if command and command in constants.ACTIONS_EXPECTING_YAML: # Only look for YAML if command is valid and expects it
                current_action_end_pos = current_action_match.end(0) 
                next_action_start_pos = action_matches[i+1].start(0) if i + 1 < len(action_matches) else len(processed_response_text)
                potential_yaml_search_space = processed_response_text[current_action_end_pos:next_action_start_pos]
                
                yaml_match = self.yaml_block_pattern_compiled.search(potential_yaml_search_space)
                
                if not yaml_match:
                    # Try to fix unclosed YAML by adding/fixing closing backticks
                    lines = potential_yaml_search_space.split('\n')
                    
                    # Check if last line has backticks (potentially malformed closing)
                    last_line = lines[-1].strip()
                    if last_line.startswith('```') or '```' in last_line:
                        # Replace last line with proper closing - if it's the last line, it's likely intended as closing
                        lines[-1] = '```'
                    else:
                        # Add closing backticks
                        lines.append('```')
                    
                    fixed_search_space = '\n'.join(lines)
                    yaml_match = self.yaml_block_pattern_compiled.search(fixed_search_space)
                
                if yaml_match:
                    text_before_yaml_in_segment = potential_yaml_search_space[:yaml_match.start(0)]
                    if text_before_yaml_in_segment.strip() == "":
                        yaml_content_str = yaml_match.group("yaml_content") 
                        try:
                            if yaml_content_str and yaml_content_str.strip(): 
                                loaded_yaml = yaml.safe_load(yaml_content_str)
                                if isinstance(loaded_yaml, dict):
                                    self._normalize_tags(loaded_yaml)
                                    yaml_data = loaded_yaml
                                elif loaded_yaml is not None:
                                    yaml_error = f"YAML for '{command}' not a dictionary (type: {type(loaded_yaml)})."
                        except yaml.YAMLError as e:
                            yaml_error = f"YAML parsing error for '{command}': {e}"
            
            # If there was an error message from an empty action, use it as yaml_error
            # (or a more specific error field if ParsedActionInfo is expanded)
            final_error = yaml_error or action_error_message
            parsed_actions.append(ParsedActionInfo(command, args, yaml_data, final_error))

        return parsed_actions

if __name__ == '__main__':
    try:
        from station import constants as main_test_constants
    except ImportError:
        print("Warning: Could not import station.constants for __main__ test. Using local definitions.")
        class MainTestConstants:
            ACTIONS_EXPECTING_YAML = {"create", "reply", "update", "speak"} 
            TAGS_KEY = "tags"
        main_test_constants = MainTestConstants()
        # This is a bit tricky for module-level imports in the class.
        # For robust testing, ensure station.constants is in PYTHONPATH or mock it more thoroughly.
        # For now, the class directly uses `from station import constants`.
        # This __main__ block might fail if station.constants cannot be found when run directly.
        # A better approach for standalone testability would be to pass constants module to parser.
        constants_module_for_test = main_test_constants # Use the local mock for test
    else:
        constants_module_for_test = main_test_constants


    parser = ActionParser() 
    # For __main__ test, we might need to temporarily patch the constants used by the parser instance
    # if it directly imports `station.constants` and that's not the one we want for the test.
    # However, the current parser directly imports, so this test relies on that import working.

    print(f"--- ActionParser Tests (Prefix: '{parser.action_prefix}', Start-of-Line, Terminal Inline, Optional Backticks, Empty Braces) ---")

    test_cases = [
        (f"  {parser.action_prefix}{{goto lobby}}", "Plain action", "goto", "lobby", None, None),
        (f"`{parser.action_prefix}{{help lobby}}`", "Action with backticks", "help", "lobby", None, None),
        (f"  `{parser.action_prefix}{{create}}`\n```yaml\ntitle: Test\n```", "Action with backticks and YAML", "create", None, {"title": "Test"}, None),
        (f"{parser.action_prefix}{{speak}}\n  ```yaml\n  message: Hello\n  ```", "Action no backticks, indented YAML", "speak", None, {"message": "Hello"}, None),
        (f"Not an action: {parser.action_prefix}{{ignored}} after text", "Mid-sentence action (ignored)", None, None, None, None),
        (f"Terminal action: {parser.action_prefix}{{help lobby}}", "Terminal inline action", "help", "lobby", None, None),
        (f"{parser.action_prefix}{{}}", "Empty action braces", "", None, None, "Empty action command received"),
        (f"  `{parser.action_prefix}{{}}`  ", "Empty action braces with backticks and spaces", "", None, None, "Empty action command received"),
    ]

    for i, (text, desc, exp_cmd, exp_args, exp_yaml_keys, exp_error_part) in enumerate(test_cases):
        print(f"\nTest {i+1}: {desc}\nInput:\n{text}")
        actions = parser.parse(text)
        print(f"Parsed: {actions}")
        if exp_cmd is None: 
            assert len(actions) == 0, f"Expected no actions for '{desc}', got {len(actions)}"
        else:
            assert len(actions) == 1, f"Expected 1 action for '{desc}', got {len(actions)}"
            action = actions[0]
            assert action.command == exp_cmd, f"For '{desc}', Cmd: Expected '{exp_cmd}', Got '{action.command}'"
            assert action.args == exp_args, f"For '{desc}', Args: Expected '{exp_args}', Got '{action.args}'"
            if exp_yaml_keys is not None:
                assert action.yaml_data is not None, f"For '{desc}', YAML data should be parsed"
                if action.yaml_data:
                    for k,v in exp_yaml_keys.items():
                        assert action.yaml_data.get(k) == v, f"For '{desc}', YAML key '{k}': Expected '{v}', Got '{action.yaml_data.get(k)}'"
            else:
                assert action.yaml_data is None, f"For '{desc}', YAML data should be None"
            
            if exp_error_part:
                assert action.yaml_error is not None, f"For '{desc}', Expected an error containing '{exp_error_part}', but got no error."
                if action.yaml_error:
                    assert exp_error_part in action.yaml_error, f"For '{desc}', Error: Expected to contain '{exp_error_part}', Got '{action.yaml_error}'"
            else:
                assert action.yaml_error is None, f"For '{desc}', YAML error should be None, Got '{action.yaml_error}'"
    
    print("\nActionParser tests completed.")
