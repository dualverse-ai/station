import re
import yaml
import os

# --- Configuration ---
CODEX_INPUT_MD_FILE = "codex.md"
MANIFEST_OUTPUT_YAML_FILE = "codex_manifest.yaml"

# Keys for the manifest, aligning with your constants.py and codex_content_guide
CODEX_MANIFEST_MODULES_KEY = "modules"
CODEX_MODULE_ID_KEY = "id"
CODEX_MODULE_TITLE_KEY = "title"
CODEX_MODULE_FILE_KEY = "file"
CODEX_MODULE_WORD_COUNT_KEY = "word_count"

def calculate_word_count(text: str) -> int:
    """Calculates a simple word count of the main body text."""
    # Word count should be based on the space-normalized text if that's what the LLM sees
    return len(_normalize_inline_spaces(text).split()) if text else 0

# --- NEW HELPER FUNCTION for space normalization ---
def _normalize_inline_spaces(text: str) -> str:
    """
    Replaces multiple inline spaces and tabs with a single space,
    while preserving newline characters for paragraph structure.
    Also strips leading/trailing whitespace from each line.
    """
    if not text:
        return ""
    lines = text.splitlines()
    normalized_lines = []
    for line in lines:
        # Replace multiple spaces/tabs within the line with a single space
        normalized_line = re.sub(r'[ \t]+', ' ', line).strip()
        normalized_lines.append(normalized_line)
    # Join lines back, preserving paragraph breaks (double newlines become single,
    # but multiple newlines from input might be condensed depending on original split behavior)
    # A more robust way to preserve paragraphs is to split by \n then normalize, then join by \n
    # The current approach normalizes each line then joins them with \n.
    # Let's refine to better preserve intentional paragraph breaks (double newlines)
    # while still normalizing spaces within lines.

    # Simpler: just replace multiple horizontal spaces in the whole block
    # This will keep all existing \n characters as they are.
    text_normalized_spaces = re.sub(r'[ \t]+', ' ', text)
    
    # To handle multiple newlines collapsing to a maximum of two (for paragraphs)
    # and removing leading/trailing newlines from the whole block before writing.
    text_normalized_newlines = re.sub(r'\n{3,}', '\n\n', text_normalized_spaces)
    return text_normalized_newlines.strip() # Final strip for the whole block

def main():
    """
    Parses the codex.md file, creates individual module files with their headings
    (with normalized spaces), and generates the codex_manifest.yaml.
    """
    if not os.path.exists(CODEX_INPUT_MD_FILE):
        print(f"Error: Input file '{CODEX_INPUT_MD_FILE}' not found in the current directory.")
        return

    with open(CODEX_INPUT_MD_FILE, 'r', encoding='utf-8') as f:
        content = f.read()

    manifest_modules = []
    created_files_count = 0

    module_heading_pattern = re.compile(r"^(## (Preface|Module\s*(\d+)): (.*))$", re.MULTILINE)
    matches = list(module_heading_pattern.finditer(content))

    if not matches:
        print("Error: No module headings (e.g., '## Preface: ...' or '## Module X: ...') found in the file.")
        return
        
    first_match_start_index = matches[0].start()
    text_before_first_match = content[:first_match_start_index].strip()
    codex_title_to_prepend_raw = ""
    if text_before_first_match == "# The Codex":
        codex_title_to_prepend_raw = text_before_first_match 
    
    # Normalize the prepended title (if any)
    codex_title_to_prepend_normalized = _normalize_inline_spaces(codex_title_to_prepend_raw)
    if codex_title_to_prepend_normalized: # Add newlines only if there's content
        codex_title_to_prepend_normalized += "\n\n"


    for i, match in enumerate(matches):
        full_heading_from_match_raw = match.group(1).strip() 
        type_or_module_part = match.group(2).strip() 
        module_title_raw = match.group(4).strip() 

        module_num_str = None
        if type_or_module_part.startswith("Module"):
            num_match = re.search(r"Module\s*(\d+)", type_or_module_part)
            if num_match:
                module_num_str = num_match.group(1)
        
        section_body_start = match.end()
        if i + 1 < len(matches):
            section_body_end = matches[i+1].start()
        else:
            section_body_end = len(content)
        
        section_body_raw = content[section_body_start:section_body_end].strip()

        # --- Normalize spaces for each part ---
        full_heading_normalized = _normalize_inline_spaces(full_heading_from_match_raw)
        # The title for the manifest should also be normalized if it had extra spaces
        module_title_normalized = _normalize_inline_spaces(module_title_raw)
        section_body_normalized = _normalize_inline_spaces(section_body_raw)

        module_id_for_manifest: Any = ""
        output_filename = ""
        content_to_write = ""

        if type_or_module_part.lower() == "preface":
            module_id_for_manifest = "preface"
            output_filename = "preface.md"
            # Use normalized parts. codex_title_to_prepend_normalized already has trailing \n\n if it exists.
            content_to_write = codex_title_to_prepend_normalized + full_heading_normalized + "\n\n" + section_body_normalized
            codex_title_to_prepend_normalized = "" # Ensure it's only prepended once
        elif module_num_str: 
            try:
                module_id_for_manifest = int(module_num_str) 
            except ValueError:
                module_id_for_manifest = module_num_str 
            output_filename = f"module_{module_num_str}.md"
            content_to_write = full_heading_normalized + "\n\n" + section_body_normalized
        else:
            print(f"Warning: Could not determine module type or number for: '{full_heading_from_match_raw}'. Skipping.")
            continue
            
        try:
            with open(output_filename, 'w', encoding='utf-8') as f_out:
                # Final strip on content_to_write to remove any leading/trailing whitespace from the whole block,
                # then add a single trailing newline for POSIX compatibility.
                f_out.write(content_to_write.strip() + "\n") 
            print(f"Created module file: {output_filename}")
            created_files_count += 1
        except IOError as e:
            print(f"Error writing file {output_filename}: {e}")
            continue

        # Word count is based on the normalized section_body
        word_count = calculate_word_count(section_body_normalized) # calculate_word_count now calls _normalize_inline_spaces

        manifest_modules.append({
            CODEX_MODULE_ID_KEY: module_id_for_manifest,
            CODEX_MODULE_TITLE_KEY: module_title_normalized, # Use normalized title for manifest
            CODEX_MODULE_FILE_KEY: output_filename,
            CODEX_MODULE_WORD_COUNT_KEY: word_count
        })

    if not manifest_modules:
        print("No modules were successfully processed to create a manifest.")
        return

    manifest_data = {
        CODEX_MANIFEST_MODULES_KEY: manifest_modules
    }

    try:
        with open(MANIFEST_OUTPUT_YAML_FILE, 'w', encoding='utf-8') as f_yaml:
            yaml.dump(manifest_data, f_yaml, sort_keys=False, allow_unicode=True, indent=2)
        print(f"\nSuccessfully created manifest: {MANIFEST_OUTPUT_YAML_FILE}")
        print(f"Total module files created: {created_files_count}")
    except IOError as e:
        print(f"Error writing manifest file {MANIFEST_OUTPUT_YAML_FILE}: {e}")
    except yaml.YAMLError as e:
        print(f"Error formatting YAML for manifest: {e}")

if __name__ == "__main__":
    main()
