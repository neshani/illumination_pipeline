# src/project_manager.py
import os
import sys
import subprocess
import json
import ebooklib
import warnings
from ebooklib import epub
from bs4 import BeautifulSoup
from pathlib import Path
import shutil
import re
import pandas as pd

# Local imports
from src.config_manager import get_default_project_config, load_global_config, load_project_config 
from src.utils import Colors

# Suppress the specific FutureWarning from ebooklib
warnings.filterwarnings("ignore", category=FutureWarning, module='ebooklib.epub')

BOOKS_FOLDER = "Books"
TRANSCRIPTS_FOLDER = "Transcripts"
ILLUMINATIONS_FOLDER = "Illuminations"

def ensure_project_folders_exist():
    """Creates the core input/output folders if they don't exist."""
    os.makedirs(BOOKS_FOLDER, exist_ok=True)
    os.makedirs(TRANSCRIPTS_FOLDER, exist_ok=True)
    os.makedirs(ILLUMINATIONS_FOLDER, exist_ok=True)

def find_projects():
    """Finds all existing project folders within the Illuminations directory."""
    projects = []
    if not os.path.exists(ILLUMINATIONS_FOLDER): return []
    for item in os.listdir(ILLUMINATIONS_FOLDER):
        item_path = os.path.join(ILLUMINATIONS_FOLDER, item)
        clean_txt_path = os.path.join(item_path, f"{item}_clean.txt")
        if os.path.isdir(item_path) and os.path.exists(clean_txt_path):
            projects.append((item, item_path))
    return sorted(projects)

def find_importable_epubs():
    """Finds .epub files in the Books folder that don't have a project yet."""
    epubs, existing_project_names = [], [p[0] for p in find_projects()]
    if not os.path.exists(BOOKS_FOLDER): return []
    for item in os.listdir(BOOKS_FOLDER):
        if item.lower().endswith(".epub"):
            book_name = Path(item).stem
            if book_name not in existing_project_names:
                epubs.append(item)
    return sorted(epubs)

def find_importable_transcripts():
    """Finds .txt files in the Transcripts folder that don't have a project yet."""
    transcripts, existing_project_names = [], [p[0] for p in find_projects()]
    if not os.path.exists(TRANSCRIPTS_FOLDER): return []
    for item in os.listdir(TRANSCRIPTS_FOLDER):
        if item.lower().endswith(".txt"):
            book_name = Path(item).stem
            if book_name not in existing_project_names:
                transcripts.append(item)
    return sorted(transcripts)

def create_project_from_transcript(transcript_name):
    """Creates the folder structure by copying a pre-made transcript."""
    transcript_path = os.path.join(TRANSCRIPTS_FOLDER, transcript_name)
    book_name = Path(transcript_path).stem
    project_folder = os.path.join(ILLUMINATIONS_FOLDER, book_name)
    images_folder = os.path.join(project_folder, "images")

    print(f"Creating new project from transcript '{book_name}'...")
    os.makedirs(images_folder, exist_ok=True)

    # Copy the transcript to its new home as the _clean.txt file
    clean_txt_path = os.path.join(project_folder, f"{book_name}_clean.txt")
    shutil.copy(transcript_path, clean_txt_path)
    print(f"  - Copied transcript to '{clean_txt_path}'")

    # Create the default config file
    config_path = os.path.join(project_folder, "config.json")
    default_config = get_default_project_config()
    with open(config_path, 'w') as f: json.dump(default_config, f, indent=4)
    print(f"  - Created project 'config.json' from global defaults.")

    print(f"\n{Colors.GREEN}Project created successfully. Ready for prompt generation.{Colors.ENDC}")
    return book_name, project_folder

def create_project_structure(epub_name):
    """Creates the folder structure and initial files for a new project."""
    epub_path = os.path.join(BOOKS_FOLDER, epub_name)
    book_name = Path(epub_path).stem
    project_folder = os.path.join(ILLUMINATIONS_FOLDER, book_name)
    images_folder = os.path.join(project_folder, "images")
    
    print(f"Creating new project for '{book_name}'...")
    os.makedirs(images_folder, exist_ok=True)
    
    clean_txt_path = os.path.join(project_folder, f"{book_name}_clean.txt")
    try:
        print(f"  - Converting EPUB...")
        book = epub.read_epub(epub_path)
        full_text_parts = ["==CHAPTER=="]
        
        for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
            soup = BeautifulSoup(item.get_content(), 'html.parser')
            
            # --- REFINED LOGIC ---
            # Find all potential paragraph-level tags.
            # This is more robust for books that use <div>s instead of <p>s.
            potential_tags = soup.find_all(['p', 'div'])
            
            chapter_paragraphs = []
            for tag in potential_tags:
                # This is the crucial check: we only want to extract text from a tag
                # if it DOES NOT contain another block-level tag within it.
                # This prevents grabbing text from a container <div> and then
                # grabbing it again from the <p> tags inside it.
                if not tag.find(['p', 'div']):
                    text = tag.get_text(strip=True)
                    if text:
                        chapter_paragraphs.append(text)

            if chapter_paragraphs:
                # Join the collected paragraphs and add the chapter separator
                full_text_parts.append('\n\n'.join(chapter_paragraphs))
                full_text_parts.append("\n\n==CHAPTER==\n\n")
                
        # Join all the chapter parts together into the final text file
        with open(clean_txt_path, 'w', encoding='utf-8') as f: f.write(''.join(full_text_parts))
        print("  - Conversion successful.")
    except Exception as e:
        print(f"  - ERROR converting EPUB: {e}"); return None

    config_path = os.path.join(project_folder, "config.json")
    default_config = get_default_project_config()
    with open(config_path, 'w') as f: json.dump(default_config, f, indent=4)
    print(f"  - Created project 'config.json' from global defaults.")

    print(f"\n{Colors.BOLD}{Colors.YELLOW}IMPORTANT:{Colors.ENDC} The book's text has been extracted to:")
    print(f"'{clean_txt_path}'")
    print("\nThe file will now open for you to edit.")
    print("Please clean it up by removing any table of contents, forewords, or other non-story text.")
    print("The text should ideally begin with the first word of the first chapter.")
    
    try:
        if sys.platform == "win32": os.startfile(os.path.realpath(clean_txt_path))
        elif sys.platform == "darwin": subprocess.call(["open", clean_txt_path])
        else: subprocess.call(["xdg-open", clean_txt_path])
    except Exception as e:
        print(f"\n{Colors.RED}Could not automatically open the text file: {e}{Colors.ENDC}")
        print("Please open it manually to clean it before generating prompts.")

    return book_name, project_folder

def cleanup_comfyui_output_for_project(project_path, config):
    """Safely finds and deletes generated images from the ComfyUI output folder that match this project."""
    clear_screen()
    print("--- ComfyUI Output Cleanup ---")
    
    # --- THIS IS THE CORRECTED LOGIC ---
    # It now correctly looks inside the 'comfyui_settings' block first.
    comfy_settings = config.get("comfyui_settings", {})
    comfyui_path = comfy_settings.get("comfyui_path")

    if not comfyui_path or not os.path.isdir(comfyui_path):
        print(f"{Colors.RED}ERROR: 'comfyui_path' is not set or is invalid in your project's config.json.{Colors.ENDC}")
        return

    comfy_output_dir = os.path.join(comfyui_path, "output")
    if not os.path.isdir(comfy_output_dir):
        print(f"{Colors.RED}ERROR: Could not find ComfyUI output directory at '{comfy_output_dir}'{Colors.ENDC}")
        return

    # 1. Get a set of all base filenames (without extension) from our project
    project_basenames = set()
    images_dir = os.path.join(project_path, "images")
    upscaled_dir = os.path.join(project_path, "images_upscaled")

    if os.path.exists(images_dir):
        for f in os.listdir(images_dir):
            project_basenames.add(os.path.splitext(f)[0])
    if os.path.exists(upscaled_dir):
        for f in os.listdir(upscaled_dir):
            project_basenames.add(os.path.splitext(f)[0].replace("_upscaled", ""))

    if not project_basenames:
        print("No images found in the project to match against. Cleanup aborted.")
        return

    # 2. Find all matching files in the ComfyUI output directory
    files_to_delete = []
    for comfy_file in os.listdir(comfy_output_dir):
        comfy_basename_with_counter = os.path.splitext(comfy_file)[0]
        for proj_basename in project_basenames:
            if comfy_basename_with_counter.startswith(proj_basename):
                files_to_delete.append(os.path.join(comfy_output_dir, comfy_file))
                break 

    if not files_to_delete:
        print("No matching project files found in the ComfyUI output folder. Nothing to clean up.")
        return

    # 3. Ask for confirmation
    print(f"\nFound {Colors.YELLOW}{len(files_to_delete)}{Colors.ENDC} files in the ComfyUI output folder matching this project:")
    for f in files_to_delete[:5]:
        print(f"  - {os.path.basename(f)}")
    if len(files_to_delete) > 5:
        print(f"  - ... and {len(files_to_delete) - 5} more.")

    confirm = input("\nAre you sure you want to PERMANENTLY delete these files? (y/n): ").lower()

    # 4. Delete if confirmed
    if confirm == 'y':
        deleted_count = 0
        for f_path in files_to_delete:
            try:
                os.remove(f_path)
                deleted_count += 1
            except OSError as e:
                print(f"  -> {Colors.RED}ERROR: Could not delete {os.path.basename(f_path)}. Reason: {e}{Colors.ENDC}")
        print(f"\nSuccessfully deleted {deleted_count} files.")
    else:
        print("\nCleanup cancelled.")

def integrate_refusals(project_path, silent_mode=False):
    """
    Parses refusals.log and attempts to merge valid PROMPT/QUOTE pairs 
    back into the project's main CSV file.
    
    Args:
        project_path: Path to the project folder.
        silent_mode: If True, suppresses UI prompts/clears and auto-archives if successful.
                     Returns the number of recovered entries.
    """
    if not silent_mode:
        clear_screen()
        print("--- Refusal Recovery Tool ---")
    
    project_name = os.path.basename(project_path)
    refusal_log_path = os.path.join(project_path, "refusals.log")
    csv_path = os.path.join(project_path, f"{project_name}_prompts.csv")

    if not os.path.exists(refusal_log_path):
        if not silent_mode:
            print(f"{Colors.YELLOW}No refusals.log found.{Colors.ENDC}")
        return 0

    if not silent_mode:
        print(f"Reading {refusal_log_path}...")
        
    with open(refusal_log_path, 'r', encoding='utf-8') as f:
        log_content = f.read()

    # Split by the dashed lines
    raw_blocks = re.split(r'-{10,}', log_content)
    
    recovered_entries = []

    for block in raw_blocks:
        if not block.strip(): continue

        # 1. Check for Header
        header_match = re.search(r"Chapter\s+(\d+),\s+Scene\s+(\d+):", block, re.IGNORECASE)
        
        if header_match:
            chapter_num = int(header_match.group(1))
            scene_num = int(header_match.group(2))
            
            # 2. Parse Body
            body = block[header_match.end():].strip()
            
            parsed_prompt = ""
            parsed_quote = ""

            for line in body.splitlines():
                line = line.strip()
                if line.startswith("PROMPT:"):
                    parsed_prompt = line[len("PROMPT:"):].strip()
                elif line.startswith("QUOTE:"):
                    parsed_quote = line[len("QUOTE:"):].strip()
            
            # 3. STRICT CHECK: We ONLY accept if we found a "PROMPT:" tag.
            # This filters out the "I refuse to generate..." blocks which lack that tag.
            if parsed_prompt:
                recovered_entries.append({
                    'chapter': chapter_num,
                    'scene': scene_num,
                    'prompt': parsed_prompt,
                    'quote': parsed_quote or " "
                })

    if not recovered_entries:
        if not silent_mode:
            print(f"{Colors.RED}No valid 'PROMPT:' entries found in log.{Colors.ENDC}")
        return 0

    if not silent_mode:
        print(f"Found {len(recovered_entries)} potential recoveries.")

    # Load Existing CSV
    current_data = []
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path, sep='|')
            df.columns = df.columns.str.lower().str.strip()
            current_data = df.to_dict('records')
        except Exception:
            pass

    # Merge (Avoid Duplicates)
    existing_keys = set((int(row['chapter']), int(row['scene'])) for row in current_data)
    
    added_count = 0
    for entry in recovered_entries:
        key = (entry['chapter'], entry['scene'])
        if key not in existing_keys:
            current_data.append(entry)
            existing_keys.add(key)
            added_count += 1

    if added_count == 0:
        if not silent_mode:
            print(f"{Colors.YELLOW}All entries already exist in CSV.{Colors.ENDC}")
        return 0

    # Save and Sort
    new_df = pd.DataFrame(current_data)
    if 'prompt' in new_df.columns and 'quote' in new_df.columns:
        new_df = new_df[['chapter', 'scene', 'prompt', 'quote']]
    new_df = new_df.sort_values(by=['chapter', 'scene'])
    
    # Backup
    if os.path.exists(csv_path) and not silent_mode:
        import shutil
        shutil.copy(csv_path, csv_path + ".bak")

    new_df.to_csv(csv_path, sep='|', index=False)
    
    if not silent_mode:
        print(f"{Colors.GREEN}Successfully merged {added_count} entries.{Colors.ENDC}")

    # Auto-archive in silent mode, or Ask in interactive mode
    should_archive = False
    if silent_mode:
        should_archive = True
    else:
        print("\nArchive refusals.log? (y/n)")
        if input("> ").lower().strip() == 'y':
            should_archive = True

    if should_archive:
        try:
            os.rename(refusal_log_path, refusal_log_path + ".processed")
            if not silent_mode: print("Log archived.")
        except OSError:
            pass

    return added_count

def cleanup_global_comfyui_output():
    """Deletes all files in the configured ComfyUI output folder."""
    clear_screen()
    print(f"{Colors.RED}{Colors.BOLD}--- GLOBAL COMFYUI CLEANUP ---{Colors.ENDC}")
    
    try:
        global_config = load_global_config()
        
        # 1. Try finding path in top-level comfyui_settings
        comfy_path = global_config.get("comfyui_settings", {}).get("comfyui_path")
        
        # 2. If not found, try finding it in default_project_settings -> comfyui_settings
        if not comfy_path:
            comfy_path = global_config.get("default_project_settings", {}) \
                                      .get("comfyui_settings", {}) \
                                      .get("comfyui_path")

        if not comfy_path:
            print("ERROR: 'comfyui_path' not found in global_config.json.")
            print("Please ensure it is set under 'comfyui_settings' or 'default_project_settings'.")
            return
        
        # Check if the path is valid
        if not os.path.isdir(comfy_path):
            print(f"ERROR: The path '{comfy_path}' does not exist or is not a directory.")
            return

        output_dir = os.path.join(comfy_path, "output")
        if not os.path.exists(output_dir):
            print(f"Output directory not found at: {output_dir}"); return
            
        files = [f for f in os.listdir(output_dir) if os.path.isfile(os.path.join(output_dir, f))]
        
        if not files:
            print(f"ComfyUI output folder is already empty.\nLocation: {output_dir}"); return

        print(f"Found {len(files)} files in: {output_dir}")
        confirm = input(f"Type 'DELETE' to permanently remove all {len(files)} files: ")
        
        if confirm == 'DELETE':
            for f in files:
                try:
                    os.remove(os.path.join(output_dir, f))
                except Exception as e:
                    print(f"Failed to delete {f}: {e}")
            print(f"{Colors.GREEN}Cleanup complete.{Colors.ENDC}")
        else:
            print("Operation cancelled.")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def copy_project_config(source_project_path, target_project_path):
    """Copies config.json from source to target."""
    src_cfg = os.path.join(source_project_path, "config.json")
    tgt_cfg = os.path.join(target_project_path, "config.json")
    if os.path.exists(src_cfg):
        shutil.copy(src_cfg, tgt_cfg)
        print(f"  -> Applied configuration from {os.path.basename(source_project_path)}")
    else:
        print(f"  -> WARNING: Source config not found at {src_cfg}")

# This helper function needs to be here as well for the cleanup function to use it.
def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')
