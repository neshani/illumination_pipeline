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

# Local imports
from src.config_manager import get_default_project_config
from src.utils import Colors

# Suppress the specific FutureWarning from ebooklib
warnings.filterwarnings("ignore", category=FutureWarning, module='ebooklib.epub')

BOOKS_FOLDER = "Books"
ILLUMINATIONS_FOLDER = "Illuminations"

def ensure_project_folders_exist():
    """Creates the core input/output folders if they don't exist."""
    os.makedirs(BOOKS_FOLDER, exist_ok=True)
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
            paragraphs = [p.get_text(strip=True) for p in soup.find_all('p') if p.get_text(strip=True)]
            if paragraphs:
                full_text_parts.append('\n\n'.join(paragraphs))
                full_text_parts.append("\n\n==CHAPTER==\n\n")
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

# This helper function needs to be here as well for the cleanup function to use it.
def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')