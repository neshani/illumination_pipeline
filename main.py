# main.py
import os
import sys
import json

# Local imports
from src.project_manager import (
    ensure_project_folders_exist,
    find_projects,
    find_importable_epubs,
    create_project_structure,
    find_importable_transcripts,
    create_project_from_transcript,
    cleanup_global_comfyui_output, 
    copy_project_config
)
from src.llm_handler import run_single_text_test_suite, run_chunking_test_suite, generate_prompts_for_project
from src.image_generator import run_image_generation, run_upscaling_process, embed_quotes_into_images
from src.style_tester import run_style_tester
from src.config_manager import load_global_config
from src.utils import Colors, open_folder_in_explorer, open_file, get_menu_choice, get_char, start_comfyui, archive_folder

BATCH_FILE = "current_batch.json"

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def press_enter_to_continue():
    print("\nPress any key to return...")
    get_char()

def handle_import_new_book():
    clear_screen()
    print("=== Import New Book ===")
    epubs = find_importable_epubs()
    transcripts = find_importable_transcripts()

    if not epubs and not transcripts:
        print(f"\n{Colors.YELLOW}No new files found in the 'Books' or 'Transcripts' folders.{Colors.ENDC}")
    else:
        if epubs:
            print("\nFound the following new books (EPUB):")
            for i, epub in enumerate(epubs):
                print(f"[{i+1}] {epub}")
        if transcripts:
            print("\nFound the following new transcripts (TXT):")
            # Start numbering transcripts after the epubs
            for i, transcript in enumerate(transcripts):
                print(f"[{i + len(epubs) + 1}] {transcript}")

    print("\n---------------------------")
    print("(O)pen Project Folders")
    print("(B)ack to Main Menu")

    choice = input("\nSelect a file to import or an option: ").lower().strip()
    if choice == 'b': return None
    if choice == 'o':
        open_folder_in_explorer("Books")
        open_folder_in_explorer("Transcripts")
        return "refresh"

    try:
        index = int(choice) - 1
        if 0 <= index < len(epubs):
            # User chose an EPUB
            return create_project_structure(epubs[index])
        elif len(epubs) <= index < len(epubs) + len(transcripts):
            # User chose a Transcript
            transcript_index = index - len(epubs)
            return create_project_from_transcript(transcripts[transcript_index])
        else:
            print("Invalid selection.")
    except (ValueError, IndexError):
        print("Invalid input.")

    press_enter_to_continue()
    return None

def handle_single_image(project_path):
    while True:
        clear_screen()
        print("--- Generate Single Fill-in Image ---")
        print("Enter details in the format: chapter_number: your prompt here")
        user_input = input("Example -> 04: a beautiful spaceship landing on a red planet\n> ")
        try:
            parts = user_input.split(':', 1)
            if len(parts) != 2: raise ValueError("Input must contain a colon ':'")
            chapter_num = int(parts[0].strip())
            prompt_part = parts[1].strip()
            if not prompt_part: raise ValueError("Prompt cannot be empty.")
            run_image_generation(project_path, single_image_details=(chapter_num, prompt_part))
        except (ValueError, IndexError) as e:
            print(f"\n{Colors.RED}Invalid format. Please try again. ({e}){Colors.ENDC}")
        
        print("\nGenerate another fill-in image? (y/n): ", end="", flush=True)
        another = get_menu_choice()
        if another != 'y': break

def handle_project_menu(project_name, project_path):
    while True:
        clear_screen()
        print(f"=== Project: {Colors.CYAN}{project_name}{Colors.ENDC} ===")
        
        from src.config_manager import load_project_config
        config = load_project_config(project_path)
        
        is_comfy_project = config.get("common_settings", {}).get("image_generator_type") == "comfyui"

        prompts_exist = os.path.exists(os.path.join(project_path, f"{project_name}_prompts.csv"))
        
        if not prompts_exist:
            print("Status: Ready for Prompt Generation.")
            print("\n[1] Generate prompts from book text")
            print("(O)pen Project Folder")
            print("(B)ack to Main Menu")
            print("\nSelect an option: ", end="", flush=True)
            choice = get_menu_choice()
            if choice == '1': generate_prompts_for_project(project_path); press_enter_to_continue()
            elif choice == 'o': open_folder_in_explorer(project_path)
            elif choice == 'b': return
        else:
            print("Status: Ready for Image Generation.")
            print("\n[1] Generate/Continue image generation")
            print("[2] Generate a single fill-in image")
            print("[3] Upscale generated images")
            print("[4] Embed quotes into images")  # NEW MENU OPTION
            print("---------------------------")
            print("(O)pen Project Folder")
            if is_comfy_project:
                print(f"(C)lean up ComfyUI Output Folder")
            print("(R)e-run prompt generation (deletes existing prompts)")
            print("(B)ack to Main Menu")
            print("\nSelect an option: ", end="", flush=True)
            choice = get_menu_choice()

            if choice == '1': run_image_generation(project_path); press_enter_to_continue()
            elif choice == '2': handle_single_image(project_path)
            elif choice == '3': run_upscaling_process(project_path); press_enter_to_continue()
            elif choice == '4':  # NEW HANDLER
                embed_quotes_into_images(project_path)
                press_enter_to_continue()
            elif choice == 'o': open_folder_in_explorer(project_path)
            elif choice == 'c' and is_comfy_project:
                from src.project_manager import cleanup_comfyui_output_for_project
                cleanup_comfyui_output_for_project(project_path, config)
                press_enter_to_continue()
            elif choice == 'r':
                print("\nAre you sure? (y/n): ", end="", flush=True)
                if get_menu_choice() == 'y':
                    os.remove(os.path.join(project_path, f"{project_name}_prompts.csv"))
            elif choice == 'b': return

def handle_testing_menu():
    while True:
        clear_screen()
        print("--- Testing Suites ---")
        print("\n[1] Chunking & Prompt Test (Live Data)")
        print("[2] Single Text Test (from llm_test_input.txt)")
        print("(B)ack to Main Menu")
        print("\nSelect a test to run: ", end="", flush=True)
        choice = get_menu_choice()
        if choice == '1': handle_chunking_test()
        elif choice == '2': run_single_text_test_suite(); press_enter_to_continue()
        elif choice == 'b': return

def handle_chunking_test():
    clear_screen()
    print("--- Chunking & Prompt Test ---")
    projects = find_projects()
    if not projects: print("No projects found. Please import a book first."); press_enter_to_continue(); return
    print("\nSelect a project to test against:")
    for i, (name, path) in enumerate(projects): print(f"[{i+1}] {name}")
    try:
        proj_index = int(input("\nProject number: ")) - 1
        if not (0 <= proj_index < len(projects)): raise IndexError
        project_path = projects[proj_index][1]
        num_chunks = int(input("How many chunks from the book would you like to test? "))
        if num_chunks <= 0: raise ValueError("Please enter a positive number.")
        run_chunking_test_suite(project_path, num_chunks)
    except (ValueError, IndexError):
        print("Invalid input.")
    press_enter_to_continue()

def save_batch_queue(project_paths):
    """Saves the list of project paths to a JSON file."""
    try:
        with open(BATCH_FILE, 'w') as f:
            json.dump(project_paths, f)
    except Exception as e:
        print(f"Error saving batch file: {e}")

def load_batch_queue():
    """Loads the list of project paths."""
    if not os.path.exists(BATCH_FILE): return []
    try:
        with open(BATCH_FILE, 'r') as f:
            return json.load(f)
    except:
        return []

def handle_batch_processing():
    while True:
        clear_screen()
        current_queue = load_batch_queue()
        queue_status = f"{len(current_queue)} projects in queue" if current_queue else "Empty"
        
        print(f"{Colors.BOLD}--- Batch Processing Menu ---{Colors.ENDC}")
        print(f"Current Batch Status: {Colors.CYAN}{queue_status}{Colors.ENDC}")
        print("\n--- Actions ---")
        print("[1] New Batch from Imports (Epub/Txt)")
        print("[2] New Batch from Existing Projects")
        
        if current_queue:
            print(f"[3] {Colors.GREEN}RESUME BATCH{Colors.ENDC} (Process Queue)")
            print("[4] Apply Style to Current Batch")
            print("[5] Embed Quotes into Images (Current Batch)") # NEW
            print("[6] Archive Images for Current Batch (Start Over)")
            print("[7] Clear Batch Queue")
        
        print("\n(B)ack")
        
        choice = get_menu_choice()
        
        if choice == '1': setup_new_batch_imports()
        elif choice == '2': setup_new_batch_existing()
        elif choice == '3' and current_queue: run_batch_execution_loop(current_queue)
        elif choice == '4' and current_queue: handle_batch_style_copy(current_queue)
        elif choice == '5' and current_queue: handle_batch_quote_embedding(current_queue) # NEW
        elif choice == '6' and current_queue: handle_batch_archive(current_queue)
        elif choice == '7': 
            if os.path.exists(BATCH_FILE): os.remove(BATCH_FILE)
        elif choice == 'b': return

def handle_batch_quote_embedding(project_paths):
    """Iterates through the batch and embeds quotes for each."""
    clear_screen()
    print(f"--- Batch Quote Embedding ({len(project_paths)} projects) ---")
    
    for i, path in enumerate(project_paths):
        project_name = os.path.basename(path)
        # We don't clear screen here to keep a scrolling log of progress
        print(f"\n{Colors.BOLD}>>> Processing Project [{i+1}/{len(project_paths)}]: {project_name} <<<{Colors.ENDC}")
        
        # We call the existing logic
        embed_quotes_into_images(path)
        
    print(f"\n{Colors.GREEN}Batch Quote Embedding Complete.{Colors.ENDC}")
    press_enter_to_continue()

def setup_new_batch_imports():
    clear_screen()
    print("--- New Batch: Imports ---")
    epubs = find_importable_epubs()
    transcripts = find_importable_transcripts()
    
    if not epubs and not transcripts:
        print("No new files found."); press_enter_to_continue(); return

    all_files = epubs + transcripts
    print("\nAvailable files:")
    for i, f in enumerate(all_files):
        ftype = "EPUB" if i < len(epubs) else "TXT"
        print(f"[{i+1}] {ftype}: {f}")
        
    selection = input("\nEnter comma-separated numbers (e.g., 1,3,4): ")
    try:
        indices = [int(x.strip())-1 for x in selection.split(',') if x.strip().isdigit()]
        selected_files = [all_files[i] for i in indices if 0 <= i < len(all_files)]
    except:
        print("Invalid selection."); press_enter_to_continue(); return

    if not selected_files: return

    created_paths = []
    print("\n--- Creating Project Structures ---")
    for f in selected_files:
        if f.endswith('.epub'):
            res = create_project_structure(f)
        else:
            res = create_project_from_transcript(f)
        if res: created_paths.append(res[1]) # res is (name, path), we just keep path

    if created_paths:
        save_batch_queue(created_paths)
        print(f"\n{len(created_paths)} projects added to batch queue.")
        
        # Ask for style immediately
        print("\nApply a specific style/config to these new projects?")
        if get_menu_choice() == 'y':
            handle_batch_style_copy(created_paths)
            
        run_batch_execution_loop(created_paths)

def setup_new_batch_existing():
    clear_screen()
    print("--- New Batch: Existing Projects ---")
    projects = find_projects() # Returns list of (name, path)
    if not projects: print("No projects found."); press_enter_to_continue(); return

    print("\nSelect projects to add to batch:")
    for i, (name, _) in enumerate(projects):
        print(f"[{i+1}] {name}")

    selection = input("\nEnter comma-separated numbers (e.g., 1,3): ")
    try:
        indices = [int(x.strip())-1 for x in selection.split(',') if x.strip().isdigit()]
        selected_projects = [projects[i][1] for i in indices if 0 <= i < len(projects)]
    except:
        print("Invalid selection."); press_enter_to_continue(); return

    if selected_projects:
        save_batch_queue(selected_projects)
        print(f"\n{len(selected_projects)} projects added to batch queue.")
        
        print("\nApply a specific style/config to this batch?")
        if input("(y/n): ").lower() == 'y':
             handle_batch_style_copy(selected_projects)
             
        run_batch_execution_loop(selected_projects)

def handle_batch_style_copy(project_paths):
    clear_screen()
    print("--- Apply Style to Batch ---")
    projects = find_projects()
    
    if not projects:
        print("No projects found to copy styles from.")
        press_enter_to_continue()
        return

    print("\nSelect Source Project (Template):")
    for i, (name, _) in enumerate(projects):
        print(f"[{i+1}] {name}")
    
    try:
        user_input = input("\nSource #: ")
        idx = int(user_input) - 1
        
        # Check if selection is valid
        if idx < 0 or idx >= len(projects):
            print(f"{Colors.RED}Invalid selection number.{Colors.ENDC}")
            press_enter_to_continue()
            return

        source_path = projects[idx][1]
        source_abs = os.path.abspath(source_path)
        
        print(f"\nCopying style from '{projects[idx][0]}'...")

        for target_path in project_paths:
            target_abs = os.path.abspath(target_path)
            target_name = os.path.basename(target_path)

            # CRITICAL FIX: Skip if source and target are the same folder
            if source_abs == target_abs:
                print(f"  -> Skipping '{target_name}' (Cannot copy style to itself)")
                continue

            try:
                copy_project_config(source_path, target_path)
                print(f"  -> Applied to '{target_name}'")
            except Exception as e:
                print(f"  -> {Colors.RED}Failed on '{target_name}': {e}{Colors.ENDC}")

        print(f"\n{Colors.GREEN}Batch Style Application Complete.{Colors.ENDC}")
        press_enter_to_continue()

    except ValueError:
        print(f"{Colors.RED}Invalid input. Please enter a number.{Colors.ENDC}")
        press_enter_to_continue()
    except Exception as e:
        # This catches unexpected errors and prints them so you can debug
        print(f"\n{Colors.RED}An unexpected error occurred: {e}{Colors.ENDC}")
        press_enter_to_continue()

def handle_batch_archive(project_paths):
    clear_screen()
    print(f"{Colors.RED}WARNING: This will rename the 'images' folder for ALL projects in the batch.{Colors.ENDC}")
    print("This forces the system to regenerate all images from scratch.")
    if input("Are you sure? (type 'archive'): ") == 'archive':
        for path in project_paths:
            archive_folder(path, "images")
        print("Archive complete.")
        press_enter_to_continue()

def run_batch_execution_loop(project_paths):
    """
    The Smart Loop:
    Iterates through projects. 
    1. Checks if prompts exist. If not, generates them.
    2. Checks if images need generating. Generates missing ones.
    3. Handles interruptions gracefully.
    """
    clear_screen()
    print(f"--- Starting Batch Execution ({len(project_paths)} projects) ---")
    print(f"{Colors.YELLOW}Press 'X' during image generation to stop the ENTIRE batch.{Colors.ENDC}\n")

    for i, path in enumerate(project_paths):
        project_name = os.path.basename(path)
        print(f"{Colors.BOLD}>>> Processing Project [{i+1}/{len(project_paths)}]: {project_name} <<<{Colors.ENDC}")

        # 1. Prompt Check
        csv_path = os.path.join(path, f"{project_name}_prompts.csv")
        if not os.path.exists(csv_path):
            print(f"  -> Prompts not found. Generating...")
            generate_prompts_for_project(path)
        else:
            print(f"  -> Prompts found. Skipping prompt generation.")

        # 2. Image Check & Generation
        # Ensure image folder exists
        os.makedirs(os.path.join(path, "images"), exist_ok=True)
        
        print(f"  -> Starting Image Generation (Fill-in mode)...")
        was_interrupted = run_image_generation(path)
        
        if was_interrupted:
            print(f"\n{Colors.RED}Batch Execution Stopped by User.{Colors.ENDC}")
            press_enter_to_continue()
            return # Exit the loop and function completely

    print(f"\n{Colors.GREEN}Batch Cycle Complete.{Colors.ENDC}")
    press_enter_to_continue()

def main():
    os.system("") # Enable ANSI colors on Windows
    ensure_project_folders_exist()
    while True:
        clear_screen()
        print(f"{Colors.BOLD}=== Illumination Pipeline V2 ==={Colors.ENDC}")
        projects = find_projects()
        if projects:
            print("\nExisting Illuminations:")
            for i, (name, path) in enumerate(projects): print(f"[{i+1}] {name}")
        else:
            print("\nNo projects found.")
        print("\n---------------------------")
        print("(I)mport New Book from 'Books' folder")
        print("(B)atch Processing") 
        print("(S)tart ComfyUI")
        print("(L)ibrary Generator (Style Tester)")
        print("(T)esting Suites")
        print("(C)leanup Global ComfyUI Output")
        print("(G)lobal Settings (config file)")
        print("(Q)uit")
        
        choice = input("\nSelect a project or an option: ").lower().strip()

        if choice == 'q': break
        elif choice == 'i': 
            result = handle_import_new_book()
            while result == "refresh":
                result = handle_import_new_book()
            if result:
                press_enter_to_continue()
                handle_project_menu(*result)
        elif choice == 'b': handle_batch_processing()
        elif choice == 'c': 
            cleanup_global_comfyui_output()
            press_enter_to_continue()
        elif choice == 's':
            try:
                global_config = load_global_config()
                script_path = global_config.get("comfyui_settings", {}).get("startup_script")
                start_comfyui(script_path)
            except FileNotFoundError:
                print(f"{Colors.RED}ERROR: global_config.json not found!{Colors.ENDC}")
            except Exception as e:
                print(f"{Colors.RED}An unexpected error occurred: {e}{Colors.ENDC}")
            press_enter_to_continue()
        elif choice == 'l':
            run_style_tester()
            press_enter_to_continue()
        elif choice == 't': handle_testing_menu()
        elif choice == 'g':
            open_file('global_config.json')
        else:
            try:
                index = int(choice) - 1
                if 0 <= index < len(projects):
                    handle_project_menu(*projects[index])
            except (ValueError, IndexError):
                print("Invalid input.")

if __name__ == "__main__":
    main()