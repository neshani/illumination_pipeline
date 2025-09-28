# main.py
import os
import sys

# Local imports
from src.project_manager import (
    ensure_project_folders_exist,
    find_projects,
    find_importable_epubs,
    create_project_structure,
)
from src.llm_handler import run_single_text_test_suite, run_chunking_test_suite, generate_prompts_for_project
from src.image_generator import run_image_generation, run_upscaling_process
from src.style_tester import run_style_tester
from src.utils import Colors, open_folder_in_explorer, open_file, get_menu_choice, get_char

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def press_enter_to_continue():
    print("\nPress any key to return...")
    get_char()

def handle_import_new_book():
    clear_screen()
    print("=== Import New Book ===")
    epubs = find_importable_epubs()
    
    if epubs:
        print("Found the following new books in the 'Books' folder:")
        for i, epub in enumerate(epubs): print(f"[{i+1}] {epub}")
    else:
        print(f"\n{Colors.YELLOW}No new .epub files found in the 'Books' folder.{Colors.ENDC}")
        print("Add your .epub files there to get started.")

    print("\n---------------------------")
    print("(O)pen 'Books' Folder")
    print("(B)ack to Main Menu")
    
    # MODIFIED: Use standard input() to allow for multi-digit numbers
    choice = input("\nSelect a book to import or an option: ").lower().strip()
    if choice == 'b': return None
    if choice == 'o':
        open_folder_in_explorer("Books")
        return "refresh" # Special signal to re-run this menu

    try:
        index = int(choice) - 1
        if 0 <= index < len(epubs):
            return create_project_structure(epubs[index])
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
        # UNCHANGED: Still uses fast get_menu_choice for y/n
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
            # UNCHANGED: This is a fixed menu, so it uses get_menu_choice
            choice = get_menu_choice()
            if choice == '1': generate_prompts_for_project(project_path); press_enter_to_continue()
            elif choice == 'o': open_folder_in_explorer(project_path)
            elif choice == 'b': return
        else:
            print("Status: Ready for Image Generation.")
            print("\n[1] Generate/Continue image generation")
            print("[2] Generate a single fill-in image")
            print("[3] Upscale generated images")
            print("---------------------------")
            print("(O)pen Project Folder")
            if is_comfy_project:
                print(f"(C)lean up ComfyUI Output Folder")
            print("(R)e-run prompt generation (deletes existing prompts)")
            print("(B)ack to Main Menu")
            print("\nSelect an option: ", end="", flush=True)
            # UNCHANGED: This is a fixed menu, so it uses get_menu_choice
            choice = get_menu_choice()

            if choice == '1': run_image_generation(project_path); press_enter_to_continue()
            elif choice == '2': handle_single_image(project_path)
            elif choice == '3': run_upscaling_process(project_path); press_enter_to_continue()
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
        # UNCHANGED: This is a fixed menu, so it uses get_menu_choice
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
        # UNCHANGED: This prompt was already using input()
        proj_index = int(input("\nProject number: ")) - 1
        if not (0 <= proj_index < len(projects)): raise IndexError
        project_path = projects[proj_index][1]
        num_chunks = int(input("How many chunks from the book would you like to test? "))
        if num_chunks <= 0: raise ValueError("Please enter a positive number.")
        run_chunking_test_suite(project_path, num_chunks)
    except (ValueError, IndexError):
        print("Invalid input.")
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
        print("(S)tyle Library Generator")
        print("(T)esting Suites")
        print("(G)lobal Settings (config file)")
        print("(Q)uit")
        
        # MODIFIED: Use standard input() to allow for multi-digit numbers
        choice = input("\nSelect a project or an option: ").lower().strip()

        if choice == 'q': break
        elif choice == 'i': 
            result = handle_import_new_book()
            while result == "refresh": # Loop back if user opened folder
                result = handle_import_new_book()
            if result:
                press_enter_to_continue()
                handle_project_menu(*result)
        elif choice == 's':
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