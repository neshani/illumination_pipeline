# src/style_tester.py
import os
import shutil
import json
import pandas as pd
from pathlib import Path

from src.config_manager import get_default_project_config
from src.image_generator import run_image_generation
from src.utils import Colors

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def run_style_tester():
    """Main function to run the style generation utility."""
    clear_screen()
    print(f"{Colors.BOLD}--- Style Library Generator ---{Colors.ENDC}")
    print("This utility will generate a sample of images for each LoRA in your input file.")
    
    # --- NEW: Read prompts from an external file ---
    PROMPTS_FILE_PATH = "style_prompts.txt"
    if not os.path.exists(PROMPTS_FILE_PATH):
        print(f"\n{Colors.RED}ERROR: Prompts file not found at '{PROMPTS_FILE_PATH}'.{Colors.ENDC}")
        print("Please create this file in the root directory and add one prompt per line.")
        return

    try:
        with open(PROMPTS_FILE_PATH, 'r', encoding='utf-8') as f:
            # Read all lines, strip whitespace from each, and ignore any that are empty
            style_test_prompts = [line.strip() for line in f if line.strip()]
    except Exception as e:
        print(f"\n{Colors.RED}ERROR: Could not read prompts file: {e}{Colors.ENDC}")
        return

    if not style_test_prompts:
        print(f"\n{Colors.YELLOW}WARNING: No prompts found in '{PROMPTS_FILE_PATH}'. Aborting.{Colors.ENDC}")
        return
    # --- END NEW LOGIC ---

    lora_file_path = input("\nEnter the path to your LoRA input file (e.g., loras.txt): ").strip()
    if not os.path.exists(lora_file_path):
        print(f"\n{Colors.RED}ERROR: File not found at '{lora_file_path}'. Aborting.{Colors.ENDC}")
        return

    STYLES_FOLDER = "Styles"
    ILLUMINATIONS_FOLDER = "Illuminations"
    TEMP_PROJECT_NAME = "_temp_style_gen"
    TEMP_PROJECT_PATH = os.path.join(ILLUMINATIONS_FOLDER, TEMP_PROJECT_NAME)
    
    os.makedirs(STYLES_FOLDER, exist_ok=True)
    os.makedirs(ILLUMINATIONS_FOLDER, exist_ok=True)

    try:
        with open(lora_file_path, 'r', encoding='utf-8') as f:
            lora_entries = [line for line in f if line.strip() and '|' in line]
    except Exception as e:
        print(f"\n{Colors.RED}ERROR: Could not read LoRA file: {e}{Colors.ENDC}")
        return

    print(f"\nFound {len(lora_entries)} LoRA(s) and {len(style_test_prompts)} prompts to process.")

    for i, line in enumerate(lora_entries):
        try:
            lora_name, trigger_words = [part.strip() for part in line.split('|', 1)]
            lora_basename = Path(lora_name).stem
            final_style_path = os.path.join(STYLES_FOLDER, lora_basename)

            print("\n" + "="*80)
            print(f"Processing LoRA {i+1}/{len(lora_entries)}: {Colors.CYAN}{lora_basename}{Colors.ENDC}")

            if os.path.exists(final_style_path):
                print(f"{Colors.YELLOW}  -> Style folder already exists. Skipping.{Colors.ENDC}")
                continue

            if os.path.exists(TEMP_PROJECT_PATH):
                shutil.rmtree(TEMP_PROJECT_PATH)
            os.makedirs(TEMP_PROJECT_PATH)

            print("  -> Creating temporary configuration...")
            config = get_default_project_config()
            config["comfyui_settings"]["workflow_overrides"]["lora_name"] = lora_name
            config["common_settings"]["prompt_prefix"] = f"{trigger_words}, "
            
            temp_config_path = os.path.join(TEMP_PROJECT_PATH, "config.json")
            with open(temp_config_path, 'w') as f: json.dump(config, f, indent=4)
            
            print("  -> Generating temporary prompts file...")
            # Use the list of prompts we loaded from the file
            prompts_data = [{'chapter': 1, 'scene': i+1, 'prompt': p} for i, p in enumerate(style_test_prompts)]
            df = pd.DataFrame(prompts_data)
            temp_csv_path = os.path.join(TEMP_PROJECT_PATH, f"{TEMP_PROJECT_NAME}_prompts.csv")
            df.to_csv(temp_csv_path, sep='|', index=False)

            print("  -> Starting image generation batch...")
            was_interrupted = run_image_generation(TEMP_PROJECT_PATH)

            generated_images_path = os.path.join(TEMP_PROJECT_PATH, "images")
            if os.path.exists(generated_images_path) and os.listdir(generated_images_path):
                shutil.move(generated_images_path, final_style_path)
                shutil.copy(temp_config_path, os.path.join(final_style_path, "config.json"))
                print(f"  -> {Colors.GREEN}SUCCESS:{Colors.ENDC} Style package for '{lora_basename}' created in '{STYLES_FOLDER}'.")
            else:
                if not was_interrupted:
                    print(f"  -> {Colors.YELLOW}WARNING:{Colors.ENDC} No images were generated for this LoRA. Check for errors.")

            if was_interrupted:
                print(f"\n{Colors.YELLOW}Halting Style Library generation due to user request.{Colors.ENDC}")
                break

        except Exception as e:
            print(f"\n{Colors.RED}An unexpected error occurred while processing '{line.strip()}': {e}{Colors.ENDC}")
            print("  -> Moving to the next item.")
            continue
    
    if os.path.exists(TEMP_PROJECT_PATH):
        shutil.rmtree(TEMP_PROJECT_PATH)
        
    print("\n" + "="*80)
    print(f"{Colors.BOLD}Style Library Generation Complete.{Colors.ENDC}")