# src/image_generator.py
import os
import json
import base64
import copy
import threading
import time
import pandas as pd
import requests
import uuid
import websocket
import random
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed

# NEW IMPORTS FOR METADATA EMBEDDING
from PIL import Image, PngImagePlugin

from src.config_manager import load_project_config, load_global_config
from src.utils import Colors

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

# --- KeyPressListener class remains unchanged ---
try:
    import msvcrt
    class KeyPressListener:
        def __init__(self, interrupt_key='x'): self.interrupt_key=interrupt_key.lower();self.key_pressed=None;self._thread=threading.Thread(target=self._listen,daemon=True);self._stop_event=threading.Event()
        def _listen(self):
            while not self._stop_event.is_set():
                if msvcrt.kbhit():
                    key=msvcrt.getch().decode('utf-8').lower()
                    if key==self.interrupt_key:self.key_pressed=key;break
                time.sleep(0.1)
        def start(self):self._thread.start()
        def stop(self):self._stop_event.set()
        def is_interrupt_pressed(self):return self.key_pressed==self.interrupt_key
except ImportError:
    import sys,select,tty,termios
    class KeyPressListener:
        def __init__(self,interrupt_key='x'):self.interrupt_key=interrupt_key.lower();self.key_pressed=None;self._thread=threading.Thread(target=self._listen,daemon=True);self._stop_event=threading.Event()
        def _listen(self):
            old_settings=termios.tcgetattr(sys.stdin)
            try:
                tty.setcbreak(sys.stdin.fileno())
                while not self._stop_event.is_set():
                    if select.select([sys.stdin],[],[],0.1)[0]:
                        key=sys.stdin.read(1).lower()
                        if key==self.interrupt_key:self.key_pressed=key;break
            finally:termios.tcsetattr(sys.stdin,termios.TCSADRAIN,old_settings)
        def start(self):self._thread.start()
        def stop(self):self._stop_event.set()
        def is_interrupt_pressed(self):return self.key_pressed==self.interrupt_key
# --- End of KeyPressListener class ---

def _create_filename_base_from_prompt(prompt_text):
    first_words = prompt_text.split()[:6]; base = "_".join(first_words)
    return "".join(c for c in base if c.isalnum() or c == '_').lower()

# --- NEW FUNCTION FOR EMBEDDING QUOTES ---
def _embed_quote_worker(image_path, quote):
    """Worker function to embed a quote into a single image. Returns status and path."""
    try:
        image = Image.open(image_path)
        metadata = PngImagePlugin.PngInfo()
        metadata.add_text("Quote", str(quote))
        image.save(image_path, "PNG", pnginfo=metadata)
        return "embedded", image_path
    except Exception as e:
        return "error", image_path, str(e)

def embed_quotes_into_images(project_path):
    """Reads the project's CSV file and embeds the text from the 'quote' column 
       into the metadata of corresponding PNG files using multiple threads."""
    # clear_screen() # Optional: Commented out to allow batch logs to scroll
    print("--- Embedding Quotes into Image Metadata (Multi-threaded) ---")

    try:
        global_config = load_global_config()
        max_workers = global_config.get("threading_settings", {}).get("quote_embedding_threads", 4)
    except FileNotFoundError:
        print(f"{Colors.YELLOW}WARNING: global_config.json not found. Defaulting to 4 threads.{Colors.ENDC}")
        max_workers = 4

    project_name = os.path.basename(project_path)
    csv_path = os.path.join(project_path, f"{project_name}_prompts.csv")
    images_folder = os.path.join(project_path, "images")

    if not os.path.exists(csv_path):
        print(f"{Colors.RED}ERROR: Prompts CSV file not found at '{csv_path}'. Cannot embed quotes.{Colors.ENDC}"); return
    if not os.path.exists(images_folder):
        print(f"{Colors.RED}ERROR: 'images' folder not found. Cannot embed quotes.{Colors.ENDC}"); return

    try:
        df = pd.read_csv(csv_path, sep='|')
        if 'quote' not in df.columns:
            print(f"{Colors.RED}ERROR: The CSV file does not contain a 'quote' column.{Colors.ENDC}"); return
    except Exception as e:
        print(f"{Colors.RED}ERROR: Failed to read or parse the CSV file: {e}{Colors.ENDC}"); return

    tasks = []
    skipped_count = 0; not_found_count = 0
    print(f"Found {len(df)} entries. Preparing tasks...")

    for index, row in df.iterrows():
        # --- FIX: Skip invalid prompts (Prevents crash on None/Float) ---
        if pd.isna(row['prompt']) or str(row['prompt']).strip().lower() == 'none' or not str(row['prompt']).strip():
            # If prompt is invalid, we can't determine the filename, so we skip
            continue

        filename_base = _create_filename_base_from_prompt(row['prompt'])
        filename = f"{str(row['chapter']).zfill(2)}-{str(row['scene']).zfill(2)}_{filename_base}.png"
        image_path = os.path.join(images_folder, filename)
        quote = row.get('quote')

        if pd.isna(quote) or str(quote).strip() == "":
            skipped_count += 1; continue
        if not os.path.exists(image_path):
            not_found_count += 1; continue
        tasks.append((image_path, quote))

    if not tasks:
        print("\nNo images found that need quotes embedded.")
    else:
        print(f"Starting embedding process for {len(tasks)} images using {max_workers} threads...")
        embedded_count = 0; error_count = 0
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {executor.submit(_embed_quote_worker, image_path, quote): (image_path, quote) for image_path, quote in tasks}
            for future in as_completed(future_to_task):
                try:
                    status, path, *error_msg = future.result()
                    if status == "embedded":
                        embedded_count += 1
                        print(f"  -> Embedded quote in: {os.path.basename(path)}")
                    elif status == "error":
                        error_count += 1
                        print(f"{Colors.RED}  -> ERROR: Failed to embed quote in {os.path.basename(path)}. Reason: {error_msg[0]}{Colors.ENDC}")
                except Exception as e:
                    error_count += 1
                    print(f"{Colors.RED}  -> ERROR: An unexpected exception occurred for a task. Reason: {e}{Colors.ENDC}")
        print(f"\n{Colors.GREEN}Successfully embedded quotes in {embedded_count} images.{Colors.ENDC}")
        if error_count > 0:
            print(f"{Colors.RED}{error_count} images failed to process due to errors.{Colors.ENDC}")

    print("\n--- Embedding Complete ---")
    if skipped_count > 0:
        print(f"{Colors.YELLOW}{skipped_count} rows were skipped due to missing quotes.{Colors.ENDC}")
    if not_found_count > 0:
         print(f"{Colors.YELLOW}{not_found_count} matching images were not found in the 'images' folder.{Colors.ENDC}")

def run_image_generation(project_path, single_image_details=None):
    """Router for image generation. Calls the correct function based on config."""
    config = load_project_config(project_path)
    generator_type = config.get("common_settings", {}).get("image_generator_type", "forge").lower()
    if generator_type == "comfyui":
        # Pass the return value up
        return run_comfyui_image_generation(project_path, config, single_image_details)
    else:
        # Pass the return value up
        return run_forge_image_generation(project_path, config, single_image_details)

def run_upscaling_process(project_path):
    """Router for upscaling. Calls the correct function based on config."""
    clear_screen()
    print("--- Post-Process Upscaling ---")
    config = load_project_config(project_path)
    generator_type = config.get("common_settings", {}).get("image_generator_type", "forge").lower()
    if generator_type == "comfyui":
        run_comfyui_upscaling(project_path, config)
    else:
        run_forge_upscaling(project_path, config)

def _queue_comfy_prompt(prompt_workflow, api_address, return_filename=False):
    """Sends workflow to ComfyUI and returns image data OR filename."""
    client_id = str(uuid.uuid4())
    try:
        p = {"prompt": prompt_workflow, "client_id": client_id}; data = json.dumps(p).encode('utf-8')
        req = requests.post(f"http://{api_address}/prompt", data=data); req.raise_for_status()
        prompt_id = req.json()['prompt_id']
        ws = websocket.WebSocket(); ws.connect(f"ws://{api_address}/ws?clientId={client_id}")
        output = None
        while True:
            out = ws.recv()
            if not isinstance(out, str): continue
            message = json.loads(out)
            if message['type'] == 'executed':
                data = message['data']
                if 'images' in data['output'] and data['prompt_id'] == prompt_id:
                    image_info = data['output']['images'][0]
                    if return_filename:
                        output = image_info['filename']
                    else:
                        resp = requests.get(f"http://{api_address}/view?filename={image_info['filename']}&subfolder={image_info['subfolder']}&type={image_info['type']}")
                        resp.raise_for_status(); output = resp.content
                    break
            if message['type'] == 'executing' and message['data']['node'] is None and message['data']['prompt_id'] == prompt_id:
                if output is None: print(f"{Colors.YELLOW}  -> WARNING: Workflow finished but no image output was detected.{Colors.ENDC}")
                break
        ws.close(); return output
    except Exception as e:
        print(f"  -> {Colors.RED}ERROR: An error occurred during ComfyUI communication: {e}{Colors.ENDC}"); return None

def run_comfyui_image_generation(project_path, config, single_image_details=None):
    """
    Runs the image generation process using a fully config-driven ComfyUI workflow.
    Returns True if interrupted by the user, otherwise False.
    """
    comfy_settings = config.get("comfyui_settings", {})
    common = config.get("common_settings", {})
    
    project_name = os.path.basename(project_path)
    csv_path = os.path.join(project_path, f"{project_name}_prompts.csv")
    images_folder = os.path.join(project_path, "images")
    os.makedirs(images_folder, exist_ok=True)

    try:
        with open(comfy_settings['generation_workflow'], 'r') as f: base_workflow = json.load(f)
    except Exception as e:
        print(f"ERROR: Could not load ComfyUI workflow '{comfy_settings.get('generation_workflow')}'. {e}"); return False

    overrides = comfy_settings.get("workflow_overrides", {})
    if overrides.get("enabled", False):
        print("  -> Applying ComfyUI workflow overrides from config file.")
        
        # 1. SMART LORA BYPASS LOGIC
        # If config says "None", we find the LoRA node, rewire connections around it, and delete it.
        lora_name = overrides.get("lora_name", "None")
        if not lora_name or str(lora_name).lower() == "none":
            lora_node_id = None
            source_node_id = None
            
            # Find the LoRA node and its source (Model input)
            for node_id, node_data in base_workflow.items():
                if node_data.get("class_type") == "LoraLoader":
                    lora_node_id = node_id
                    # Assuming Input 'model' connects to the previous node (CheckpointLoader)
                    # Comfy connection format: [NodeID, SlotIndex]
                    if "model" in node_data["inputs"]:
                        source_node_id = node_data["inputs"]["model"][0]
                    break
            
            # If found, bypass it
            if lora_node_id and source_node_id:
                print("  -> 'lora_name' is None. Bypassing and removing LoraLoader node.")
                for node_data in base_workflow.values():
                    if "inputs" not in node_data: continue
                    for input_name, input_val in node_data["inputs"].items():
                        # If a node expects input from the LoRA, point it to the LoRA's source instead
                        if isinstance(input_val, list) and input_val[0] == lora_node_id:
                            node_data["inputs"][input_name][0] = source_node_id
                
                # Delete the now-unused LoRA node
                del base_workflow[lora_node_id]

        # 2. STANDARD OVERRIDES
        # We iterate again to apply settings. If the LoRA node was deleted above, it won't be found here.
        for node in base_workflow.values():
            if "class_type" not in node: continue
            class_type = node["class_type"]
            inputs = node.get("inputs", {})

            if class_type in ["CheckpointLoader", "CheckpointLoaderSimple"]:
                if "ckpt_name" in overrides: inputs["ckpt_name"] = overrides["ckpt_name"]
            
            elif class_type == "LoraLoader":
                # This part was likely missing or broken in your current file
                if "lora_name" in overrides and overrides["lora_name"] != "None":
                    inputs["lora_name"] = overrides["lora_name"]
                if "lora_strength" in overrides:
                    inputs["strength_model"] = overrides["lora_strength"]
                    inputs["strength_clip"] = overrides["lora_strength"]

            elif class_type == "EmptyLatentImage":
                if "width" in overrides: inputs["width"] = overrides["width"]
                if "height" in overrides: inputs["height"] = overrides["height"]
                if "batch_size" in overrides: inputs["batch_size"] = overrides["batch_size"]

            elif class_type in ["KSampler", "KSamplerAdvanced"]:
                if "seed" in overrides: inputs["seed"] = overrides["seed"]
                if "steps" in overrides: inputs["steps"] = overrides["steps"]
                if "cfg" in overrides: inputs["cfg"] = overrides["cfg"]
                if "sampler_name" in overrides: inputs["sampler_name"] = overrides["sampler_name"]
                if "scheduler" in overrides: inputs["scheduler"] = overrides["scheduler"]
                if "denoise" in overrides: inputs["denoise"] = overrides["denoise"]
    else:
        print("  -> Overrides disabled. Only replacing prompt placeholders.")

    rows_to_process = []
    # ... [Row processing logic remains exactly the same] ...
    if single_image_details:
        chapter, prompt_text = single_image_details; scene_num = 1
        for f in sorted(os.listdir(images_folder)):
            if f.startswith(f"{int(chapter):02d}-"):
                try: scene_num = int(f[3:5]) + 1
                except (ValueError, IndexError): continue
        filename_base = _create_filename_base_from_prompt(prompt_text)
        filename_prefix = f"{str(chapter).zfill(2)}-{str(scene_num).zfill(2)}_{filename_base}"
        rows_to_process.append({'chapter': chapter, 'scene': scene_num, 'prompt': prompt_text, 'filename_prefix': filename_prefix})
    else:
        df = pd.read_csv(csv_path, sep='|')
        for _, row in df.iterrows():
            if pd.isna(row['prompt']) or str(row['prompt']).strip().lower() == 'none' or str(row['prompt']).strip() == "":
                print(f"{Colors.YELLOW}Skipping Chapter {row['chapter']}, Scene {row['scene']}: No valid prompt found.{Colors.ENDC}")
                continue
            filename_base = _create_filename_base_from_prompt(row['prompt'])
            filename_prefix = f"{str(row['chapter']).zfill(2)}-{str(row['scene']).zfill(2)}_{filename_base}"
            rows_to_process.append({'chapter': row['chapter'], 'scene': row['scene'], 'prompt': row['prompt'], 'filename_prefix': filename_prefix})
    
    listener = KeyPressListener()
    if not single_image_details: listener.start(); print(f"{Colors.YELLOW}Press 'X' at any time to gracefully stop.{Colors.ENDC}")
    
    interrupted = False
    total_images = len(rows_to_process) 
    
    try:
        for i, item in enumerate(rows_to_process):
            if listener.is_interrupt_pressed(): 
                print("\nUser interruption detected. Halting generation.")
                interrupted = True
                break
            output_path = os.path.join(images_folder, f"{item['filename_prefix']}.png")
            progress = f"[{i+1}/{total_images}]"
            
            if not single_image_details and os.path.exists(output_path):
                print(f"{progress} Skipping {os.path.basename(output_path)}, already exists."); continue
            
            print(f"\n{progress} Generating image: {os.path.basename(output_path)}")
            
            # --- START OF MODIFIED SECTION ---
            prompt_workflow = copy.deepcopy(base_workflow)
            
            # Generate a fresh 64-bit random seed for this image
            current_seed = random.randint(1, 4294967294)
            seed_applied = False

            for node in prompt_workflow.values():
                # 1. SMART SEED REPLACEMENT
                if "inputs" in node:
                    for input_key in ["seed", "noise_seed"]:
                        if input_key in node["inputs"]:
                            val = node["inputs"][input_key]
                            # Check if it's an integer and exactly 0
                            if isinstance(val, int) and val == 0:
                                node["inputs"][input_key] = current_seed
                                seed_applied = True

                # 2. Save Image Prefix
                if node["class_type"] == "SaveImage":
                    node["inputs"]["filename_prefix"] = item['filename_prefix']
                
                # 3. Prompt Replacement
                for key, value in node["inputs"].items():
                    if str(value) == "<prompt>": node["inputs"][key] = common.get("prompt_prefix", "") + item['prompt']
                    if str(value) == "<negprompt>": node["inputs"][key] = common.get('negative_prompt', '')
            
            # DEBUG LOGGING
            #if seed_applied:
            #    print(f"  -> Seed: {current_seed}")
            #else:
            #    print(f"  -> WARNING: No 'seed: 0' found in workflow. Using fixed seed from JSON.")

            image_data = _queue_comfy_prompt(prompt_workflow, comfy_settings.get("api_address"))
            if image_data:
                with open(output_path, 'wb') as f: f.write(image_data)
                print(f"  -> Successfully saved {os.path.basename(output_path)}")
            else:
                print(f"{Colors.RED}  -> Failed to retrieve image from ComfyUI. Halting generation.{Colors.ENDC}"); break
    finally:
        listener.stop()

    return interrupted

def run_comfyui_upscaling(project_path, config):
    # This function does not need to change as it is not part of the style tester loop
    comfy_settings = config.get("comfyui_settings", {})
    upscale_settings = comfy_settings.get("upscaling", {})
    api_address = comfy_settings.get("api_address")
    comfyui_path = comfy_settings.get("comfyui_path")
    if not comfyui_path or not os.path.isdir(comfyui_path):
        print(f"{Colors.RED}ERROR: 'comfyui_path' is not set or is invalid in your config.json.{Colors.ENDC}"); return
    source_folder = os.path.join(project_path, "images")
    target_folder = os.path.join(project_path, "images_upscaled")
    os.makedirs(target_folder, exist_ok=True)
    try:
        with open(upscale_settings['workflow_file'], 'r') as f: base_workflow = json.load(f)
    except Exception as e:
        print(f"ERROR: Could not load ComfyUI upscale workflow file. {e}"); return
    images_to_process = [f for f in sorted(os.listdir(source_folder)) if f.lower().endswith(('.png', '.jpg', '.jpeg')) and not os.path.exists(os.path.join(target_folder, f))]
    if not images_to_process:
        print("All images have already been upscaled."); return
    print(f"Found {len(images_to_process)} image(s) to upscale. {Colors.YELLOW}Press 'X' to stop.{Colors.ENDC}")
    listener = KeyPressListener(); listener.start()
    try:
        for i, filename in enumerate(images_to_process):
            if listener.is_interrupt_pressed(): print("\nUser interruption detected."); break
            print(f"\n[{i+1}/{len(images_to_process)}] Upscaling {filename}...")
            source_path = os.path.join(source_folder, filename)
            with open(source_path, 'rb') as f: image_data = f.read()
            files = {'image': (filename, image_data, 'image/png'), 'overwrite': (None, 'true')}
            resp = requests.post(f"http://{api_address}/upload/image", files=files)
            if resp.status_code != 200:
                print(f"  -> {Colors.RED}ERROR: Failed to upload image to ComfyUI.{Colors.ENDC}"); continue
            uploaded_filename = resp.json()['name']
            workflow = copy.deepcopy(base_workflow)
            for node in workflow.values():
                if node['class_type'] == 'LoadImage': node['inputs']['image'] = uploaded_filename
                if node['class_type'] == 'UpscaleModelLoader': node['inputs']['model_name'] = upscale_settings.get('upscaler_model')
                if node['class_type'] == 'SaveImage': node['inputs']['filename_prefix'] = f"{os.path.splitext(filename)[0]}_upscaled"
            output_filename = _queue_comfy_prompt(workflow, api_address, return_filename=True)
            if output_filename:
                comfy_output_path = os.path.join(comfyui_path, "output", output_filename)
                target_path = os.path.join(target_folder, filename)
                try:
                    shutil.move(comfy_output_path, target_path)
                    print(f"  -> Successfully upscaled and moved to {os.path.basename(target_folder)}")
                except Exception as move_e:
                    print(f"  -> {Colors.RED}ERROR: Could not move file from ComfyUI output. {move_e}{Colors.ENDC}")
            else:
                print(f"  -> {Colors.RED}ERROR: Upscaling failed in ComfyUI.{Colors.ENDC}")
    finally:
        listener.stop()

def run_forge_image_generation(project_path, config, single_image_details=None):
    """
    Runs the image generation process using Forge/A1111 API.
    Returns True if interrupted by the user, otherwise False.
    """
    forge_settings = config.get("forge_settings", {})
    common = config.get("common_settings", {})
    project_name = os.path.basename(project_path)
    csv_path = os.path.join(project_path, f"{project_name}_prompts.csv")
    images_folder = os.path.join(project_path, "images")
    os.makedirs(images_folder, exist_ok=True)
    if not single_image_details and not os.path.exists(csv_path):
        print(f"ERROR: Prompts file not found at '{csv_path}'."); return False
    rows_to_process = []
    if single_image_details:
        chapter, prompt_text = single_image_details; scene_num = 1
        for f in sorted(os.listdir(images_folder)):
            if f.startswith(f"{int(chapter):02d}-"):
                try: scene_num = int(f[3:5]) + 1
                except (ValueError, IndexError): continue
        filename_base = _create_filename_base_from_prompt(prompt_text)
        filename = f"{int(chapter):02d}-{scene_num:02d}_{filename_base}.png"
        rows_to_process.append({'chapter': chapter, 'scene': scene_num, 'prompt': prompt_text, 'filename': filename})
    else:
        df = pd.read_csv(csv_path, sep='|')
        for _, row in df.iterrows():
            if pd.isna(row['prompt']) or str(row['prompt']).strip().lower() == 'none' or not str(row['prompt']).strip():
                print(f"{Colors.YELLOW}Skipping Chapter {row['chapter']}, Scene {row['scene']}: No valid prompt found.{Colors.ENDC}")
                continue
            filename_base = _create_filename_base_from_prompt(row['prompt'])
            filename = f"{str(row['chapter']).zfill(2)}-{str(row['scene']).zfill(2)}_{filename_base}.png"
            rows_to_process.append({'chapter': row['chapter'], 'scene': row['scene'], 'prompt': row['prompt'], 'filename': filename})

    listener = KeyPressListener()
    if not single_image_details: listener.start(); print(f"{Colors.YELLOW}Press 'X' at any time to gracefully stop.{Colors.ENDC}")
    
    interrupted = False
    try:
        for item in rows_to_process:
            if listener.is_interrupt_pressed(): 
                print("\nUser interruption detected.")
                interrupted = True
                break
            output_path = os.path.join(images_folder, item['filename'])
            if not single_image_details and os.path.exists(output_path):
                print(f"Skipping {item['filename']}, already exists."); continue
            print(f"\nGenerating image for Chapter {item['chapter']}, Scene {item['scene']}: {item['filename']}")
            payload = copy.deepcopy(forge_settings.get("generation_payload", {}))
            payload["prompt"] = common.get("prompt_prefix", "") + item['prompt']
            payload["negative_prompt"] = common.get("negative_prompt", "")
            try:
                response = requests.post(url=forge_settings.get("api_url"), json=payload)
                response.raise_for_status()
                r = response.json(); image_data = base64.b64decode(r['images'][0])
                with open(output_path, 'wb') as f: f.write(image_data)
                print(f"  -> Successfully saved {item['filename']}")
            except requests.RequestException as e:
                print(f"  -> ERROR sending payload to API. Is it running with --api flag?")
                if e.response: print(f"  -> Response: {e.response.text}")
                break 
    finally:
        listener.stop()
    
    return interrupted

def run_forge_upscaling(project_path, config):
    # This function does not need to change
    forge_settings = config.get("forge_settings", {})
    source_folder = os.path.join(project_path, "images")
    target_folder = os.path.join(project_path, "images_upscaled")
    os.makedirs(target_folder, exist_ok=True)
    if not os.path.exists(source_folder) or not os.listdir(source_folder):
        print(f"\n{Colors.YELLOW}No images found in 'images' folder to upscale.{Colors.ENDC}"); return
    images_to_process = [f for f in sorted(os.listdir(source_folder)) if f.lower().endswith(('.png', '.jpg', '.jpeg')) and not os.path.exists(os.path.join(target_folder, f))]
    if not images_to_process: print("All images have already been upscaled."); return
    print(f"Found {len(images_to_process)} image(s) to upscale. {Colors.YELLOW}Press 'X' to stop.{Colors.ENDC}")
    listener = KeyPressListener(); listener.start()
    try:
        for i, filename in enumerate(images_to_process):
            if listener.is_interrupt_pressed(): print("\nUser interruption detected."); break
            print(f"\n[{i+1}/{len(images_to_process)}] Upscaling {filename}...")
            source_path = os.path.join(source_folder, filename)
            target_path = os.path.join(target_folder, filename)
            _upscale_single_image_forge(source_path, target_path, forge_settings)
    finally:
        listener.stop()

def _upscale_single_image_forge(source_path, target_path, forge_settings):
    # This function does not need to change
    settings = forge_settings.get("upscaling", {})
    url = forge_settings.get("api_url", "").replace('txt2img', 'extra-single-image')
    upscaler_name_with_ext = settings.get("upscaler", "None")
    upscaler_name_for_api, _ = os.path.splitext(upscaler_name_with_ext)
    with open(source_path, 'rb') as f: encoded_image = base64.b64encode(f.read()).decode('utf-8')
    payload = {"image": encoded_image, "upscaling_resize": settings.get("scale_by", 2.0), "upscaler_1": upscaler_name_for_api}
    try:
        response = requests.post(url=url, json=payload); response.raise_for_status(); r = response.json()
        if 'image' in r:
            with open(target_path, 'wb') as f: f.write(base64.b64decode(r['image']))
            print(f"  -> Successfully saved upscaled image.")
        else:
            print(f"  -> ERROR: Upscale API response did not contain an image.")
    except requests.RequestException as e:
        print(f"  -> ERROR sending payload to Upscale API. Details: {e}")