# src/style_tester.py
import os
import json
import requests
import pandas as pd
import random
import copy
import uuid
from datetime import datetime
from src.config_manager import load_global_config, load_project_config
from src.utils import Colors, get_menu_choice, open_file

HISTORY_FILE = "style_lab_history.json"
PROMPT_FILE = "style_lab_prompt.txt"

# --- PERSISTENCE HELPERS ---

def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r') as f: return json.load(f)
        except: pass
    return {}

def save_history(history):
    try:
        with open(HISTORY_FILE, 'w') as f: json.dump(history, f, indent=4)
    except: pass

def input_with_default(label, history_key, history_dict, default_val=""):
    """
    Prompts user with a default value from history.
    Returns the user input (or default) and updates the history dict.
    """
    current_val = history_dict.get(history_key, default_val)
    display_default = f" [{current_val}]" if str(current_val).strip() else ""
    
    user_input = input(f"{label}{Colors.CYAN}{display_default}{Colors.ENDC}: ").strip()
    
    if not user_input and current_val:
        return current_val
    elif user_input:
        history_dict[history_key] = user_input
        save_history(history_dict)
        return user_input
    return ""

# --- PROMPT TEMPLATE MANAGER ---

def get_style_prompt_template():
    default_template = (
        "I need to illustrate a book with this genre/vibe: '{vibe}'.\n"
        "Please generate {num_styles} distinct visual style descriptions.\n"
        "Focus on: Art Medium, Lighting, Color Palette, Texture.\n"
        "CRITICAL: Return ONLY a Python-parseable list of strings. Do not describe subject matter.\n"
        "Example Output:\n"
        "1. Oil painting, impasto, chiaroscuro lighting\n"
        "2. Anime style, cel shaded, neon palette"
    )
    
    if not os.path.exists(PROMPT_FILE):
        try:
            with open(PROMPT_FILE, 'w') as f: f.write(default_template)
        except: pass
        return default_template
    
    try:
        with open(PROMPT_FILE, 'r') as f: return f.read()
    except:
        return default_template

# --- COMFTYUI GRAPH UTILS ---

def bypass_lora_node(workflow):
    lora_ids = [nid for nid, n in workflow.items() if n['class_type'] in ['LoraLoader', 'LoraLoaderModelOnly']]
    if not lora_ids: return workflow
    # print(f"  -> {Colors.YELLOW}Bypassing {len(lora_ids)} LoRA node(s)...{Colors.ENDC}")
    for lora_id in lora_ids:
        node = workflow[lora_id]
        model_source = node['inputs'].get('model')
        clip_source = node['inputs'].get('clip')
        for other_node in workflow.values():
            if 'inputs' not in other_node: continue
            for input_name, input_val in other_node['inputs'].items():
                if isinstance(input_val, list) and len(input_val) == 2 and str(input_val[0]) == str(lora_id):
                    if input_val[1] == 0 and model_source: other_node['inputs'][input_name] = model_source
                    elif input_val[1] == 1 and clip_source: other_node['inputs'][input_name] = clip_source
        del workflow[lora_id]
    return workflow

def apply_lora_to_workflow(workflow, lora_filename, strength=1.0):
    lora_found = False
    for node in workflow.values():
        if node['class_type'] == 'LoraLoader':
            node['inputs']['lora_name'] = lora_filename
            node['inputs']['strength_model'] = strength
            node['inputs']['strength_clip'] = strength
            lora_found = True
    if not lora_found:
        print(f"  -> {Colors.YELLOW}Warning: No LoraLoader node found. LoRA ignored.{Colors.ENDC}")
    return workflow

# --- LOCAL CONNECTION HELPER ---

def queue_prompt_debug(prompt_workflow, api_address):
    client_id = str(uuid.uuid4())
    p = {"prompt": prompt_workflow, "client_id": client_id}
    url = f"http://{api_address}/prompt"
    if "http://" in api_address or "https://" in api_address: url = f"{api_address}/prompt"

    try:
        data = json.dumps(p).encode('utf-8')
        req = requests.post(url, data=data)
        if req.status_code == 400:
            print(f"\n{Colors.RED}--- ComfyUI API Error (400) ---")
            print(f"Server Message: {req.text}")
            with open("debug_failed_payload.json", "w") as f: json.dump(p, f, indent=2)
            return None
        req.raise_for_status()
        return req.json().get('prompt_id')
    except Exception as e:
        print(f"{Colors.RED}Connection Error: {e}{Colors.ENDC}")
        return None

def get_image_result(prompt_id, api_address):
    import time
    history_url = f"http://{api_address}/history/{prompt_id}"
    if "http://" in api_address: history_url = f"{api_address}/history/{prompt_id}"
    for _ in range(60):
        time.sleep(1)
        try:
            res = requests.get(history_url)
            if res.status_code == 200:
                data = res.json()
                if prompt_id in data:
                    outputs = data[prompt_id].get('outputs', {})
                    for _, output_data in outputs.items():
                        if 'images' in output_data:
                            img_info = output_data['images'][0]
                            filename = img_info['filename']
                            view_url = f"http://{api_address}/view?filename={filename}&subfolder={img_info['subfolder']}&type={img_info['type']}"
                            if "http://" in api_address: view_url = f"{api_address}/view?..."
                            return requests.get(view_url).content
        except: pass
    return None

# --- LLM HELPERS ---

def fetch_available_models(api_url):
    base_url = api_url.replace("/chat/completions", "").replace("/completions", "")
    if not base_url.endswith("/v1"): base_url = f"{base_url}/v1"
    try:
        resp = requests.get(f"{base_url}/models", timeout=5)
        if resp.status_code == 200: return [m['id'] for m in resp.json()['data']]
    except: pass
    return []

def generate_style_descriptions(vibe, num_styles, llm_config, specific_model=None):
    model = specific_model if specific_model else llm_config.get("model_name")
    
    # LOAD FROM FILE
    user_prompt_template = get_style_prompt_template()
    user_prompt = user_prompt_template.replace("{vibe}", vibe).replace("{num_styles}", str(num_styles))

    system_prompt = "You are an expert Art Director. Create distinct, high-quality visual style descriptions for Stable Diffusion."
    
    headers = {"Content-Type": "application/json", "Authorization": "Bearer not-needed"}
    data = {
        "model": model, 
        "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        "temperature": 0.9
    }
    print(f"\n{Colors.CYAN}Thinking... (Model: {model}){Colors.ENDC}")
    try:
        response = requests.post(llm_config['api_url'], headers=headers, json=data, timeout=90)
        response.raise_for_status()
        content = response.json()['choices'][0]['message']['content']
        styles = []
        for line in content.splitlines():
            line = line.strip()
            if len(line) > 3 and line[0].isdigit():
                parts = line.split('.', 1)
                if len(parts) > 1: line = parts[1].strip()
            if len(line) > 10: styles.append(line)
        return styles[:num_styles]
    except Exception as e:
        print(f"{Colors.RED}LLM Error: {e}{Colors.ENDC}")
        return []

def load_available_loras():
    loras = []
    if os.path.exists("loras.txt"):
        with open("loras.txt", "r", encoding="utf-8") as f:
            for line in f:
                if "|" in line:
                    parts = line.strip().split("|")
                    loras.append({"file": parts[0].strip(), "trigger": parts[1].strip()})
    return loras

# --- CORE LOGIC ---

def run_style_tester():
    os.system('cls' if os.name == 'nt' else 'clear')
    print(f"{Colors.BOLD}=== Style Lab: The AI Art Director ==={Colors.ENDC}")
    
    history = load_history()
    
    from src.project_manager import find_projects
    projects = find_projects()
    if not projects: print("No projects found."); return

    # 1. Project Selection (Memory Aware)
    print("\nSelect a Book Project:")
    default_proj_idx = -1
    last_proj_name = history.get("last_project_name")
    
    for i, (name, _) in enumerate(projects):
        marker = "*" if name == last_proj_name else " "
        print(f"[{i+1}] {marker} {name}")
        if name == last_proj_name: default_proj_idx = i

    try:
        p_input = input_with_default("Project #", "last_project_index", history, str(default_proj_idx + 1))
        p_idx = int(p_input) - 1
        project_name, project_path = projects[p_idx]
        history["last_project_name"] = project_name 
        save_history(history)
    except: print("Invalid selection."); return

    # Load Prompts
    csv_path = os.path.join(project_path, f"{project_name}_prompts.csv")
    if not os.path.exists(csv_path): print("Error: Prompts CSV not found."); return
    try:
        df = pd.read_csv(csv_path, sep='|')
        valid_rows = df[df['prompt'].notna() & (df['prompt'].str.len() > 10)]
        if len(valid_rows) < 1: raise ValueError("No valid prompts")
    except Exception as e: print(f"Error reading CSV: {e}"); return

    # 2. Scene Count Input (Memory Aware)
    num_samples_str = input_with_default("\nScenes to test", "last_num_samples", history, "3")
    try: num_samples = int(num_samples_str)
    except: num_samples = 3
    
    test_prompts = valid_rows.sample(n=min(num_samples, len(valid_rows)))['prompt'].tolist()

    # 3. LoRA Selection (Memory Aware)
    available_loras = load_available_loras()
    selected_lora = None
    
    if available_loras:
        print(f"\n{Colors.BOLD}Select a Base LoRA:{Colors.ENDC}")
        print("[0] None (Pure Model)")
        
        last_lora_file = history.get("last_lora_file")
        default_lora_idx = 0
        
        for i, l in enumerate(available_loras):
            marker = "*" if l['file'] == last_lora_file else " "
            print(f"[{i+1}] {marker} {l['file']} ({l['trigger']})")
            if l['file'] == last_lora_file: default_lora_idx = i + 1

        try:
            l_input = input_with_default("Selection", "last_lora_index", history, str(default_lora_idx))
            l_idx = int(l_input)
            if l_idx > 0:
                selected_lora = available_loras[l_idx-1]
                history["last_lora_file"] = selected_lora['file']
            else:
                history["last_lora_file"] = "None"
            save_history(history)
        except: pass
    
    # 4. LLM Setup (Memory Aware)
    global_config = load_global_config()
    llm_config = global_config.get("llm_settings", {})
    available_models = fetch_available_models(llm_config['api_url'])
    
    # Prefer history model, fallback to config model
    default_model = history.get("last_llm_model", llm_config.get("model_name"))
    selected_model = default_model

    if available_models:
        print(f"\n{Colors.BOLD}Available LLM Models:{Colors.ENDC}")
        default_model_idx = -1
        for i, m in enumerate(available_models):
            marker = "*" if m == default_model else " "
            print(f"[{i+1}] {marker} {m}")
            if m == default_model: default_model_idx = i

        try:
            m_input = input_with_default("Select model", "last_model_index", history, str(default_model_idx+1))
            m_idx = int(m_input) - 1
            if 0 <= m_idx < len(available_models): 
                selected_model = available_models[m_idx]
                history["last_llm_model"] = selected_model
                save_history(history)
        except: pass

    # 5. Vibe Input (Memory Aware)
    vibe = input_with_default("\nDescribe vibe/genre", "last_vibe", history, "Cinematic lighting")
    
    # 6. Num Styles Input (Memory Aware)
    num_styles_str = input_with_default("Number of styles", "last_num_styles", history, "5")
    try: num_styles = int(num_styles_str)
    except: num_styles = 5

    # Generate Styles (First Run)
    styles = generate_style_descriptions(vibe, num_styles, llm_config, selected_model)
    
    # 7. The Review Loop
    while True:
        print(f"\n{Colors.BOLD}Proposed Styles:{Colors.ENDC}")
        for i, s in enumerate(styles): print(f"[{i+1}] {s}")
        
        print(f"\n{Colors.CYAN}Options:{Colors.ENDC} (C)ontinue, (D)elete, (M)anual add, (R)egenerate, (S)witch Model, (Q)uit")
        choice = get_menu_choice()
        
        if choice == 'q': return
        elif choice == 'c': break
        
        elif choice == 'r': 
            # Regenerate with current settings
            styles = generate_style_descriptions(vibe, num_styles, llm_config, selected_model)
        
        elif choice == 's':
            # Switch Model Logic
            if available_models:
                print(f"\n{Colors.BOLD}Select New Model:{Colors.ENDC}")
                for i, m in enumerate(available_models):
                    print(f"[{i+1}] {m}")
                try:
                    idx = int(input("Model #: ")) - 1
                    if 0 <= idx < len(available_models):
                        selected_model = available_models[idx]
                        history["last_llm_model"] = selected_model
                        save_history(history)
                        print(f"Switched to {selected_model}. Regenerating...")
                        styles = generate_style_descriptions(vibe, num_styles, llm_config, selected_model)
                except: print("Invalid selection.")
            else:
                print("No models found to switch to.")

        elif choice == 'd': 
            try: styles.pop(int(input("Delete #: ")) - 1)
            except: pass
        elif choice == 'm': styles.append(input("Style: "))

    if not styles: return

    # Load Workflow
    project_config = load_project_config(project_path)
    comfy_settings = project_config.get("comfyui_settings", {})
    try:
        with open(comfy_settings['generation_workflow'], 'r') as f: base_workflow = json.load(f)
    except Exception as e: print(f"Workflow Error: {e}"); return

    # Apply LoRA
    if selected_lora:
        print(f"  -> Injecting LoRA: {selected_lora['file']}")
        base_workflow = apply_lora_to_workflow(base_workflow, selected_lora['file'])
    else:
        base_workflow = bypass_lora_node(base_workflow)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_folder = os.path.join(project_path, "Style_Experiments", timestamp)
    os.makedirs(experiment_folder, exist_ok=True)
    
    print(f"\n{Colors.GREEN}Starting Matrix Generation... Output: {experiment_folder}{Colors.ENDC}")
    report_data = {"styles": styles, "prompts": test_prompts, "matrix": {}, "lora": selected_lora}
    total_imgs = len(styles) * len(test_prompts); count = 0

    for s_idx, style_desc in enumerate(styles):
        style_id = f"Style_{s_idx+1:02d}"
        report_data["matrix"][style_id] = []
        
        for p_idx, prompt_text in enumerate(test_prompts):
            count += 1
            scene_id = f"Scene_{p_idx+1:02d}"
            filename = f"{style_id}_{scene_id}.png"
            print(f"[{count}/{total_imgs}] Generating {filename}...")
            
            lora_trigger = f", {selected_lora['trigger']}" if selected_lora else ""
            full_prompt = f"{str(prompt_text).strip()} . {str(style_desc).strip()}{lora_trigger}"
            
            prompt_workflow = copy.deepcopy(base_workflow)
            
            for node_id, node in prompt_workflow.items():
                if node.get("class_type") == "KSampler" and "seed" in node["inputs"]:
                    node["inputs"]["seed"] = int(random.randint(0, 2**32 - 1))

                if "inputs" in node:
                    for key, value in node["inputs"].items():
                        if isinstance(value, str):
                            if value == "<prompt>": 
                                node["inputs"][key] = full_prompt
                            elif value == "<negprompt>": 
                                node["inputs"][key] = project_config.get("common_settings", {}).get("negative_prompt", "")

            prompt_id = queue_prompt_debug(prompt_workflow, comfy_settings.get("api_address"))
            
            if prompt_id:
                image_data = get_image_result(prompt_id, comfy_settings.get("api_address"))
                if image_data:
                    output_path = os.path.join(experiment_folder, filename)
                    with open(output_path, 'wb') as f: f.write(image_data)
                    report_data["matrix"][style_id].append(filename)
                    print(f"   -> Saved.")
                else:
                    print(f"   -> Failed to retrieve image.")
            else:
                print(f"   -> Skipped due to API error.")

    generate_html_report(experiment_folder, report_data, vibe)
    print(f"\n{Colors.BOLD}Experiment Complete!{Colors.ENDC}")
    open_file(os.path.join(experiment_folder, "index.html"))

def generate_html_report(folder_path, data, vibe):
    # Build the LoRA info string with Trigger words
    if data['lora']:
        file_name = data['lora']['file']
        trigger_words = data['lora']['trigger']
        # formatting: LoRA Filename [Trigger: words]
        lora_info = (
            f"<span style='color:#4da6ff'>LoRA: {file_name}</span> "
            f"<span style='color:#aaa; font-size:0.8em'>[Trigger: {trigger_words}]</span>"
        )
    else:
        lora_info = "<span style='color:#666'>(No LoRA)</span>"

    html = f"""<!DOCTYPE html><html><head><title>Style Lab - {vibe}</title>
    <style>body{{font-family:sans-serif;background:#1a1a1a;color:#fff;padding:20px;}}
    table{{width:100%;border-collapse:collapse;}}th,td{{border:1px solid #333;padding:10px;vertical-align:top;}}
    th{{background:#252525;text-align:left;}}img{{max-width:300px;height:auto;transition:transform 0.2s;}}
    img:hover{{transform:scale(1.5);border:2px solid #fff;position:relative;z-index:10;}}
    .prompt-text{{font-size:0.8em;color:#888;height:40px;overflow:hidden;}}
    </style></head><body>
    <h1>Style Lab Report</h1>
    <h3>{vibe}<br>{lora_info}</h3>
    <table><thead><tr><th>Style</th>"""
    
    for i, p in enumerate(data['prompts']): 
        html += f"<th>Scene {i+1}<div class='prompt-text'>{p}</div></th>"
    
    html += "</tr></thead><tbody>"
    
    for i, style in enumerate(data['styles']):
        style_id = f"Style_{i+1:02d}"
        html += f"<tr><td><strong>{style_id}</strong><br><small>{style}</small></td>"
        for img in data['matrix'].get(style_id, []):
            html += f"<td style='text-align:center;'><a href='{img}' target='_blank'><img src='{img}'></a></td>"
        html += "</tr>"
    
    html += "</tbody></table></body></html>"
    
    with open(os.path.join(folder_path, "index.html"), "w", encoding='utf-8') as f: 
        f.write(html)