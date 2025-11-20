# src/llm_handler.py
import os
import requests
import re
import pandas as pd
import concurrent.futures
from tqdm import tqdm
from src.config_manager import load_global_config
from src.utils import Colors

REFUSAL_KEYWORDS = [
    "i cannot", "i can't", "i apologize", "i'm sorry", "im sorry", 
    "unable to generate", "policy", "forbidden", "restricted", 
    "cannot fulfill", "as an ai", "language model"
]

def _smart_chunk_text(text, max_chunk_words):
    """
    Splits a large text into smaller chunks based on a maximum word count.
    It prioritizes grouping by paragraphs, but falls back to splitting by
    sentences if a paragraph is too large to fit in the current chunk.
    """
    if not text.strip() or max_chunk_words <= 0:
        return []

    # Use a regular expression to split text into sentences while keeping punctuation.
    # This finds any sequence of non-punctuation characters followed by punctuation.
    def split_into_sentences(paragraph_text):
        if not paragraph_text:
            return []
        return re.findall(r'[^.!?]+[.!?]?', paragraph_text)

    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    chunks = []
    current_chunk_words = []

    for paragraph in paragraphs:
        paragraph_words = paragraph.split()
        if not paragraph_words:
            continue

        # --- NEW LOGIC: Check if this paragraph can be added without breaking it up ---
        if len(current_chunk_words) + len(paragraph_words) <= max_chunk_words:
            # If it fits, add the whole paragraph and continue
            current_chunk_words.extend(paragraph_words)
        else:
            # If the paragraph itself is too big, or adding it would overflow the chunk,
            # we need to process it sentence by sentence.

            # First, finalize the current chunk if it has anything in it.
            if current_chunk_words:
                chunks.append(" ".join(current_chunk_words))
                current_chunk_words = []

            # Now, process the current paragraph sentence by sentence.
            sentences = split_into_sentences(paragraph)
            for sentence in sentences:
                sentence_words = sentence.strip().split()
                if not sentence_words:
                    continue

                # If adding the next sentence would overflow, finalize the current chunk.
                if len(current_chunk_words) + len(sentence_words) > max_chunk_words and current_chunk_words:
                    chunks.append(" ".join(current_chunk_words))
                    current_chunk_words = []
                
                # Add the sentence words to the current chunk.
                current_chunk_words.extend(sentence_words)

    # Add the last remaining chunk
    if current_chunk_words:
        chunks.append(" ".join(current_chunk_words))

    return chunks

def _get_llm_response(prompt, llm_config):
    """Sends a prompt to the configured LLM API and returns the response."""
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {llm_config.get('api_key', 'not-needed')}"}
    data = {
        "model": llm_config.get("model_name", "local-model"),
        "messages": [{"role": "user", "content": prompt}],
        "temperature": llm_config.get("temperature", 0.7)
    }
    try:
        response = requests.post(llm_config['api_url'], headers=headers, json=data, timeout=60)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except requests.exceptions.RequestException as e: return {"error": f"API Request Failed: {e}"}
    except (KeyError, IndexError): return {"error": "Invalid API Response Format"}

def _clean_response(text, parsing_config):
    """Cleans the raw LLM response based on parsing rules."""
    if not isinstance(text, str): return ""
    text = text.strip()
    if parsing_config.get("strip_code_fences", False):
        text = re.sub(r'^```[\w]*\n', '', text)
        text = re.sub(r'\n```$', '', text)
    text = text.strip('"')
    for prefix in parsing_config.get("ignore_lines_starting_with", []):
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix):].lstrip(' :')
    return text

def _get_prompt_template(llm_config):
    """Loads the prompt template from the specified file."""
    template_file = llm_config.get("prompt_template_file", "prompt_template.txt")
    if not os.path.exists(template_file):
        raise FileNotFoundError(f"Prompt template file not found: {template_file}")
    with open(template_file, 'r', encoding='utf-8') as f:
        return f.read()

def _process_chunk(args):
    """
    Processes a single chunk, parsing for a quote and a prompt with robust fallbacks.
    Returns a dictionary containing chapter, scene, prompt, and quote.
    """
    chunk_data, llm_config, prompt_template = args
    final_prompt = prompt_template.replace("<text>", chunk_data['chunk'])
    raw_response = _get_llm_response(final_prompt, llm_config)
    if isinstance(raw_response, dict) and 'error' in raw_response:
        print(f"\nERROR for Chapter {chunk_data['chapter_num']}, Scene {chunk_data['scene_num']}: {raw_response['error']}")
        return None

    # Initialize variables for parsed data
    parsed_prompt = ""
    parsed_quote = ""

    # Parse the raw response line by line
    for line in raw_response.strip().splitlines():
        line = line.strip()
        if line.lower().startswith("prompt:"):
            parsed_prompt = line[len("prompt:"):].strip()
        elif line.lower().startswith("quote:"):
            parsed_quote = line[len("quote:"):].strip()

    # --- NEW: Clean surrounding quotes from the parsed quote ---
    if parsed_quote:
        parsed_quote = parsed_quote.strip('"')

    # --- Fallback Logic ---
    if not parsed_prompt and not parsed_quote:
        # Case 1: Neither was found. Clean the whole response and use it as the prompt.
        print(f"\n{Colors.YELLOW}WARNING: Could not parse PROMPT/QUOTE for C:{chunk_data['chapter_num']}-S:{chunk_data['scene_num']}. Using full response as prompt.{Colors.ENDC}")
        parsed_prompt = _clean_response(raw_response, llm_config.get('parsing_config', {}))
        parsed_quote = " "  # Use a space as a placeholder to keep CSV structure
    elif not parsed_prompt:
        # Case 2: Quote exists, but prompt is missing. Use the quote as the prompt.
        print(f"\n{Colors.YELLOW}WARNING: PROMPT missing for C:{chunk_data['chapter_num']}-S:{chunk_data['scene_num']}. Using QUOTE as prompt.{Colors.ENDC}")
        parsed_prompt = parsed_quote
    elif not parsed_quote:
        # Case 3: Prompt exists, but quote is missing. Use a placeholder for the quote.
        parsed_quote = " "

    return {
        'chapter': chunk_data['chapter_num'],
        'scene': chunk_data['scene_num'],
        'prompt': parsed_prompt,
        'quote': parsed_quote
    }

def run_chunking_test_suite(project_path, num_chunks_to_test):
    """Runs a live, color-coded test on the first N chunks of a selected book."""
    clear_screen()
    print("--- Chunking & Prompt Test Suite (Live Data) ---")
    project_name = os.path.basename(project_path)
    clean_txt_path = os.path.join(project_path, f"{project_name}_clean.txt")
    try:
        global_config = load_global_config()
        llm_config = global_config.get("llm_settings", {})
        prompt_template = _get_prompt_template(llm_config)
        with open(clean_txt_path, 'r', encoding='utf-8') as f: full_text = f.read()
    except Exception as e:
        print(f"ERROR: Could not load required files. {e}"); return
    cleaned_full_text = full_text.replace("==CHAPTER==", " ").strip()
    chunk_size = llm_config.get("chunk_size_words", 350)
    all_chunks = _smart_chunk_text(cleaned_full_text, chunk_size)
    if not all_chunks:
        print("Could not find any text chunks in the selected book."); return
    chunks_to_test = all_chunks[:num_chunks_to_test]
    print(f"Reloaded config & prompt template. Testing the first {len(chunks_to_test)} of {len(all_chunks)} total chunks.\n")
    for i, chunk in enumerate(chunks_to_test):
        print(f"--- Chunk {i+1}/{len(chunks_to_test)} ---")
        print(f"{Colors.BOLD}CHUNK TEXT:{Colors.ENDC}\n{Colors.CYAN}\"{chunk[:300]}...\"{Colors.ENDC}\n")
        final_prompt = prompt_template.replace("<text>", chunk)
        template_parts = prompt_template.split('<text>')
        print(f"{Colors.BOLD}FULL PROMPT SENT TO LLM:{Colors.ENDC}")
        if len(template_parts) == 2:
            before, after = template_parts
            print(f"{Colors.YELLOW}\"{before}{Colors.CYAN}{chunk}{Colors.YELLOW}{after}\"{Colors.ENDC}\n")
        else:
            print(f"{Colors.YELLOW}\"{final_prompt}\"{Colors.ENDC}\n")
        print(f"{Colors.BOLD}LLM RESPONSE:{Colors.ENDC}")
        raw_response = _get_llm_response(final_prompt, llm_config)
        if isinstance(raw_response, dict) and 'error' in raw_response:
            print(f"  -> ERROR: {raw_response['error']}")
        else:
            print(f"  -> {Colors.GREEN}\"{raw_response}\"{Colors.ENDC}")
        print("\n" + "="*80 + "\n")

def run_single_text_test_suite():
    """Runs a single test using the llm_test_input.txt file and shows parsed output."""
    clear_screen(); print("--- Single Text Test Suite ---")
    try:
        print("1. Loading configurations..."); global_config = load_global_config()
        llm_config = global_config.get("llm_settings", {})
        prompt_template = _get_prompt_template(llm_config)
        with open('llm_test_input.txt', 'r', encoding='utf-8') as f: test_text = f.read()
        print("   ...done.")
    except Exception as e:
        print(f"ERROR: Could not find required file: {e}"); return
    print("\n2. Preparing the Prompt:")
    final_prompt = prompt_template.replace("<text>", test_text)
    print(f"  - Prompt Template File: {llm_config.get('prompt_template_file')}")
    print(f"  - Model to be used: {llm_config.get('model_name', 'default')}")
    print(f"  - Final Prompt Sent to LLM: \n      \"{final_prompt}\"")
    print("\n3. Sending request to LLM API...")
    raw_response = _get_llm_response(final_prompt, llm_config)
    if isinstance(raw_response, dict) and 'error' in raw_response:
        print(f"\n--- Test Failed: {raw_response['error']} ---"); return
    print("   ...response received.")
    print("\n4. Analyzing the Response:"); print(f"  - Raw LLM Response:\n\"{raw_response}\"")

    # --- NEW PARSING LOGIC FOR TEST SUITE ---
    parsed_prompt = ""; parsed_quote = ""
    for line in raw_response.strip().splitlines():
        line = line.strip()
        if line.lower().startswith("prompt:"):
            parsed_prompt = line[len("prompt:"):].strip()
        elif line.lower().startswith("quote:"):
            parsed_quote = line[len("quote:"):].strip()
    
    # Apply fallback logic for demonstration
    if not parsed_prompt and not parsed_quote:
        parsed_prompt = _clean_response(raw_response, llm_config.get('parsing_config', {}))
        parsed_quote = " " # Placeholder
    elif not parsed_prompt:
        parsed_prompt = parsed_quote
    elif not parsed_quote:
        parsed_quote = " " # Placeholder
        
    print(f"\n  - Parsed Quote: \"{parsed_quote}\"")
    print(f"  - Parsed Prompt: \"{parsed_prompt}\"")

    print("\n5. Final Result Simulation:")
    print(f"  - Example CSV Output: 01|01|{parsed_prompt}|{parsed_quote}")
    print("\n--- Test Complete ---")


def generate_prompts_for_project(project_path):
    """Generates a prompts CSV file from a book's clean text file using a warm-up prompt."""
    clear_screen(); print("--- Starting High-Speed Prompt Generation (with Smart Chunking) ---")
    project_name = os.path.basename(project_path)
    clean_txt_path = os.path.join(project_path, f"{project_name}_clean.txt")
    output_csv_path = os.path.join(project_path, f"{project_name}_prompts.csv")
    try:
        global_config = load_global_config()
        llm_config = global_config.get("llm_settings", {})
        prompt_template = _get_prompt_template(llm_config)
        with open(clean_txt_path, 'r', encoding='utf-8') as f: full_text = f.read()
    except Exception as e:
        print(f"ERROR: Could not load required files. {e}"); return
    
    chapters = [ch for ch in full_text.split("==CHAPTER==") if ch.strip()]
    chunk_size = llm_config.get("chunk_size_words", 350)
    
    tasks = []
    for i, chapter_text in enumerate(chapters):
        chapter_num = i + 1
        chunks = _smart_chunk_text(chapter_text, chunk_size)
        for j, chunk in enumerate(chunks):
            scene_num = j + 1
            tasks.append({'chunk': chunk, 'chapter_num': chapter_num, 'scene_num': scene_num})
    
    if not tasks:
        print("No text chunks found to process."); return
    
    print(f"Found {len(chapters)} chapters. Text was divided into {len(tasks)} chunks of approx. {chunk_size} words each.")
    
    all_prompts = []
    
    # --- WARM-UP LOGIC ---
    first_task = tasks.pop(0)
    print("\nSending a single 'warm-up' prompt to ensure the LLM is loaded...")
    
    first_result = _process_chunk((first_task, llm_config, prompt_template))
    
    if first_result:
        print("  -> Warm-up successful. LLM is loaded and responding.")
        all_prompts.append(first_result)
    else:
        print(f"{Colors.RED}  -> The warm-up prompt failed. Halting generation.{Colors.ENDC}")
        return

    # Process remaining tasks in parallel
    if tasks:
        num_workers = llm_config.get("concurrent_requests", 4)
        print(f"Processing the remaining {len(tasks)} chunks with {num_workers} parallel workers...")
        tasks_with_config = [(task, llm_config, prompt_template) for task in tasks]
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = list(tqdm(executor.map(_process_chunk, tasks_with_config, chunksize=1), total=len(tasks), desc="Generating Prompts"))

        successful_results = [res for res in results if res is not None]
        all_prompts.extend(successful_results)
    
    if not all_prompts:
        print("\nNo prompts were successfully generated."); return

    all_prompts.sort(key=lambda x: (x['chapter'], x['scene']))
    # Define the desired column order for the output CSV
    df = pd.DataFrame(all_prompts, columns=['chapter', 'scene', 'prompt', 'quote'])
    df.to_csv(output_csv_path, sep='|', index=False, header=True)
    print(f"\nSUCCESS: Saved {len(all_prompts)} prompts and quotes to '{output_csv_path}'")

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')