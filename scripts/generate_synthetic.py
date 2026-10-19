from dotenv import load_dotenv
load_dotenv()
import os
import logging
import json
import time
import random
import requests
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from typing import List, Dict, Any

# Initialize logging
logging.basicConfig(
    filename='/home/mintlover/Documents/python/blender_copilot/generation.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==============================================================================
# CONFIGURATION
# ==============================================================================
CONFIG = {
    "omniroute_url": "http://localhost:20128/v1/chat/completions",
    "seed_file": "/home/mintlover/Documents/python/blender_copilot/data/doc_data.jsonl",
    "output_file": "/home/mintlover/Documents/python/blender_copilot/data/synthetic_training_data.jsonl",
    "model": "auto/coding", # Using specialized coding route for high-capacity models
    "batch_size": 10,
}

def clean_json_response(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    return text.strip()

def call_llm(prompt: str, system_prompt: str = "You are a Blender API expert. Respond with ONLY valid JSON with keys 'question' and 'answer'.") -> str:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ.get('OMNIROUTE_API_KEY', 'sk-no-key')}",
    }
    payload = {
        "model": CONFIG["model"],
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 2000,
    }
    try:
        response = requests.post(CONFIG["omniroute_url"], headers=headers, json=payload, timeout=120)
        if response.status_code == 429:
            logger.warning("Rate limited on OmniRoute. Retrying in 5s...")
            time.sleep(5)
            return call_llm(prompt, system_prompt)
        response.raise_for_status()
        result = response.json()
        return clean_json_response(result["choices"][0]["message"]["content"].strip())
    except Exception as e:
        logger.error(f"OmniRoute call failed: {e}")
        return ""
def parse_json_output(text: str) -> Dict[str, Any]:
    if not text: return {}
    cleaned = clean_json_response(text)
    data = {}
    try:
        data = json.loads(cleaned)
    except Exception:
        try:
            q_match = re.search(r'"question":\s*"((?:\".*?|[^"])*)"', text, re.DOTALL)
            a_match = re.search(r'"answer":\s*"((?:\".*?|[^"])*)"', text, re.DOTALL)
            if q_match and a_match:
                data = {"question": q_match.group(1).replace('\\"', '"').replace('\\n', '\n'), 
                        "answer": a_match.group(1).replace('\\"', '"').replace('\\n', '\n')}
        except Exception:
            pass
    
    if not data: return {}
    
    # SAFETY NET: Flatten dict-answers
    if "answer" in data and isinstance(data["answer"], dict):
        ans_dict = data["answer"]
        flattened = []
        if "Architecture Overview" in ans_dict:
            flattened.append(f"1) Architecture Overview\n{ans_dict['Architecture Overview']}")
        if "The Code" in ans_dict:
            flattened.append(f"2) The Code\n{ans_dict['The Code']}")
        for k, v in ans_dict.items():
            if k not in ["Architecture Overview", "The Code"]:
                flattened.append(f"{k}: {v}")
        data["answer"] = "\n\n".join(flattened)
        
    return data

# ==============================================================================
# STRATEGIES
# ==============================================================================
def generate_type_3(seed: Dict) -> Dict:
    """Architect: Deep dive explanation + code."""
    code = seed.get("answer", "")
    if "```python" in code: code = code.split("```python")[1].split("```")[0].strip()
    prompt = f"Given this Blender Python snippet:\n\n{code}\n\nCreate a training pair where 'question' asks for a deep dive explanation and 'answer' provides a technical breakdown. The answer MUST be raw markdown text starting with '1) Architecture Overview' and including '2) The Code'. DO NOT USE JSON IN THE ANSWER. Return as JSON: {{\"question\": \"...\", \"answer\": \"...\"}}"
    sys_p = "You are a world-class Blender API Architect. Respond with ONLY valid JSON with keys 'question' and 'answer'. The 'answer' MUST be raw markdown starting with '1) Architecture Overview' and including '2) The Code'. Provide a professional narrative deep-dive."
    return parse_json_output(call_llm(prompt, sys_p))

def generate_type_4(seeds: List[Dict]) -> Dict:
    """Architect: Combine multiple tools into one advanced tool."""
    context = "\n".join([f"- {s['question']}: {s['answer']}" for s in seeds])
    prompt = f"Combine these into one advanced tool:\n\n{context}\n\nReturn as JSON where 'answer' is raw markdown text starting with '1) Architecture Overview' and including '2) The Code'. DO NOT USE JSON IN THE ANSWER. Format: {{\"question\": \"...\", \"answer\": \"...\"}}"
    sys_p = "You are a world-class Blender API Architect. Respond with ONLY valid JSON with keys 'question' and 'answer'. The 'answer' MUST be raw markdown starting with '1) Architecture Overview' and including '2) The Code'. Provide a professional narrative deep-dive."
    return parse_json_output(call_llm(prompt, sys_p))

def generate_type_5(seed: Dict) -> Dict:
    """Architect: Buggy version and a fix (with explanation)."""
    code = seed.get("answer", "")
    if "```python" in code: code = code.split("```python")[1].split("```")[0].strip()
    prompt = f"Given this correct code:\n\n{code}\n\nCreate a buggy version and a fix. The answer MUST be raw markdown text starting with '1) Architecture Overview' and including '2) The Code'. DO NOT USE JSON IN THE ANSWER. Return as JSON: {{\"question\": \"...\", \"answer\": \"...\"}}"
    sys_p = "You are a world-class Blender API Architect. Respond with ONLY valid JSON with keys 'question' and 'answer'. The 'answer' MUST be raw markdown starting with '1) Architecture Overview' and including '2) The Code'. Provide a professional narrative deep-dive."
    return parse_json_output(call_llm(prompt, sys_p))

def generate_type_6(seed: Dict) -> Dict:
    """Pure Coder: Code Completion (FIM)."""
    code = seed.get("answer", "")
    if "```python" in code: code = code.split("```python")[1].split("```")[0].strip()
    mid = len(code) // 2
    prefix = code[:mid]
    suffix = code[mid:]
    prompt = f"Task: Code Completion. \nGiven the prefix:\n{prefix}\n\nand the suffix:\n{suffix}\n\nCreate a training pair. The 'question' should be the prefix and suffix combined (e.g. 'Prefix... <MASK> ...Suffix'). The 'answer' MUST be ONLY the missing middle code. \n\nSTRICTLY FORBIDDEN: \n- NO 'Architecture Overview'\n- NO explanations\n- NO 'The Code' section\n- NO markdown surrounding the answer\n\nSTART IMMEDIATELY WITH THE CODE. Return as JSON: {{\"question\": \"...\", \"answer\": \"...\"}}"
    sys_p = "You are a raw code generator. Respond with ONLY valid JSON with keys 'question' and 'answer'. The 'answer' MUST be raw code. NEVER use the words 'Architecture' or 'Overview'. NEVER explain yourself. JUST CODE."
    return parse_json_output(call_llm(prompt, sys_p))

def generate_type_7(seed: Dict) -> Dict:
    """Pure Coder: Buggy -> Fixed (No Fluff)."""
    code = seed.get("answer", "")
    if "```python" in code: code = code.split("```python")[1].split("```")[0].strip()
    prompt = f"Given this correct code:\n\n{code}\n\nCreate a 'Fix this bug' pair. The 'question' must be a buggy version of the code. The 'answer' MUST be the corrected version. \n\nSTRICTLY FORBIDDEN: \n- NO 'Architecture Overview'\n- NO explanations\n- NO 'The Code' section\n\nSTART IMMEDIATELY WITH THE CODE in a markdown block. Return as JSON: {{\"question\": \"...\", \"answer\": \"...\"}}"
    sys_p = "You are a raw code generator. Respond with ONLY valid JSON with keys 'question' and 'answer'. The 'answer' MUST be raw code. NEVER use the words 'Architecture' or 'Overview'. NEVER explain yourself. JUST CODE."
    return parse_json_output(call_llm(prompt, sys_p))

def generate_type_8(seed: Dict) -> Dict:
    """Pure Coder: Junior -> Senior Refactor."""
    code = seed.get("answer", "")
    if "```python" in code: code = code.split("```python")[1].split("```")[0].strip()
    prompt = f"Given this correct code:\n\n{code}\n\nCreate a 'Refactor' pair. The 'question' should be a 'Junior' version (working but messy). The 'answer' MUST be the 'Senior' version (optimized). \n\nSTRICTLY FORBIDDEN: \n- NO 'Architecture Overview'\n- NO explanations\n- NO 'The Code' section\n\nSTART IMMEDIATELY WITH THE CODE. Return as JSON: {{\"question\": \"...\", \"answer\": \"...\"}}"
    sys_p = "You are a raw code generator. Respond with ONLY valid JSON with keys 'question' and 'answer'. The 'answer' MUST be raw code. NEVER use the words 'Architecture' or 'Overview'. NEVER explain yourself. JUST CODE."
    return parse_json_output(call_llm(prompt, sys_p))
def main():
    print("--- SCRIPT STARTING (EXTERNAL FREE MODELS) ---")
    logger.info("Starting generation process using free high-quality models")
    existing_count = 0
    if os.path.exists(CONFIG['output_file']):
        with open(CONFIG['output_file'], 'r', encoding='utf-8') as f:
            existing_count = sum(1 for _ in f)
    print(f"Resuming from sample {existing_count}...")
    CAP = 1000
    if existing_count >= CAP:
        print(f"Already reached the cap of {CAP} samples. Nothing to do.")
        return
    try:
        with open(CONFIG['seed_file'], 'r', encoding='utf-8') as f:
            seeds = [json.loads(line) for line in f if line.strip()]
        print(f"Loaded {len(seeds)} seeds.")
        def run_one(i: int) -> Dict:
            seed = seeds[i]
            # Rotate through 6 strategies: 3 Architect (3,4,5) and 3 Pure Coder (6,7,8)
            gen_type = (i % 6) + 3
            if gen_type == 3: return generate_type_3(seed)
            if gen_type == 4: return generate_type_4([seed] + random.sample(seeds, 2))
            if gen_type == 5: return generate_type_5(seed)
            if gen_type == 6: return generate_type_6(seed)
            if gen_type == 7: return generate_type_7(seed)
            return generate_type_8(seed)
        indices = list(range(existing_count, min(len(seeds), CAP)))
        
        # Use moderate concurrency to avoid instant 429s
        with open(CONFIG['output_file'], 'a', encoding='utf-8') as out_f:
            with ThreadPoolExecutor(max_workers=3) as pool:
                futures = {pool.submit(run_one, i): i for i in indices}
                for fut in tqdm(as_completed(futures), total=len(futures), desc="Generating"):
                    i = futures[fut]
                    try:
                        res = fut.result()
                        if res and "question" in res and "answer" in res and not is_hallucination(res["answer"]):
                            out_f.write(json.dumps(res) + "\n")
                            out_f.flush()
                    except Exception as e:
                        logger.error(f"Seed {i} failed: {e}")
    except Exception as e:
        logger.critical(f"Main loop crashed: {e}", exc_info=True)
    print("--- SCRIPT FINISHED ---")

if __name__ == "__main__":
    main()
