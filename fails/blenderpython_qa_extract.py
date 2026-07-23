#!/usr/bin/env python
import json
import os
import re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

from bs4 import BeautifulSoup

DOC_DIR = "/home/mintlover/Documents/blender fine-tune/blender_python_reference_5_2"
OUTPUT_FILE = "/home/mintlover/Documents/blender fine-tune/blender_QA_pairs.jsonl"
MAX_PARALLEL_WORKERS = min(16, os.cpu_count() or 4)

def clean_signature(text):
    text = ' '.join(text.split())
    text = re.sub(r'\s*\(\s*', '(', text)
    text = re.sub(r'\s*\)\s*', ')', text)
    text = re.sub(r'\s*,\s*', ', ', text)
    text = re.sub(r'\.\s+', '.', text)
    return text

def parse_html_for_qa(file_path):
    """Phase 2 file agent: parse one Sphinx-generated Blender API doc page."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except Exception:
        return []

    soup = BeautifulSoup(content, 'html.parser')
    qas = []
    for dl in soup.find_all('dl'):
        if 'py' not in dl.get('class', []):
            continue

        dt = dl.find('dt')
        dd = dl.find('dd')
        if not dt or not dd:
            continue

        full_name = dt.get('id')
        if not full_name:
            continue

        for a in dt.find_all('a', class_='headerlink'):
            a.decompose()
        signature = clean_signature(dt.get_text())

        # Direct-child <p> tags only, so we don't pull the nested
        # Parameters/Returns field-list into the description.
        desc_parts = [p.get_text().strip() for p in dd.find_all('p', recursive=False)]
        explanation = ' '.join(p for p in desc_parts if p)
        if not explanation:
            continue

        qas.append({
            "question": f"What does {full_name} do?",
            "answer": explanation + f"\n\n```python\n{signature}\n```",
            "name": full_name,
        })

    return qas

def process_file(file_path):
    results = parse_html_for_qa(file_path)
    return file_path, results

def main():
    print("=== Blender API Docs Q&A Extraction Pipeline ===")
    print(f"Source: {DOC_DIR}")

    # Phase 1: orchestrator - scan and filter files
    print("\nPhase 1: Scanning files...")
    html_files = []
    for root, _dirs, files in os.walk(DOC_DIR):
        for name in files:
            if name.endswith('.html'):
                html_files.append(os.path.join(root, name))
    print(f"Found {len(html_files)} HTML files")

    skip_keywords = ['overview', 'index', 'nav', 'search', 'copyright']
    files_to_keep = []
    for fpath in html_files:
        fname = os.path.basename(fpath).lower()
        if any(kw in fname for kw in skip_keywords):
            continue
        try:
            if os.path.getsize(fpath) < 200:
                continue
        except OSError:
            continue
        files_to_keep.append(fpath)
    print(f"Kept {len(files_to_keep)} files for processing")

    print(f"\nPhase 2: Extracting Q&A pairs ({MAX_PARALLEL_WORKERS} workers)...")
    qas_all = []
    with ProcessPoolExecutor(max_workers=MAX_PARALLEL_WORKERS) as executor:
        futures = {executor.submit(process_file, f): f for f in files_to_keep}
        done = 0
        for future in as_completed(futures):
            fpath = futures[future]
            done += 1
            try:
                _, results = future.result()
                if results:
                    qas_all.extend(results)
            except Exception as e:
                print(f"  Error processing {fpath}: {e}")
            if done % 200 == 0:
                print(f"  ...{done}/{len(files_to_keep)} files processed, {len(qas_all)} Q&A so far")

    print(f"\nTotal Q&A pairs generated: {len(qas_all)}")
    if not qas_all:
        print("No Q&A pairs generated. Exiting.")
        return

    print("\nPhase 3: Deduplicating...")
    grouped = defaultdict(list)
    for qa in qas_all:
        grouped[qa['name']].append(qa)
    unique_qas = [group[0] for group in grouped.values()]
    print(f"Deduplication complete: {len(unique_qas)} unique Q&A pairs")

    print(f"\nWriting {len(unique_qas)} Q&A pairs to: {OUTPUT_FILE}")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for qa in unique_qas:
            record = {"question": qa["question"], "answer": qa["answer"]}
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    print("Done!")

    print("\nSample Q&A (first 3):")
    for qa in unique_qas[:3]:
        print(f"\nQ: {qa['question']}")
        print(f"A: {qa['answer'][:150]}...")

if __name__ == '__main__':
    main()