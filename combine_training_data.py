import json
import os

training_data = []
training_dir = 'training_data'

files = sorted(
    os.path.join(training_dir, filename)
    for filename in os.listdir(training_dir)
    if os.path.isfile(os.path.join(training_dir, filename)) and filename.endswith('.jsonl')
)

for file_path in files:
    with open(file_path, 'r', encoding='utf-8') as f:
        print(f"Loading {file_path}...")
        for line in f:
            line = line.strip()
            if not line:
                continue
            training_data.append(json.loads(line))

print(f"Loaded total: {len(training_data)}")

with open('training_data.jsonl', 'w', encoding='utf-8') as f:
    for item in training_data:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')
