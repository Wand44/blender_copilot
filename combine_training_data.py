import json 

files = ['training_data/blender_addons.jsonl', 
         'training_data/general_blender_info.jsonl',
         'training_data/blender_QA_pairs.jsonl'] 

training_data = []

for file in files:
    with open(file, 'r') as f:
        for line in f:
            training_data.append(json.loads(line.strip()))
            
print(f"Loaded total: {len(training_data)}")
            
with open('training_data.jsonl', 'w') as f:
    for item in training_data:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')
