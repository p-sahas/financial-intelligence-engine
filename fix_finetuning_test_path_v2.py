import json

file_path = r"c:\Development\financial-intelligence-engine\notebooks\02_finetuning_intern.ipynb"
target_string = 'data_files="../data/processed/golden_test_set.jsonl"'
replacement_string = 'data_files=r"c:/Development/financial-intelligence-engine/data/processed/golden_test_set.jsonl"'

print(f"Reading {file_path}...")
with open(file_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

count = 0
for cell in nb['cells']:
    source_lines = cell['source']
    new_source = []
    modified = False
    for line in source_lines:
        if target_string in line:
            print("Found test dataset loading line.")
            new_line = line.replace(target_string, replacement_string)
            new_source.append(new_line)
            modified = True
        else:
            new_source.append(line)
    
    if modified:
        cell['source'] = new_source
        count += 1

if count > 0:
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print(f"Updated {count} cell(s).")
else:
    print("Target string not found.")
