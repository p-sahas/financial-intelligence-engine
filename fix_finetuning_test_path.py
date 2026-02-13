import json
import os

file_path = r"c:\Development\financial-intelligence-engine\notebooks\02_finetuning_intern.ipynb"
abs_test_path = r"c:/Development/financial-intelligence-engine/data/processed/golden_test_set.jsonl"

print(f"Reading {file_path}...")
with open(file_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

count = 0
for cell in nb['cells']:
    source = "".join(cell['source'])
    if "golden_test_set.jsonl" in source and "load_dataset" in source:
        # Check if absolute path is already there
        if abs_test_path in source:
            print("Absolute test path already present.")
            continue
            
        print("Found test set loading cell to update...")
        # Since I don't have the exact cell content, I'll replace the path logic pattern
        # I'll look for the list definition if it exists, or just the filename usage.
        
        # Heuristic: If it uses a list of paths like the train set
        if "possible_paths = [" in source:
             # Similar structure to train set loading?
             # Let's inspect the cell content by printing it first
             pass
        else:
             # It might be a simple `load_dataset("json", data_files=...)`
             # I'll try to make it robust by adding the search logic if it's simple
             print("Complex update needed. Replacing cell with robust version if it seems standard.")
             
        # Actually, let's just make it robust.
        # I'll replace the specific string "golden_test_set.jsonl" with the absolute path in the `load_dataset` call?
        # No, that's brittle.
        
        # Let's insert the absolute path search logic.
        new_source = []
        replaced = False
        for line in cell['source']:
            if "golden_test_set.jsonl" in line and "possible_paths" not in line and not replaced:
                # If it's just a file path string, we can try to prepend logic?
                # This is risky without seeing code.
                pass
        
        # Better approach: Read the cell, see if it has `possible_paths`. 
        # If yes, add abs path. If no, assume it's hardcoded and verify.
        
        if "possible_paths = [" in source:
             # It has the list. Add the path.
             new_cell_source = []
             skip = False
             for line in cell['source']:
                 if "possible_paths = [" in line:
                     new_cell_source.append(line)
                     new_cell_source.append(f'    r"{abs_test_path}", # Absolute path\n')
                 else:
                     new_cell_source.append(line)
             cell['source'] = new_cell_source
             count += 1
             print("Added absolute path to possible_paths list.")

if count > 0:
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print(f"Updated {count} cell(s).")
else:
    print("No cells updated. Either test set not loaded via list or already fixed.")
