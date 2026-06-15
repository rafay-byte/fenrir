import os
import json

# Define the full path to the file
file_path = r"C:\PY PROJECTS\fenrir\data\qa_dataset.json"

# Check if the file exists
if os.path.exists(file_path):
    print("✅ File exists.")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            print("✅ JSON loaded successfully.")
            print(f"Found {len(data)} entries.")
    except json.JSONDecodeError as e:
        print("❌ Failed to parse JSON:", e)
else:
    print("❌ File does not exist at the path.")
