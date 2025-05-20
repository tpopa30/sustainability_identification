import json
import re
import os

# input_path = "results/overnight/"
input_path = "results/gemini" 
output_file = "results/test/"

for filename in os.listdir(input_path):
    file_path = os.path.join(input_path, filename) 

    results = []
    cleaned_data = []

    if filename.endswith(".json"):
        with open(file_path, 'r') as file:
            data = json.load(file)

    for entry in data:
        if "raw_response" in entry:
            raw_response = entry["raw_response"].strip()

            cleaned_response = re.sub(r"```json\s*|\s*```", "", raw_response).strip()

            try:
                parsed_json = json.loads(cleaned_response)
                cleaned_data.append(parsed_json)
            except json.JSONDecodeError:
                print(f"Warning: Could not parse JSON for entry {entry}. Keeping raw text.")
                cleaned_data.append({"raw_response": raw_response})
        else:
            cleaned_data.append(entry)

    with open(output_file + filename, "w") as f:
        json.dump(cleaned_data, f, indent=4)