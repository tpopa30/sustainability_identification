import json
import pandas as pd
import os

print("Performance of definitions")

compare_dataset =  pd.read_csv("values/ground_truth.csv")
dir_omni = "results/gpt4_omni"
dir_deepseek_r1 = "results/test"
dir_deepseek_v3 = "results/moe"
dir_o3_mini = "results/o3_mini"
dir_gemini = "results/gemini_2" 

dir_paths = [dir_omni, dir_deepseek_r1, dir_deepseek_v3, dir_o3_mini, dir_gemini]
results = []

for location in dir_paths:
  for filename in os.listdir(location):

    file_path = os.path.join(location, filename) 

    if filename.endswith(".json"):

      false_pos = 0
      false_neg = 0
      true_pos = 0
      true_neg = 0  

      with open(file_path, 'r') as file:
        response_json = json.load(file)
        
      for elem in response_json:
        elem_id = int(elem["id"])
        
        if "label" in elem:
          elem_value = elem["label"].strip().lower()
        elif "labels" in elem and isinstance(elem["labels"], str):
          label_list = [lbl.strip().lower() for lbl in elem["labels"].split(",")]
          elem_value = "yes" if "sustainability" in label_list else "no"
                
        relevancy = str(compare_dataset[compare_dataset["Id"] == elem_id]["relevant"].values[0]).strip().lower()

        if (elem_value != "yes" and elem_value != "no"):
          if elem_value == "sustainability":
            elem_value = "yes"
          else:
            elem_value = "no"

        if elem_value == relevancy:
            if relevancy == "yes":
                true_pos += 1
            else:
                true_neg += 1
        else:
            if relevancy == "yes":
                false_neg += 1
            else:
                false_pos += 1

      precision = round((true_pos / (true_pos + false_pos)) if (true_pos + false_pos) > 0 else 0, 4)
      recall = round((true_pos / (true_pos + false_neg)) if (true_pos + false_neg) > 0 else 0, 4)
      fscore = round((2 * ((precision * recall) / (precision + recall))) if (precision + recall) > 0 else 0, 4)
      accuracy = round((true_pos + true_neg) / (true_pos + true_neg + false_neg + false_pos), 4)

      results.append([filename, accuracy, precision, recall, fscore])

df_results = pd.DataFrame(results, columns=["Model", "Accuracy", "Precision", "Recall", "F1-Score"])

df_sorted = df_results.sort_values(by="F1-Score", ascending=False)
df_sorted.to_csv("results.csv", index=False)

df_temp_1 = df_sorted[df_sorted['Model'].str.contains(r'temp_1|temp_m')]
df_temp_1.to_csv("results_temp_1.0.csv", index=False)

df_temp_0 = df_sorted[df_sorted['Model'].str.contains(r'temp_0|temp_l|temp_h')]
df_temp_0.to_csv("results_temp_0.0.csv", index=False)

