# This is the README for the repository of the Sustainability Identification thesis

### Structure

Inside there are 3 folders containing the output of the LLMs (`results` folder), the dataset and the prompts. The rest of the files outside the folder structure contain the results as CSV format, functions to clean the resulting JSONs (some LLMs do not support structured output - `clean_json.py`).

### Running the code

This file should contain the basics needed to run the files. The repository contains one Python script for OpenAI, DeepSeek and Google's Gemini. The scripts require an API key in order for them to be operated. The messages used are loaded from the path. 

``A requirements.txt file is provided for the needed dependencies.``
