# This file evaluates expert comments on candidates by scoring them using an AI model and saves the results to a modified CSV file.

import pandas as pd
import requests
import json
import os

# Load the CSV file into a DataFrame
parent_folder = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "../../../"))

# Construct the path to the data.csv file
data_path = os.path.join(parent_folder, "data.csv")

df = pd.read_csv(data_path)  # Load all columns
df = df.dropna(subset=['comment'])  # Drop rows where 'comment' is NaN


def extract_message(response, key_path):
    try:
        result_json = response.json()
        # Navigate through the JSON structure using the key path
        for key in key_path:
            result_json = result_json[key]
        return result_json
    except (ValueError, KeyError, IndexError) as e:
        print(f"Error extracting message: {e}")
        return None


def extract_score(response):
    try:
        content = extract_message(
            response, ['choices', 0, 'message', 'content'])
        score_json = json.loads(content)
        score = score_json.get('score')
        return score
    except (ValueError, IndexError, KeyError) as e:
        print(f"Error extracting score: {e}")
        return None


def evaluate_expert_comment(comment):
    messages = [
        {
            "role": "system",
            "content": """You are an advanced AI model designed to evaluate a candidate based on the following expert comment. Please provide a score from 1 to 100 based on the candidate's qualifications as described in the comment. For example, a valid response could be in the following JSON format: {"score": 85}.
            """,
        },
        {
            "role": "user",
            "content": f"<comment> {comment} </comment>"
        },
    ]

    # Send the data to the model
    response = requests.post('http://localhost:5001/v1/chat/completions',
                             json={"messages": messages})
    return extract_score(response)


# Iterate over the DataFrame row by row
for index, row in df.iterrows():
    # Evaluate the expert comment
    expert_score = evaluate_expert_comment(row['comment'])

    # Save the score back to the DataFrame
    df.at[index, 'expert_score'] = expert_score

# Save the modified DataFrame
data_path = os.path.join(parent_folder, "modified_data.csv")
df.to_csv(data_path, index=False)
