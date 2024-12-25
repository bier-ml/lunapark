# This file processes CVs and job descriptions, summarizing their content using an AI model and saving the results to a CSV file.

import pandas as pd
import os
import requests
import json

# Load the CSV file into a DataFrame
parent_folder = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "../../../"))

# Construct the path to the data.csv file
data_path = os.path.join(parent_folder, "data.csv")

# Read the entire CSV file
df = pd.read_csv(data_path)

# Drop rows with missing values in 'cv' and 'job_description' columns
df = df.dropna(subset=['cv', 'job_description'])

# Function to summarize CV


def summarize_cv(cv_content):
    messages = [
        {
            "role": "system",
            "content": """You are an advanced AI model designed to summarize a CV into the following structure:
1. Professional Summary
2. Work Experience
3. Education
4. Skills
5. Certifications and Licenses

If any category is missing, please respond with "Not specified" for that category.
""",
        },
        {
            "role": "user",
            "content": f"<CV> {cv_content} </CV>"
        },
    ]

    # Send the data to the model
    response = requests.post('http://localhost:5001/v1/chat/completions',
                             json={"messages": messages})
    return extract_message(response, ['choices', 0, 'message', 'content'])

# Function to summarize job description


def summarize_job_description(job_description_content):
    messages = [
        {
            "role": "system",
            "content": """You are an advanced AI model designed to summarize a job description into the following structure:
1. Job Summary
2. Responsibilities
3. Qualifications (Required and preferred levels of education or fields of study, Minimum years and type of work experience)
4. Required Skills
5. Preferred Skills (Optional)
6. Certifications (Optional)

If any category is missing, please respond with "Not specified" for that category.
""",
        },
        {
            "role": "user",
            "content": f"<job_description> {job_description_content} </job_description>"
        },
    ]

    # Send the data to the model
    response = requests.post('http://localhost:5001/v1/chat/completions',
                             json={"messages": messages})
    return extract_message(response, ['choices', 0, 'message', 'content'])

# Function to extract score from the model's response


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

# Function to extract a specific message from a structured JSON response


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


df['cv_summary'] = ""
df['job_summary'] = ""


# New loop to summarize each CV and job description
for index, row in df.iterrows():

    if pd.isna(row['cv_summary']) or row['cv_summary'] == "":
        summary_response = summarize_cv(row['cv'])
        df.at[index, 'cv_summary'] = summary_response

    if pd.isna(row['job_summary']) or row['job_summary'] == "":
        job_summary_response = summarize_job_description(
            row['job_description'])
        df.at[index, 'job_summary'] = job_summary_response

    summary_data_path = os.path.join(parent_folder, "summarized_data.csv")
    df.to_csv(summary_data_path, index=False)
