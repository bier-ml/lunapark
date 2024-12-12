import pandas as pd
import requests
import json
import os

# Load the CSV file into a DataFrame
parent_folder = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "../../../"))

# Construct the path to the data.csv file
data_path = os.path.join(parent_folder, "data.csv")

df = pd.read_csv(data_path)[['cv', 'job_description']]
df.dropna()

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


# Iterate over the DataFrame row by row
for index, row in df.iterrows():

    # Call the summarize_cv function
    summary_response = summarize_cv(row['cv'])
    print(f"Summary Response: {summary_response}")

    # Call the summarize_job_description function
    job_summary_response = summarize_job_description(row['job_description'])
    print(f"Job Summary Response: {job_summary_response}")

    # Prepare the data for the model
    messages = [
        {
            "role": "system",
            "content": """You are an advanced AI model designed to analyze the compatibility between a CV and a job description. Your task is to provide only the compatibility score in JSON format as follows:
{
  "score": 85
}
""",
        },
        {
            "role": "user",
            "content": f"<CV> {summary_response} </CV>\n<job_description> {job_summary_response} </job_description>"
        },
    ]

    # Send the data to the model
    response = requests.post('http://localhost:5001/v1/chat/completions',
                             json={"messages": messages})

    # Extract the score from the final response
    score = extract_score(response)
    print(f"Extracted Score: {score}")

    # Save the score back to the DataFrame
    df.at[index, 'result'] = score

# Save the modified DataFrame
data_path = os.path.join(parent_folder, "modified_data.csv")
df.to_csv(data_path, index=False)
