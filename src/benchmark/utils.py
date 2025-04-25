import json

import requests


def extract_message(response, key_path):
    try:
        result_json = response.json()
        for key in key_path:
            result_json = result_json[key]
        return result_json
    except (ValueError, KeyError, IndexError) as e:
        print(f"Error extracting message: {e}")
        return None


def extract_score(response):
    try:
        content = extract_message(response, ["choices", 0, "message", "content"])
        score_json = json.loads(content)
        score = score_json.get("score")
        return score
    except (ValueError, IndexError, KeyError) as e:
        print(f"Error extracting score: {e}")
        return None


def send_request_to_ai(messages):
    response = requests.post(
        "http://localhost:5001/v1/chat/completions", json={"messages": messages}
    )
    return extract_message(response, ["choices", 0, "message", "content"])


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
        {"role": "user", "content": f"<CV> {cv_content} </CV>"},
    ]
    return send_request_to_ai(messages)


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
            "content": f"<job_description> {job_description_content} </job_description>",
        },
    ]
    return send_request_to_ai(messages)


def evaluate_expert_comment(comment):
    messages = [
        {
            "role": "system",
            "content": """You are an advanced AI model designed to evaluate a candidate based on the following expert comment. Please provide a score from 1 to 100 based on the candidate's qualifications as described in the comment. For example, a valid response could be in the following JSON format: {"score": 85}.
            """,
        },
        {"role": "user", "content": f"<comment> {comment} </comment>"},
    ]
    return extract_score(send_request_to_ai(messages))
