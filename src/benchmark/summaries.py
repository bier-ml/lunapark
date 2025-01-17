import pandas as pd
import os
from .utils import summarize_cv, summarize_job_description

parent_folder = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "../../../"))

data_path = os.path.join(parent_folder, "data.csv")

df = pd.read_csv(data_path)

df = df.dropna(subset=['cv', 'job_description'])

df['cv_summary'] = ""
df['job_summary'] = ""

# Process CVs and job descriptions, summarizing their content using an AI model and save the results to a CSV file
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
