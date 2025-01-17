# This file .

import pandas as pd
import os
from .utils import evaluate_expert_comment

parent_folder = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "../../../"))

data_path = os.path.join(parent_folder, "data.csv")

df = pd.read_csv(data_path)
df = df.dropna(subset=['comment'])

# Evaluate expert comments on candidates by scoring them using an AI model and save the results to a modified CSV file
for index, row in df.iterrows():
    expert_score = evaluate_expert_comment(row['comment'])
    df.at[index, 'expert_score'] = expert_score

data_path = os.path.join(parent_folder, "modified_data.csv")
df.to_csv(data_path, index=False)
