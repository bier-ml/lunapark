"""
Streamlit web interface for candidate-position matching predictions.
"""
from typing import Optional
import streamlit as st
import requests

# Configuration
API_URL = "http://localhost:8000"  # Added API URL configuration

def get_match_score(vacancy_text: str, candidate_text: str, predictor_type: str) -> tuple[float, Optional[str]]:
    """Get match score from the API."""
    response = requests.post(
        f"{API_URL}/match",
        json={
            "vacancy_description": vacancy_text,
            "candidate_description": candidate_text,
            "predictor_type": predictor_type
        },
    )
    if response.status_code == 200:
        data = response.json()
        return data["score"], data.get("description")
    else:
        st.error(f"Error: {response.status_code}")
        return 0.0, None

def main():
    st.title("CV-Vacancy Matcher")
    
    # Add predictor selection
    predictor_type = st.selectbox(
        "Select matching algorithm",
        options=["dummy"],  # Add more options as they are implemented
        index=0,
        help="Choose which algorithm to use for matching"
    )
    
    # Existing form inputs
    with st.form("matching_form"):
        vacancy_text = st.text_area("Vacancy Description", height=200)
        candidate_text = st.text_area("Candidate Description", height=200)
        
        submitted = st.form_submit_button("Calculate Match")
        
        if submitted and vacancy_text and candidate_text:
            score, description = get_match_score(vacancy_text, candidate_text, predictor_type)
            
            # Display results
            st.header("Results")
            st.metric("Match Score", f"{score:.0%}")
            
            if description:
                st.write("Analysis:")
                st.write(description)

if __name__ == "__main__":
    main()
