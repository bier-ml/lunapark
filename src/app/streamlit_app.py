"""
Streamlit web interface for candidate-position matching predictions.

This application provides a user-friendly interface for matching job candidates
with vacancy descriptions using various prediction algorithms. It allows users
to input vacancy and candidate descriptions and get a matching score along
with detailed analysis.
"""
from typing import Optional, Tuple
import streamlit as st
import requests
from http import HTTPStatus

# Configuration
API_URL = "http://localhost:8000"

def set_page_config() -> None:
    """Configure the Streamlit page settings."""
    st.set_page_config(
        page_title="CV-Vacancy Matcher",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="expanded",
    )

def display_header() -> None:
    """Display the application header with title and description."""
    st.title("🎯 CV-Vacancy Matcher")
    st.markdown("""
        Welcome to the CV-Vacancy Matcher! This tool helps you evaluate how well
        a candidate matches a job position by analyzing their descriptions.
        
        Simply input the vacancy description and candidate information below to get started.
    """)

def get_predictor_selection() -> str:
    """Get the selected prediction algorithm from the user."""
    return st.selectbox(
        "Select matching algorithm 🔍",
        options=["dummy", "semantic", "skills"],  # Add more options as implemented
        index=0,
        help="Choose which algorithm to use for matching:\n"
             "- Dummy: Simple text matching\n"
             "- Semantic: Advanced semantic analysis\n"
             "- Skills: Skills-based matching"
    )

def get_match_score(
    vacancy_text: str,
    candidate_text: str,
    predictor_type: str
) -> Tuple[float, Optional[str]]:
    """
    Get match score from the API.
    
    Args:
        vacancy_text: The job vacancy description
        candidate_text: The candidate's description/CV
        predictor_type: The type of prediction algorithm to use
        
    Returns:
        Tuple containing the match score and optional analysis description
    """
    try:
        response = requests.post(
            f"{API_URL}/match",
            json={
                "vacancy_description": vacancy_text,
                "candidate_description": candidate_text,
                "predictor_type": predictor_type
            },
            timeout=10  # Add timeout to prevent hanging
        )
        
        if response.status_code == HTTPStatus.OK:
            data = response.json()
            return data["score"], data.get("description")
        
        st.error(f"API Error: {response.status_code} - {response.text}")
        return 0.0, None
        
    except requests.exceptions.RequestException as e:
        st.error(f"Connection Error: {str(e)}")
        return 0.0, None

def display_results(score: float, description: Optional[str]) -> None:
    """Display the matching results with appropriate styling."""
    st.header("📊 Results")
    
    # Display score with color coding
    score_percentage = f"{score:.0%}"
    if score >= 0.7:
        st.success(f"Match Score: {score_percentage}")
    elif score >= 0.4:
        st.warning(f"Match Score: {score_percentage}")
    else:
        st.error(f"Match Score: {score_percentage}")
    
    # Display the score gauge
    st.progress(score)
    
    # Display analysis if available
    if description:
        st.subheader("🔍 Analysis")
        st.write(description)

def input_form() -> None:
    """Handle the input form and matching logic."""
    predictor_type = get_predictor_selection()
    
    with st.form("matching_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            vacancy_text = st.text_area(
                "Vacancy Description 📝",
                height=300,
                help="Paste the job description here",
                placeholder="Enter the job vacancy description..."
            )
            
        with col2:
            candidate_text = st.text_area(
                "Candidate Description 👤",
                height=300,
                help="Paste the candidate's CV or description here",
                placeholder="Enter the candidate's information..."
            )
        
        submitted = st.form_submit_button("Calculate Match 🚀")
        
        if submitted:
            if not vacancy_text.strip() or not candidate_text.strip():
                st.error("⚠️ Please fill in both the vacancy and candidate descriptions!")
                return
                
            with st.spinner("Calculating match..."):
                score, description = get_match_score(
                    vacancy_text,
                    candidate_text,
                    predictor_type
                )
                display_results(score, description)

def main():
    """Main application entry point."""
    set_page_config()
    display_header()
    input_form()
    
    # Add footer with additional information
    st.markdown("---")
    st.markdown("""
        💡 **Tip**: For better results, provide detailed descriptions for both
        the vacancy and the candidate.
        
        ℹ️ This tool uses AI-powered matching algorithms to analyze the compatibility
        between job requirements and candidate qualifications.
    """)

if __name__ == "__main__":
    main()
