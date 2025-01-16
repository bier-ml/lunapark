import os
from http import HTTPStatus
from typing import List, Optional, Tuple

import requests
import streamlit as st
from tools import PDFToText  # , LinkedInProfileParser

API_URL = os.getenv("API_URL", "http://localhost:8000")


def set_page_config() -> None:
    """Configure the Streamlit page settings."""
    st.set_page_config(
        page_title="JobMatch",
        page_icon="https://hrlunapark.com/favicon-32x32.png",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def display_header() -> None:
    """Display the application header with title and description."""
    st.write(
        "<style>div.block-container{padding-top:1rem;}</style>", unsafe_allow_html=True
    )
    st.markdown(
        "# JobMatch // <span style='color: #ff6db3;'>Luna Park</span>",
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner="Loading predictors...")
def get_available_predictors() -> List[str]:
    """Fetch available predictor types from the API."""
    try:
        response = requests.get(f"{API_URL}/available-models", timeout=180)

        if response.status_code == HTTPStatus.OK:
            data = response.json()
            return data["predictor_types"]

        st.error(
            f"Failed to fetch available models: {response.status_code} - {response.text}"
        )
        return ["dummy"]  # Fallback to dummy predictor

    except requests.exceptions.RequestException as e:
        st.error(f"Connection Error while fetching models: {str(e)}")
        return ["dummy"]  # Fallback to dummy predictor


@st.cache_resource(show_spinner="Loading models...")
def get_available_models_per_predictor() -> dict:
    """Fetch available models for each predictor type from the API."""
    try:
        response = requests.get(
            f"{API_URL}/available-models-per-predictor", timeout=180
        )

        if response.status_code == HTTPStatus.OK:
            data = response.json()
            return data["models"]

        st.error(
            f"Failed to fetch available models: {response.status_code} - {response.text}"
        )
        return ["dummy-model-v1"]

    except requests.exceptions.RequestException as e:
        st.error(f"Connection Error while fetching models: {str(e)}")
        return ["dummy-model-v1"]


def get_predictor_selection() -> Tuple[str, Optional[str]]:
    """Get the selected prediction algorithm and model from the user."""
    available_predictors = get_available_predictors()
    models_per_predictor = get_available_models_per_predictor()

    col1, col2 = st.columns(2)
    with col1:
        predictor_type = st.selectbox(
            "Select matching algorithm 🔍",
            options=available_predictors,
            index=0,
            help="Choose which algorithm to use for matching:\n"
            "- Dummy: Simple text matching\n"
            "- LM: Language Model-based matching using AI\n",
        )

    with col2:
        selected_model = None
        if (
            predictor_type in models_per_predictor
            and models_per_predictor[predictor_type]
        ):
            selected_model = st.selectbox(
                "Model",
                options=models_per_predictor[predictor_type],
                index=0,
                help="Choose which specific model to use for the selected algorithm",
            )

    return predictor_type, selected_model


def get_match_score(
    vacancy_text: str,
    resume_text: str,
    hr_comment: str,
    predictor_type: str,
    model: Optional[str] = None,
) -> Tuple[float, Optional[str]]:
    """
    Get match score from the API.

    Args:
        vacancy_text: The job vacancy description
        resume_text: The candidate's description/CV
        hr_comment: HR comments about candidate
        predictor_type: The type of prediction algorithm to use
        model: The specific model to use (optional)

    Returns:
        Tuple containing the match score and optional analysis description
    """
    try:
        request_data = {
            "vacancy_description": vacancy_text,
            "candidate_description": resume_text,
            "hr_comment": hr_comment,
            "predictor_type": predictor_type,
        }

        # Add predictor parameters if a specific model is selected
        if model:
            request_data["predictor_parameters"] = {"model": model}  # type: ignore

        response = requests.post(
            f"{API_URL}/match",
            json=request_data,
            timeout=180,
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
    st.subheader("📊 Results")

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
    predictor_type, selected_model = get_predictor_selection()

    # Create two columns for input method selection
    col1, col2 = st.columns(2)
    resume_text = ""
    vacancy_text = ""
    hr_comment = ""

    with col1:
        st.subheader("Candidate")
        candidate_input_method = st.radio(
            "Select Candidate Input Method",
            label_visibility="collapsed",
            options=["Text Input", "Upload PDF"],  #
            horizontal=True,
            index=0,
        )

        if candidate_input_method == "Text Input":
            resume_text = st.text_area(
                "Candidate Description 👤",
                height=200,
                help="Example:\n"
                "Experienced software developer with 6 years in the industry, "
                "specializing in Python and cloud technologies. Proven track record "
                "of leading development teams and implementing scalable solutions. "
                "Holds a Master's degree in Computer Science and has extensive "
                "experience with AWS, Docker, and microservices architecture. "
                "Successfully led a team of 5 developers in previous role, delivering "
                "multiple high-impact projects on time and within budget.",
                placeholder=("Enter the candidate's information...\n\n"),
            )
        elif candidate_input_method == "Upload PDF":
            resume_pdf = st.file_uploader(label="Upload resume (.pdf)", type="pdf")
            if resume_pdf:
                cv_pdf_to_text = PDFToText(resume_pdf)
                resume_text = cv_pdf_to_text.extract_text()
        elif candidate_input_method == "LinkedIn URL":
            linkedin_url = st.text_input(
                "LinkedIn URL", placeholder="Enter the LinkedIn profile URL here"
            )
            if linkedin_url:
                with st.popover("Open popover"):
                    st.markdown("Enter creds 👋")
                    # linkedin_email = st.text_input("email?")
                    # linkedin_password = st.text_input("pass?")

                    # Initialize LinkedInProfileParser and fetch the resume text
                    # linkedin_email = "your_email@example.com"  # Replace with your LinkedIn email
                    # linkedin_password = "your_password"  # Replace with your LinkedIn password
                    # parser = LinkedInProfileParser(linkedin_email, linkedin_password)
                    # resume_text = parser.get_text_resume(linkedin_url)
                    # parser.close()  # Close the parser after fetching the data

        # st.write(f"resume_text: {resume_text}")
        if candidate_input_method != "Text Input" and resume_text:
            st.text_area(
                "Parsed resume", resume_text, key="cv", height=250, disabled=True
            )

    with col2:
        st.subheader("Vacancy")
        job_input_method = st.radio(
            "Select resume input method",
            label_visibility="collapsed",
            options=["Text Input", "Upload PDF"],
            horizontal=True,
            index=0,
        )

        if job_input_method == "Upload PDF":
            job_pdf = st.file_uploader(label="Upload .pdf job description", type="pdf")
            if job_pdf:
                job_pdf_to_text = PDFToText(job_pdf)
                vacancy_text = job_pdf_to_text.extract_text()
        else:
            vacancy_text = st.text_area(
                "Vacancy Description 📝",
                height=200,
                help="Example:\n"
                "We are seeking a Senior Software Engineer with at least 5 years "
                "of experience in software development. The ideal candidate should "
                "have strong expertise in Python, Docker, and AWS. You will be "
                "responsible for designing and implementing scalable solutions "
                "for our cloud infrastructure. Bachelor's degree in Computer Science "
                "or related field is required. Experience with microservices "
                "architecture and team leadership is a plus.",
                placeholder=("Enter the job vacancy description...\n\n"),
            )

        if job_input_method != "Text Input" and vacancy_text:
            st.text_area(
                "Parsed vacancy", vacancy_text, key="vacancy", height=250, disabled=True
            )

    hr_comment = st.text_area(
        "HR Comment 📝",
        # height=250,
        help="Example:\n" "Great expirience, but rather bad match ",
        placeholder=("Enter any comments...\n\n"),
    )

    submitted = st.button("Calculate Match 🚀")

    if submitted:
        if not vacancy_text.strip() or not resume_text.strip():
            st.warning("⚠️ Please fill in both the vacancy and candidate descriptions!")
            return

        with st.spinner("Calculating match..."):
            score, description = get_match_score(
                vacancy_text, resume_text, hr_comment, predictor_type, selected_model
            )
            display_results(score, description)


def main():
    """Main application entry point."""
    set_page_config()
    display_header()
    input_form()

    # Add footer with additional information
    # st.markdown("---")
    # st.markdown("""
    #     💡 **Tip**: For better results, provide detailed descriptions for both
    #     the vacancy and the candidate.

    #     This tool uses AI-powered matching algorithms to analyze the compatibility
    #     between job requirements and candidate qualifications.
    # """)


if __name__ == "__main__":
    main()
