import os
import re
import time
from http import HTTPStatus

import requests
import streamlit as st
from tools import PDFToText  # type: ignore

API_URL = os.getenv("API_URL", "http://localhost:8000")


def main():
    """Main application function containing all the Streamlit app logic."""
    # Page Configuration
    st.set_page_config(
        page_title="JobMatch",
        page_icon="https://hrlunapark.com/favicon-32x32.png",
        layout="wide",
    )

    # Header
    st.write(
        "<style>div.block-container{padding-top:1rem;}</style>", unsafe_allow_html=True
    )
    st.markdown(
        "# JobMatch // <span style='color: #ff6db3;'>Luna Park</span>",
        unsafe_allow_html=True,
    )

    # Tabs for different functionalities
    tab1, tab2 = st.tabs(["Candidate-Job Matching", "Job Search"])

    with tab1:
        candidate_job_matching()

    with tab2:
        job_search()


def candidate_job_matching():
    """Candidate-Job matching functionality (original app)"""
    # GPU Pod Management
    st.markdown("## 🖥️ GPU Pod Management")
    max_retries = 100
    st.write(API_URL)

    try:
        # Initialize retry counter in session state if not exists
        if "status_check_count" not in st.session_state:
            st.session_state.status_check_count = 0

        # Check if pod exists and is ready
        with st.spinner("Checking GPU Pod status..."):
            # Get pods list
            response = requests.get(f"{API_URL}/pods", timeout=180)
            if response.status_code == 200:
                pods = response.json().get("pods", [])
                if pods:
                    pod = pods[0]
                    pod_id = pod["pod_id"]

                    # Check pod status
                    status_response = requests.get(
                        f"{API_URL}/pods/{pod_id}/status", timeout=180
                    )
                    if status_response.status_code == 200:
                        status_data = status_response.json()
                        if status_data.get("is_ready"):
                            # Pod is ready - show status and terminate button
                            st.session_state.status_check_count = 0  # Reset counter
                            st.success("✅ GPU Pod is ready")
                            if st.button("🛑 Terminate GPU Pod"):
                                with st.spinner("Terminating GPU Pod..."):
                                    delete_response = requests.delete(
                                        f"{API_URL}/pods/{pod_id}", timeout=180
                                    )
                                    if delete_response.status_code == 200:
                                        pass
                                        # st.rerun()
                                    else:
                                        st.error("Failed to terminate GPU Pod")
                        else:
                            # Increment counter and check if exceeded max retries
                            st.session_state.status_check_count += 1
                            if st.session_state.status_check_count >= max_retries:
                                # Terminate pod after too many retries
                                st.error("Pod startup timeout. Terminating pod...")
                                delete_response = requests.delete(
                                    f"{API_URL}/pods/{pod_id}", timeout=180
                                )
                                st.session_state.status_check_count = 0  # Reset counter
                                # st.rerun()
                            else:
                                # Pod exists but not ready - keep spinner and refresh
                                pass
                                # time.sleep(10)
                                # st.rerun()
                else:
                    # No pods - show create button
                    st.session_state.status_check_count = 0  # Reset counter
                    if st.button("🚀 Create GPU Pod"):
                        response = requests.post(f"{API_URL}/pods", timeout=180)
                        if response.status_code == 200:
                            time.sleep(10)
                            # st.rerun()
                        else:
                            st.error("Failed to create GPU Pod")
    except requests.exceptions.RequestException as e:
        st.error(f"Error managing GPU pod: {str(e)}")

    # Get available predictors and models
    try:
        predictor_response = requests.get(f"{API_URL}/available-models", timeout=180)
        models_response = requests.get(
            f"{API_URL}/available-models-per-predictor", timeout=180
        )

        available_predictors = ["dummy"]
        models_per_predictor = {"dummy": ["dummy-model-v1"]}

        if predictor_response.status_code == HTTPStatus.OK:
            available_predictors = predictor_response.json()["predictor_types"]
        if models_response.status_code == HTTPStatus.OK:
            models_per_predictor = models_response.json()["models"]
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching models: {str(e)}")
        available_predictors = ["dummy"]
        models_per_predictor = {"dummy": ["dummy-model-v1"]}

    # Predictor Selection
    col1, col2 = st.columns(2)
    with col1:
        predictor_type = st.selectbox(
            "Select matching algorithm 🔍",
            options=available_predictors,
            index=0,
            help="Choose which algorithm to use for matching",
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
                help="Choose which specific model to use",
            )

    # Input Form
    col1, col2 = st.columns(2)
    resume_text = ""
    vacancy_text = ""

    with col1:
        st.subheader("Candidate")
        candidate_input_method = st.radio(
            "Select Candidate Input Method",
            label_visibility="collapsed",
            options=["Text Input", "Upload PDF"],
            horizontal=True,
            index=0,
        )

        if candidate_input_method == "Text Input":
            resume_text = st.text_area(
                "Candidate Description 👤",
                height=200,
                help="Enter candidate description",
                placeholder="Enter the candidate's information...",
            )
        else:
            resume_pdf = st.file_uploader(label="Upload resume (.pdf)", type="pdf")
            if resume_pdf:
                cv_pdf_to_text = PDFToText(resume_pdf)
                resume_text = cv_pdf_to_text.extract_text()
                st.text_area("Parsed resume", resume_text, height=250, disabled=True)

    with col2:
        st.subheader("Vacancy")
        job_input_method = st.radio(
            "Select resume input method",
            label_visibility="collapsed",
            options=["Text Input", "Upload PDF"],
            horizontal=True,
            index=0,
        )

        if job_input_method == "Text Input":
            vacancy_text = st.text_area(
                "Vacancy Description 📝",
                height=200,
                help="Enter vacancy description",
                placeholder="Enter the job vacancy description...",
            )
        else:
            job_pdf = st.file_uploader(label="Upload .pdf job description", type="pdf")
            if job_pdf:
                job_pdf_to_text = PDFToText(job_pdf)
                vacancy_text = job_pdf_to_text.extract_text()
                st.text_area("Parsed vacancy", vacancy_text, height=250, disabled=True)

    hr_comment = st.text_area(
        "HR Comment 📝",
        help="Enter any additional comments",
        placeholder="Enter any comments...",
    )

    # Match Calculation
    if st.button("Calculate Match 🚀"):
        if not vacancy_text.strip() or not resume_text.strip():
            st.warning("⚠️ Please fill in both the vacancy and candidate descriptions!")
            return

        with st.spinner("Calculating match..."):
            try:
                request_data = {
                    "vacancy_description": vacancy_text,
                    "candidate_description": resume_text,
                    "hr_comment": hr_comment,
                    "predictor_type": predictor_type,
                }
                if selected_model:
                    request_data["predictor_parameters"] = {"model": selected_model}

                response = requests.post(
                    f"{API_URL}/match",
                    json=request_data,
                    timeout=180,
                )

                if response.status_code == HTTPStatus.OK:
                    data = response.json()
                    score = data["score"]
                    description = data.get("description")

                    # Display Results
                    st.subheader("📊 Results")
                    if score >= 4:
                        st.success(f"Match Score: {score}")
                    elif score >= 3:
                        st.warning(f"Match Score: {score}")
                    else:
                        st.error(f"Match Score: {score}")

                    st.progress(min(score / 5, 1.0))

                    if description:
                        st.subheader("🔍 Analysis")

                        # Extract any XML tags if present
                        explanation = description
                        llm_score = None
                        pros = []
                        cons = []

                        # Remove thought tag content if present
                        explanation = re.sub(
                            r"<thought>.*?</thought>", "", explanation, flags=re.DOTALL
                        )

                        # Extract score if present
                        score_match = re.search(
                            r"<score>(.*?)</score>", explanation, flags=re.DOTALL
                        )
                        if score_match:
                            try:
                                llm_score = float(score_match.group(1).strip())
                                # Remove the score tag from explanation
                                explanation = re.sub(
                                    r"<score>.*?</score>",
                                    "",
                                    explanation,
                                    flags=re.DOTALL,
                                )
                            except (ValueError, TypeError):
                                pass

                        # Extract pros if present
                        pros_match = re.search(
                            r"<pros>(.*?)</pros>", explanation, flags=re.DOTALL
                        )
                        if pros_match:
                            pros_text = pros_match.group(1).strip()
                            pros = [
                                line.strip()[2:].strip()
                                for line in pros_text.split("\n")
                                if line.strip().startswith("-")
                            ]
                            # Remove the pros tag from explanation
                            explanation = re.sub(
                                r"<pros>.*?</pros>", "", explanation, flags=re.DOTALL
                            )

                        # Extract cons if present
                        cons_match = re.search(
                            r"<cons>(.*?)</cons>", explanation, flags=re.DOTALL
                        )
                        if cons_match:
                            cons_text = cons_match.group(1).strip()
                            cons = [
                                line.strip()[2:].strip()
                                for line in cons_text.split("\n")
                                if line.strip().startswith("-")
                            ]
                            # Remove the cons tag from explanation
                            explanation = re.sub(
                                r"<cons>.*?</cons>", "", explanation, flags=re.DOTALL
                            )

                        # Extract explanation tag content if present
                        explanation_match = re.search(
                            r"<explanation>(.*?)</explanation>",
                            explanation,
                            flags=re.DOTALL,
                        )
                        if explanation_match:
                            explanation = explanation_match.group(1).strip()

                        # Display the cleaned explanation
                        explanation = explanation.strip()
                        if explanation:
                            st.write(explanation)

                        # Display LLM score if present
                        if llm_score is not None:
                            st.write(f"**LLM Score**: {llm_score}")

                        # Display pros and cons if present
                        if pros or cons:
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown("#### Pros")
                                if pros:
                                    for pro in pros:
                                        st.markdown(f"- {pro}")
                                else:
                                    st.markdown("None specified")

                            with col2:
                                st.markdown("#### Cons")
                                if cons:
                                    for con in cons:
                                        st.markdown(f"- {con}")
                                else:
                                    st.markdown("None specified")
                else:
                    st.error(f"API Error: {response.status_code} - {response.text}")

            except requests.exceptions.RequestException as e:
                st.error(f"Connection Error: {str(e)}")


def job_search():
    """Job search functionality to find matching candidates for a position"""
    st.markdown("## 🔍 Find Matching Candidates")
    st.write(
        "Enter a job description or query to find the best matching candidates in the database."
    )

    # Input area for job description or query
    query_method = st.radio(
        "Input Method",
        options=["Job Description", "Search Query"],
        horizontal=True,
        index=0,
    )

    if query_method == "Job Description":
        job_input_method = st.radio(
            "Job Description Input Method",
            options=["Text Input", "Upload PDF"],
            horizontal=True,
            index=0,
        )

        job_query = ""
        if job_input_method == "Text Input":
            job_query = st.text_area(
                "Job Description 📝",
                height=200,
                help="Enter the full job description",
                placeholder="Enter the job description with requirements, skills, location, etc...",
            )
        else:
            job_pdf = st.file_uploader(
                label="Upload job description (.pdf)", type="pdf", key="job_search_pdf"
            )
            if job_pdf:
                pdf_to_text = PDFToText(job_pdf)
                job_query = pdf_to_text.extract_text()
                st.text_area(
                    "Parsed job description", job_query, height=200, disabled=True
                )
    else:
        job_query = st.text_input(
            "Search Query",
            help="Enter a specific search query like 'Web Developer with 5+ years experience in React, located in Berlin'",
            placeholder="Job title, skills, experience level, location, etc...",
        )

    # Number of candidates to return
    top_k = st.slider(
        "Number of candidates to return", min_value=1, max_value=20, value=5
    )

    # Advanced options
    with st.expander("Advanced Options"):
        include_vector_search = st.checkbox(
            "Include vector similarity search",
            value=True,
            help="Search using semantic embedding similarity",
        )
        include_graph_search = st.checkbox(
            "Include graph-based search",
            value=True,
            help="Search using graph relationships (skills, location, experience)",
        )
        min_skill_match = st.slider(
            "Minimum skill match ratio",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.1,
            help="Minimum ratio of matched skills (for graph search)",
        )

    if st.button("Find Candidates 🚀"):
        if not job_query.strip():
            st.warning("⚠️ Please enter a job description or search query!")
            return

        with st.spinner("Searching for matching candidates..."):
            try:
                # Build request parameters
                request_data = {
                    "job_query": job_query,
                    "top_k": top_k,
                    "include_vector_search": include_vector_search,
                    "include_graph_search": include_graph_search,
                    "min_skill_match": min_skill_match,
                    "predictor_type": "graph_rag",  # Always use graph_rag for candidate search
                }

                response = requests.post(
                    f"{API_URL}/search_candidates",
                    json=request_data,
                    timeout=180,
                )

                if response.status_code == HTTPStatus.OK:
                    data = response.json()
                    candidates = data.get("candidates", [])

                    if not candidates:
                        st.warning("No matching candidates found.")
                        return

                    # Display results
                    st.subheader(f"📊 Found {len(candidates)} matching candidates")

                    for i, candidate in enumerate(candidates):
                        with st.expander(
                            f"#{i+1}: {candidate['name']} - Score: {candidate['combined_score']:.2f}"
                        ):
                            col1, col2 = st.columns([1, 2])

                            with col1:
                                st.markdown("### Scores")
                                st.markdown(
                                    f"**Combined Score:** {candidate['combined_score']:.2f}"
                                )
                                if include_vector_search:
                                    st.markdown(
                                        f"**Vector Score:** {candidate.get('vector_score', 0):.2f}"
                                    )
                                if include_graph_search:
                                    st.markdown(
                                        f"**Graph Score:** {candidate.get('graph_score', 0):.2f}"
                                    )

                                st.markdown("### Matched Skills")
                                skills = candidate.get("matched_skills", [])
                                if skills:
                                    for skill in skills:
                                        st.markdown(f"- {skill}")
                                else:
                                    st.markdown("No specific skill matches found")

                            with col2:
                                st.markdown("### Analysis")
                                explanation = candidate.get("explanation", "")
                                if explanation:
                                    # Check if explanation contains XML tags and clean them
                                    explanation = re.sub(
                                        r"<thought>.*?</thought>",
                                        "",
                                        explanation,
                                        flags=re.DOTALL,
                                    )
                                    explanation = re.sub(
                                        r"<score>.*?</score>",
                                        "",
                                        explanation,
                                        flags=re.DOTALL,
                                    )
                                    explanation = re.sub(
                                        r"<explanation>(.*?)</explanation>",
                                        r"\1",
                                        explanation,
                                        flags=re.DOTALL,
                                    )
                                    explanation = explanation.strip()
                                    st.markdown(explanation)

                                # Display LLM score if provided
                                if "llm_score" in candidate.get("pros_cons", {}):
                                    llm_score = candidate["pros_cons"]["llm_score"]
                                    st.markdown(f"**LLM Score**: {llm_score}")

                                # Display pros and cons
                                pros_cons = candidate.get("pros_cons", {})
                                if pros_cons:
                                    col_a, col_b = st.columns(2)
                                    with col_a:
                                        st.markdown("#### Pros")
                                        pros = pros_cons.get("pros", [])
                                        if pros:
                                            for pro in pros:
                                                st.markdown(f"- {pro}")
                                        else:
                                            st.markdown("None specified")

                                    with col_b:
                                        st.markdown("#### Cons")
                                        cons = pros_cons.get("cons", [])
                                        if cons:
                                            for con in cons:
                                                st.markdown(f"- {con}")
                                        else:
                                            st.markdown("None specified")

                else:
                    st.error(f"API Error: {response.status_code} - {response.text}")

            except requests.exceptions.RequestException as e:
                st.error(f"Connection Error: {str(e)}")


if __name__ == "__main__":
    main()
