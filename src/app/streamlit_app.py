import os
import time
from http import HTTPStatus

import requests
import streamlit as st
from tools import PDFToText  # type: ignore

API_URL = os.getenv("API_URL", "http://localhost:8000")


def check_pod_status():
    """Check pod status without causing page refreshes."""
    try:
        response = requests.get(f"{API_URL}/pods", timeout=180)
        if response.status_code == 200:
            pods = response.json().get("pods", [])
            if pods:
                pod = pods[0]
                pod_id = pod["pod_id"]
                
                status_response = requests.get(
                    f"{API_URL}/pods/{pod_id}/status", timeout=180
                )
                if status_response.status_code == 200:
                    status_data = status_response.json()
                    if status_data is not None:
                        return {
                            "exists": True,
                            "pod_id": pod_id,
                            "status": status_data.get("status", "Unknown"),
                            "message": status_data.get("status_message", "No status message"),
                            "is_ready": status_data.get("is_ready", False)
                        }
                    else:
                        return {"exists": True, "pod_id": pod_id, "error": "Empty response from status endpoint"}
                else:
                    return {"exists": True, "pod_id": pod_id, "error": "Failed to check status"}
            else:
                return {"exists": False}
        else:
            return {"error": f"Failed to get pods: {response.status_code}"}
    except requests.exceptions.RequestException as e:
        return {"error": f"Connection error: {str(e)}"}


def render_pod_management():
    """Render the GPU pod management section with manual controls."""
    st.markdown("## 🖥️ GPU Pod Management")
    
    # Create columns for status and controls
    status_col, control_col = st.columns([1, 1])
    
    # Manual refresh button
    if control_col.button("🔄 Refresh Status", key="refresh_status"):
        st.session_state.force_status_check = True
    
    # Only check status if forced or first load
    if st.session_state.get("force_status_check", True):
        with status_col:
            with st.spinner("Checking GPU Pod status..."):
                pod_info = check_pod_status()
        
        # Clear the force check flag
        st.session_state.force_status_check = False
        st.session_state.pod_info = pod_info
    else:
        pod_info = st.session_state.get("pod_info", {"exists": False})
    
    # Display status
    with status_col:
        if "error" in pod_info:
            st.error(f"❌ {pod_info['error']}")
        elif pod_info.get("exists"):
            pod_status = pod_info.get("status", "Unknown")
            status_message = pod_info.get("message", "")
            is_ready = pod_info.get("is_ready", False)
            
            st.markdown(f"**Pod Status:** {pod_status}")
            
            if is_ready:
                st.success(f"✅ {status_message}")
            elif pod_status == "INITIALIZING":
                st.info(f"🔄 {status_message}")
                st.info("💡 Click 'Refresh Status' to check progress")
            elif pod_status == "STARTING":
                st.warning(f"⏳ {status_message}")
                st.info("💡 Click 'Refresh Status' to check progress")
            elif pod_status == "ERROR":
                st.error(f"❌ {status_message}")
            else:
                st.info(f"ℹ️ {status_message}")
        else:
            st.info("No GPU Pod running")
    
    # Control buttons
    with control_col:
        if pod_info.get("exists"):
            pod_id = pod_info.get("pod_id")
            if st.button("🛑 Terminate Pod", key="terminate_pod"):
                with st.spinner("Terminating GPU Pod..."):
                    try:
                        delete_response = requests.delete(
                            f"{API_URL}/pods/{pod_id}", timeout=180
                        )
                        if delete_response.status_code == 200:
                            st.success("Pod terminated successfully!")
                            st.session_state.force_status_check = True
                            time.sleep(2)  # Brief pause to let termination complete
                            st.rerun()
                        else:
                            st.error("Failed to terminate GPU Pod")
                    except requests.exceptions.RequestException as e:
                        st.error(f"Error terminating pod: {str(e)}")
        else:
            if st.button("🚀 Create GPU Pod", key="create_pod"):
                with st.spinner("Creating GPU Pod..."):
                    try:
                        response = requests.post(f"{API_URL}/pods", timeout=180)
                        if response.status_code == 200:
                            st.success("Pod creation initiated!")
                            st.session_state.force_status_check = True
                            time.sleep(2)  # Brief pause to let creation start
                            st.rerun()
                        else:
                            st.error("Failed to create GPU Pod")
                    except requests.exceptions.RequestException as e:
                        st.error(f"Error creating pod: {str(e)}")


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

    # GPU Pod Management with improved stability
    render_pod_management()

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

                    st.progress(score / 5)

                    if description:
                        st.subheader("🔍 Analysis")
                        st.write(description)
                else:
                    st.error(f"API Error: {response.status_code} - {response.text}")

            except requests.exceptions.RequestException as e:
                st.error(f"Connection Error: {str(e)}")


if __name__ == "__main__":
    main()
