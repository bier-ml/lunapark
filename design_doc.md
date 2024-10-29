# Design Document: ML-Powered Resume Screening System for HR Luna Park

## Table of Contents

1. [Business Task Definition](#1-business-task-definition)
   1. [Business Problem Statement](#11-business-problem-statement)
   2. [Success Criteria](#12-success-criteria)
   3. [Business Requirements](#13-business-requirements)
   4. [Typical Use Cases](#14-typical-use-cases)
2. [Technical Task Definition](#2-technical-task-definition)
   1. [Technical Problem Formulation](#21-technical-problem-formulation)
   2. [Quality Metrics and Success Criteria](#22-quality-metrics-and-success-criteria)
   3. [Solution Architecture Diagram](#23-solution-architecture-diagram)
   4. [Solution Implementation Stages](#24-solution-implementation-stages)
   5. [Data Description](#25-data-description)
3. [Project Productionization](#3-project-productionization)
   1. [Technical Architecture Diagram](#31-technical-architecture-diagram)
   2. [Infrastructure Description](#32-infrastructure-description)
   3. [Technical Requirements](#33-technical-requirements)
4. [Quality Characteristics](#4-quality-characteristics)
   1. [System Scalability](#41-system-scalability)
   2. [Performance Requirements](#42-performance-requirements)
   3. [System Reliability](#43-system-reliability)
   4. [Model Retraining / Automated Model Replacement](#44-model-retraining--automated-model-replacement)
   5. [Load Testing Results](#45-load-testing-results)
   6. [Future System Extensions](#46-future-system-extensions)

## 1. Business Task Definition

### 1.1 Business Problem Statement

HR Luna Park seeks to enhance its recruitment process by automating the initial resume screening phase. The current manual process is time-consuming, inconsistent, and heavily reliant on individual recruiters' expertise. 

Each candidate's application must pass through multiple stakeholders – first the recruiter for initial screening, then various technical experts who need to evaluate specific skills for different vacancies. This multi-stage manual review creates significant bottlenecks, as technical experts must balance their primary roles with timely candidate evaluations. 

The process becomes particularly challenging when managing multiple open positions simultaneously, as experts need to accurately rank candidates across different vacancies while maintaining quick turnaround times. This leads to delays in the hiring process and risks losing top talent to competitors who can move faster.

The current pipeline's manual nature makes it fundamentally unscalable as the company grows. With HR Luna Park's expansion plans, the volume of applications is expected to increase, making it impossible to maintain quality and speed with the existing process. Technical experts are already at capacity, and hiring additional experts solely for resume screening is not cost-effective. The AI-powered system is therefore crucial not just for optimization, but as an enabler for the company's growth strategy - allowing the screening process to scale linearly with application volume while maintaining consistent quality and quick turnaround times.

### 1.2 Success Criteria

- Reduce resume screening time.
- Achieve the particular agreement rate with expert recruiters in identifying unsuitable candidates (crucial for reducing expert workload).
- Achieve the particular agreement rate with expert recruiters for candidate ranking.
- Increase daily candidate processing capacity per recruiter by 100% (from average `20` to `40` candidates per day).

### 1.3 Business Requirements

1. Automated Resume Processing
   - Parse multiple document formats (PDF, DOCX, *LinkedIn profile*).
   - Extract relevant information about the candidate automatically.
   - Handle multilingual resumes (English, Russian, Ukranian).

2. Intelligent Matching
   - Match candidates to job requirements.
   - Score candidates based on skills and experience needed for the particular vacancy.
   - Provide explainable results in natural language.
   - Process all data using locally deployed open-source models only, as sending candidate PII to third-party LLM services (like OpenAI) is prohibited for data privacy reasons.

### 1.4 Typical Use Cases

1. **Individual Candidate Assessment**
   - Input: 
     - Candidate information (LinkedIn profile URL, resume PDF, or text description)
     - Target position requirements
   - Process: 
     - Automatic parsing and analysis
     - Skill matching and experience evaluation
   - Output:
     - Numerical score (0-100)
     - Detailed natural language explanation of the score
     - Specific strengths and potential gaps relative to the position

2. **Automated Batch Analysis**
   - Input:
     - Collection of candidate profiles accumulated over time
     - Position requirements
   - Process:
     - Batch processing and comparative analysis
     - Statistical evaluation across all candidates
   - Output:
     - Ranked list of candidates
     - Individual scores and explanations
     - Aggregate statistics and trends
     - Summary of top candidates with comparative strengths

3. **API Integration**
   - Description:
     - RESTful API endpoints for external system integration
     - Comprehensive API documentation with OpenAPI/Swagger
     - Authentication using API keys or OAuth2
    
   - Example Endpoints:
     <details>
       <summary>Click to expand</summary>

       ```json
       POST /api/v1/match
       {
         "vacancy_description": "We are looking for a Senior Backend Developer with 5+ years of experience in Python, FastAPI, and PostgreSQL. The role is remote-friendly and requires strong system design skills.",
         "candidate_description": "Senior software engineer with 7 years of experience in web development. Expert in Python, having built multiple production services using FastAPI. Familiar with PostgreSQL through side projects. Contributed to open-source projects and mentored junior developers.",
         "predictor_type": "lm",
         "predictor_parameters": {
           "api_base_url": "http://localhost:1234/v1",
           "model": "mistral-7b-instruct"
         }
       }
       ```

     </details>
   - Example Response:
     <details>
       <summary>Click to expand</summary>

     ```json
     {
       "score": 85.5,
       "description": "Strong match for the Senior Backend Developer position. Key strengths: 7 years of Python development experience and extensive FastAPI usage in production. Areas for consideration: PostgreSQL experience is limited to side projects rather than production systems. Overall, the candidate's technical expertise and experience level align well with the core requirements."
     }
     ```
     </details>


### 2. Technical Task Definition

The technical task for this project is designed to meet the business requirements by building two core machine learning components: a **Resume-Job Matching** system and a **Success Prediction** model. Together, these models automate resume screening by matching candidate resumes to job requirements and estimating hiring likelihood based on historical hiring data.

### 2.1 Technical Problem Formulation

The project consists of two primary ML tasks that align with the business objectives:

1. **Resume-Job Matching**:

    - **Objective**: Develop a similarity scoring system to match the content of resumes with specific job requirements, which aids recruiters in ranking candidates.
    - **Methods**:
        - **Text Embedding and Semantic Comparison**: Convert job descriptions and resumes into embeddings using models like BERT to capture contextual similarities.
        - **Skills and Experience Extraction**: Identify and validate key skills and experience within resumes, comparing these with job description requirements to enhance the matching accuracy.
        - **Algorithms**: Use semantic similarity techniques, cosine similarity, and possibly transformer-based models like BERT or DistilBERT for feature extraction and comparison.
        - **LLM Integration**: Leverage large language models to enhance understanding of context and nuances in resumes and job descriptions, improving the accuracy of matching through advanced natural language understanding and generation capabilities.

2. **Success Prediction**:
    - **Objective**: Develop a predictive model to assess the probability of a candidate’s success throughout the hiring process.
    - **Methods**:
        - **Binary Classification**: Predict the likelihood of successful hiring by analyzing patterns in historical hiring data.
        - **Multi-stage Prediction**: Implement intermediate prediction stages for each interview phase to refine success estimates throughout the recruitment process.
        - **Probability Scoring**: Assign a probability score representing the hiring likelihood, providing a basis for further recruiter consideration.
        - **LLM Integration**: Utilize large language models to analyze candidate responses and feedback during interviews, enhancing the predictive model's ability to assess candidate success based on nuanced language patterns and contextual understanding.

The main focus will be on LLM methods, but other approaches may also be considered if they prove effective.

### 2.2 Quality Metrics and Success Criteria

The solution’s quality and success will be measured by metrics that correspond to business goals and ML model requirements. These metrics evaluate the models’ performance, prediction accuracy, and system efficiency.

| Metric             | Target                                                                                                        | Measurement Method                          | Business Goal Alignment                       |
| ------------------ | ------------------------------------------------------------------------------------------------------------- | ------------------------------------------- | --------------------------------------------- |
| **Model Accuracy** | ≤ 5% of predictions differ by more than 1 point from expert evaluations; ≤ 20% differ by more than 0.5 points | Compare model predictions to expert ratings | Ensures model reliability in screening        |
| **Response Time**  | ≤ 60 seconds                                                                                                  | Monitor response time for API endpoints     | Supports efficient candidate screening        |
| **System Uptime**  | ≥ 99.9%                                                                                                       | Infrastructure monitoring                   | Ensures high availability of screening system |

### 2.3 Solution Architecture Diagram

The solution is designed to operate as a microservices-based system, facilitating integration with HR Luna Park’s existing databases and ensuring scalability. The architecture consists of components for frontend interaction, backend processing, ML model inference, and data storage.

```mermaid
graph TD
    A[Frontend] --> B[Backend]
    B --> C[AI Platform]
    C --> D[Model Server]
    B --> E[Database]
    D --> F[Model Storage]
    F --> G[Embedding Models - BERT]
```

This architecture supports:

-   **Frontend**: The recruiter-facing interface (e.g., Streamlit), allowing resume uploads, job description inputs, and viewing ranked candidates.
-   **Backend**: FastAPI serves as the core API layer, interfacing with ML models and handling data processing tasks.
-   **AI Platform and Model Server**: Hosts ML models for resume-job matching and success prediction, i.e. managed by the vLLM model server.
-   **Embedding Models**: Models like BERT or DistilBERT are used to create semantic embeddings for text-based matching.
-   **Database**: A database stores resumes, job descriptions, model outcomes, and feedback data for ongoing model retraining and performance monitoring.

### 2.4 Solution Implementation Stages

The solution is developed in multiple stages, each focusing on data preparation, feature engineering, model development, and evaluation. Each stage builds on the previous one to ensure a robust solution that meets the business objectives.

1. **Data Preparation & ETL**:
    - **Objective**: Prepare existing historical data on resumes, job descriptions, and hiring outcomes for analysis and modeling.
    - **Tasks**:
        - Standardize and anonymize historical resume data.
        - Prepare labeled data for model training, including success labels based on prior hires.
        - Build an ETL pipeline to automate data cleaning, validation, and feature extraction.
2. **Feature Engineering**:

    - **Objective**: Transform raw text data into meaningful features for ML models, incorporating both traditional and LLM approaches.
    - **Code Example**:
        ```python
        def process_resume(text):
            # Remove PII
            text = remove_personal_info(text)
            # Extract key sections
            sections = extract_sections(text)
            # Generate embeddings
            embeddings = bert_model.encode(text)
            return create_feature_vector(sections, embeddings)
        ```
    - **Tasks**:
        - Extract key skills and experiences from resumes using traditional methods.
        - Embed job descriptions and resumes for semantic similarity comparison using both traditional models (e.g., BERT) and LLMs.
        - Utilize LLMs to analyze vacancies and resumes, outputting a score and a short comment about the candidate's fit for the position.
        - Create features from historical interview outcomes to train the success prediction model, potentially integrating insights derived from LLM analyses.

3. **Model Development Phases**:

    - **MVP**: Develop a baseline binary classification model using traditional ML algorithms like logistic regression or random forests to establish a preliminary resume-job matching system.
    - **Advanced Model**: Implement BERT or a similar transformer-based model for embedding-based matching. Train a transformer-based model to compare resumes and job descriptions, using fine-tuned embeddings for greater accuracy in matching.
    - **LLM Model Development**: Fine-tune a large language model specifically for analyzing resumes and job descriptions. This phase will involve:
        - Developing effective prompt engineering strategies to elicit meaningful outputs from the LLM, such as scores and comments on candidate fit.
        - Training the LLM on domain-specific data to enhance its understanding of recruitment nuances.
    - **Production Model**: Apply the most successful, potentially an ensemble model, combining BERT-based embeddings, a binary classifier, and the fine-tuned LLM, to improve prediction accuracy for hiring success.

4. **Evaluation and Model Selection**:

    - Compare the MVP and advanced models using cross-validation to determine model accuracy, false positive rate, and overall performance.
    - Select the model configuration that best meets quality metrics for deployment.

5. **Deployment and Integration**:
    - Deploy the final model into production, integrate it with HR Luna Park’s backend systems, and configure API endpoints for live resume screening.
    - Establish monitoring for quality metrics, including model accuracy, response time, and system uptime.

### 2.5 Data Description

The project uses a combination of structured and unstructured data sources, including resumes and job descriptions. Other data types, such as interview outcomes, are covered by a non-disclosure agreement. Each data type is standardized, processed, and transformed to fit the needs of the ML models.

| Data Type              | Source               | Volume | Update Frequency | Description                   |
| ---------------------- | -------------------- | ------ | ---------------- | ----------------------------- |
| **Resumes**            | Provided by Lunapark | NDA    | NDA              | Text resumes from applicants. |
| **Job Descriptions**   | Provided by Lunapark | NDA    | NDA              | Job details.                  |
| **Interview Outcomes** | Provided by Lunapark | NDA    | NDA              | Historical hiring outcomes.   |

**Exploratory Data Analysis**:
An initial EDA was conducted to assess data completeness, identify missing values, and determine relevant features for both matching and prediction tasks. However, the specific findings and insights from the EDA are covered by a non-disclosure agreement.

## 3. Project Productionization

### 3.1 Technical Architecture Diagram

```mermaid
graph TD
    A[Load Balancer] --> B[API Cluster]
    B --> C[Model Inference]
    C --> D[GPU Pool]
    B --> E[Database Cluster]
```

### 3.2 Infrastructure Description

1. **Compute Resources**

    - API Servers: 4x t2.large
    - Model Servers: 2x g4dn.xlarge
    - Database: RDS r5.large

2. **Storage Requirements**
    - Model Artifacts: 100GB
    - Document Storage: 500GB/year
    - Database: 1TB with replication

### 3.3 Technical Requirements

1. **Performance**

    - Latency: < 2s per request
    - Throughput: 100 requests/second
    - Concurrent Users: 50

2. **Security**
    - Data Encryption at rest
    - HTTPS/TLS
    - Role-based access control

## 4. Quality Characteristics

### 4.1 System Scalability

-   Horizontal scaling of API servers
-   Auto-scaling based on load
-   Database read replicas
-   Distributed model inference

### 4.2 Performance Requirements

| Component       | Metric           | Target       |
| --------------- | ---------------- | ------------ |
| API Response    | P95 Latency      | < 2s         |
| Model Inference | Batch Processing | 50 resumes/s |
| Database        | Query Response   | < 100ms      |

### 4.3 System Reliability

-   High Availability: 99.9% uptime
-   Automated failover
-   Regular backups
-   Error monitoring and alerting

### 4.4 Model Retraining / Automated Model Replacement

1. **Monitoring Triggers**

    - Performance degradation
    - Data drift detection
    - Weekly evaluation cycles

2. **Retraining Pipeline**
    - Automated data collection
    - A/B testing framework
    - Shadow deployment

### 4.5 Load Testing Results

| Concurrent Users | Response Time (ms) | Error Rate (%) |
| ---------------- | ------------------ | -------------- |
| 10               | 150                | 0              |
| 50               | 300                | 0.1            |
| 100              | 600                | 0.5            |
| 500              | 1200               | 2.0            |

### 4.6 Future System Extensions

1. **Enhanced Features**

    - Multi-language support
    - Video interview integration
    - Automated reference checking

2. **Technical Improvements**
    - Real-time processing
    - Advanced analytics dashboard
    - Integration with additional HR systems
