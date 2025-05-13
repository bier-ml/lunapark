"""
Load sample resume and job data for demonstration purposes.

This script creates sample candidate profiles in the knowledge graph
and embeddings store for testing the GraphRAG-based resume-job matching system.
"""

import json
import os
import sys
from typing import Dict, List, Optional

# Add src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from src.platform.rag.create_vector_indexes import create_candidate_vector_index
from src.platform.rag.graph_rag_predictor import GraphRAGPredictor

# Sample resumes for demonstration
SAMPLE_RESUMES = [
    {
        "name": "Alex Johnson",
        "location": "San Francisco, CA",
        "summary": "Senior software engineer with 8+ years of experience in web development and cloud architecture.",
        "skills": [
            "JavaScript",
            "React",
            "Node.js",
            "AWS",
            "Python",
            "Docker",
            "Kubernetes",
        ],
        "experience": [
            {
                "title": "Senior Software Engineer",
                "company": "TechCorp Inc.",
                "years": 3.5,
                "start_date": "2020-01",
                "end_date": "Present",
                "description": "Lead developer for cloud-native applications, responsible for architecture and implementation.",
            },
            {
                "title": "Full Stack Developer",
                "company": "WebSolutions",
                "years": 4.0,
                "start_date": "2016-01",
                "end_date": "2019-12",
                "description": "Developed and maintained client-facing web applications using React and Node.js.",
            },
        ],
        "text": """
        Alex Johnson
        San Francisco, CA | alex.johnson@example.com | (555) 123-4567
        
        SUMMARY
        Senior software engineer with 8+ years of experience in web development and cloud architecture.
        Strong expertise in JavaScript/React frontend and Node.js/Python backend development.
        
        EXPERIENCE
        TechCorp Inc. - Senior Software Engineer (Jan 2020 - Present)
        - Lead developer for cloud-native applications using React, Node.js, and AWS
        - Implemented CI/CD pipelines with Docker and Kubernetes
        - Reduced deployment time by 75% through automation
        - Mentored junior developers and conducted code reviews
        
        WebSolutions - Full Stack Developer (Jan 2016 - Dec 2019)
        - Developed and maintained client-facing web applications using React and Node.js
        - Implemented RESTful APIs with Express.js and MongoDB
        - Improved application performance by 40% through code optimization
        
        SKILLS
        Programming: JavaScript, Python, TypeScript, HTML, CSS
        Frameworks/Libraries: React, Node.js, Express, Redux, Jest
        Tools/Platforms: AWS (EC2, S3, Lambda), Docker, Kubernetes, Git
        """,
    },
    {
        "name": "Emily Chen",
        "location": "New York, NY",
        "summary": "Data scientist with 5 years of experience in machine learning and statistical analysis.",
        "skills": [
            "Python",
            "TensorFlow",
            "PyTorch",
            "SQL",
            "R",
            "Data Visualization",
            "Natural Language Processing",
        ],
        "experience": [
            {
                "title": "Senior Data Scientist",
                "company": "DataMinds Analytics",
                "years": 2.0,
                "start_date": "2021-03",
                "end_date": "Present",
                "description": "Lead ML model development for financial prediction and natural language processing projects.",
            },
            {
                "title": "Data Scientist",
                "company": "TechAnalytica",
                "years": 3.0,
                "start_date": "2018-02",
                "end_date": "2021-02",
                "description": "Developed predictive models and data pipelines for customer segmentation and recommendation systems.",
            },
        ],
        "text": """
        Emily Chen
        New York, NY | emily.chen@example.com | (555) 987-6543
        
        SUMMARY
        Data scientist with 5 years of experience in machine learning and statistical analysis.
        Expertise in Python, TensorFlow, and NLP with a focus on financial applications.
        
        EXPERIENCE
        DataMinds Analytics - Senior Data Scientist (Mar 2021 - Present)
        - Lead machine learning model development for financial prediction
        - Implemented NLP solutions for sentiment analysis of financial news
        - Improved prediction accuracy by 15% using ensemble methods
        - Managed a team of 3 junior data scientists
        
        TechAnalytica - Data Scientist (Feb 2018 - Feb 2021)
        - Developed predictive models for customer segmentation
        - Created recommendation systems using collaborative filtering
        - Built data pipelines for ETL processes using Airflow
        - Presented findings to non-technical stakeholders
        
        EDUCATION
        M.S. in Data Science, Columbia University, 2018
        B.S. in Computer Science, Cornell University, 2016
        
        SKILLS
        Programming: Python, R, SQL
        ML/AI: TensorFlow, PyTorch, scikit-learn, NLTK, spaCy
        Data: Pandas, NumPy, SQL, Tableau, Power BI
        Tools: Git, Docker, AWS, Google Cloud
        """,
    },
    {
        "name": "Michael Rodriguez",
        "location": "Austin, TX",
        "summary": "UX/UI designer with 6 years of experience creating user-centered design solutions.",
        "skills": [
            "UI Design",
            "UX Research",
            "Figma",
            "Adobe XD",
            "Sketch",
            "Prototyping",
            "User Testing",
        ],
        "experience": [
            {
                "title": "Senior UX Designer",
                "company": "Creative Design Studio",
                "years": 3.0,
                "start_date": "2020-06",
                "end_date": "Present",
                "description": "Lead UX designer for mobile and web applications, conducting user research and creating wireframes and prototypes.",
            },
            {
                "title": "UI Designer",
                "company": "TechUI",
                "years": 3.0,
                "start_date": "2017-05",
                "end_date": "2020-05",
                "description": "Created user interfaces for web and mobile applications following design systems and accessibility guidelines.",
            },
        ],
        "text": """
        Michael Rodriguez
        Austin, TX | michael.rodriguez@example.com | (555) 456-7890
        
        SUMMARY
        UX/UI designer with 6 years of experience creating user-centered design solutions.
        Skilled in user research, wireframing, prototyping, and visual design.
        
        EXPERIENCE
        Creative Design Studio - Senior UX Designer (Jun 2020 - Present)
        - Lead UX designer for mobile and web applications across various industries
        - Conducted user research and usability testing to inform design decisions
        - Created wireframes, prototypes, and high-fidelity designs using Figma
        - Collaborated with development teams to ensure design implementation
        
        TechUI - UI Designer (May 2017 - May 2020)
        - Designed user interfaces for web and mobile applications
        - Created and maintained design systems and component libraries
        - Ensured designs met accessibility standards (WCAG 2.1)
        - Collaborated with UX researchers to incorporate user feedback
        
        EDUCATION
        B.F.A. in Graphic Design, Rhode Island School of Design, 2017
        
        SKILLS
        Design: UI Design, UX Research, Visual Design, Design Systems, Accessibility
        Tools: Figma, Adobe XD, Sketch, InVision, Adobe Creative Suite
        Research: User Testing, Heuristic Evaluation, A/B Testing, User Interviews
        Collaboration: Agile/Scrum, Jira, Confluence
        """,
    },
    {
        "name": "Sarah Williams",
        "location": "Chicago, IL",
        "summary": "Marketing manager with 7 years of experience in digital marketing and brand strategy.",
        "skills": [
            "Digital Marketing",
            "SEO",
            "Content Strategy",
            "Social Media",
            "Google Analytics",
            "Email Marketing",
        ],
        "experience": [
            {
                "title": "Marketing Manager",
                "company": "Global Brands Inc.",
                "years": 4.0,
                "start_date": "2019-08",
                "end_date": "Present",
                "description": "Led digital marketing strategy and campaigns, increasing online engagement by 45% and sales by 30%.",
            },
            {
                "title": "Digital Marketing Specialist",
                "company": "MarketPro Agency",
                "years": 3.0,
                "start_date": "2016-07",
                "end_date": "2019-07",
                "description": "Managed SEO, content, and social media campaigns for B2B and B2C clients across multiple industries.",
            },
        ],
        "text": """
        Sarah Williams
        Chicago, IL | sarah.williams@example.com | (555) 789-0123
        
        SUMMARY
        Marketing manager with 7 years of experience in digital marketing and brand strategy.
        Proven track record of increasing engagement, conversion rates, and ROI.
        
        EXPERIENCE
        Global Brands Inc. - Marketing Manager (Aug 2019 - Present)
        - Led digital marketing strategy and campaigns for a global consumer goods company
        - Managed a team of 5 marketing specialists and a $1M annual budget
        - Increased online engagement by 45% and e-commerce sales by 30% YoY
        - Implemented data-driven marketing approach using Google Analytics and Tableau
        
        MarketPro Agency - Digital Marketing Specialist (Jul 2016 - Jul 2019)
        - Managed SEO, content marketing, and social media campaigns for 12+ clients
        - Improved client website rankings by an average of 15 positions for target keywords
        - Created and optimized Google and Meta ad campaigns with 200% ROI
        - Developed email marketing campaigns with 25% open rates and 10% conversion
        
        EDUCATION
        M.B.A., Marketing, University of Chicago Booth School of Business, 2016
        B.A., Communications, Northwestern University, 2014
        
        SKILLS
        Marketing: Digital Marketing, Brand Strategy, Content Marketing, Campaign Management
        Technical: SEO, SEM, Google Analytics, Meta Business Suite, Mailchimp, HubSpot
        Analysis: Google Data Studio, Tableau, Excel, A/B Testing
        """,
    },
    {
        "name": "David Lee",
        "location": "Seattle, WA",
        "summary": "DevOps engineer with 5 years of experience in cloud infrastructure and CI/CD pipelines.",
        "skills": [
            "AWS",
            "Azure",
            "Docker",
            "Kubernetes",
            "Terraform",
            "Jenkins",
            "Python",
            "Bash",
        ],
        "experience": [
            {
                "title": "Senior DevOps Engineer",
                "company": "CloudTech Solutions",
                "years": 2.5,
                "start_date": "2020-09",
                "end_date": "Present",
                "description": "Led cloud infrastructure design and implementation, reducing deployment time by 80% and infrastructure costs by 35%.",
            },
            {
                "title": "DevOps Engineer",
                "company": "InfrastructurePro",
                "years": 2.5,
                "start_date": "2018-02",
                "end_date": "2020-08",
                "description": "Implemented CI/CD pipelines and maintained cloud infrastructure for enterprise clients.",
            },
        ],
        "text": """
        David Lee
        Seattle, WA | david.lee@example.com | (555) 234-5678
        
        SUMMARY
        DevOps engineer with 5 years of experience in cloud infrastructure and CI/CD pipelines.
        Expertise in AWS, Azure, Kubernetes, and infrastructure as code.
        
        EXPERIENCE
        CloudTech Solutions - Senior DevOps Engineer (Sep 2020 - Present)
        - Led cloud infrastructure design and implementation for enterprise clients
        - Developed Terraform modules for standardized, repeatable infrastructure deployments
        - Reduced deployment time by 80% and infrastructure costs by 35%
        - Implemented Kubernetes clusters for microservices architecture
        
        InfrastructurePro - DevOps Engineer (Feb 2018 - Aug 2020)
        - Implemented CI/CD pipelines using Jenkins, GitLab CI, and GitHub Actions
        - Maintained AWS and Azure cloud infrastructure for 15+ applications
        - Automated routine operations with Python and Bash scripting
        - Implemented monitoring and alerting using Prometheus and Grafana
        
        EDUCATION
        B.S. in Computer Science, University of Washington, 2018
        
        CERTIFICATIONS
        AWS Certified DevOps Engineer - Professional
        Microsoft Certified: Azure DevOps Engineer Expert
        Certified Kubernetes Administrator (CKA)
        
        SKILLS
        Cloud: AWS (EC2, S3, Lambda, ECS), Azure, Google Cloud Platform
        Infrastructure: Docker, Kubernetes, Terraform, Ansible, Packer
        CI/CD: Jenkins, GitLab CI, GitHub Actions, CircleCI
        Monitoring: Prometheus, Grafana, ELK Stack, New Relic
        Programming: Python, Bash, Go
        """,
    },
]

# Sample job descriptions for testing
SAMPLE_JOBS = [
    {
        "title": "Senior Software Engineer",
        "description": """
        We are looking for a Senior Software Engineer with strong experience in web development 
        and cloud architecture. The ideal candidate will have expertise in JavaScript/React frontend 
        and Node.js backend development. Experience with AWS and containerization (Docker, Kubernetes) 
        is highly desired.
        
        Responsibilities:
        - Develop and maintain web applications using React, Node.js, and other modern web technologies
        - Design and implement cloud-native architecture on AWS
        - Write clean, maintainable, and efficient code
        - Mentor junior developers and conduct code reviews
        
        Requirements:
        - 5+ years of experience in software development
        - Strong proficiency in JavaScript/TypeScript, React, and Node.js
        - Experience with AWS services (EC2, S3, Lambda, etc.)
        - Knowledge of containerization and orchestration (Docker, Kubernetes)
        - Good understanding of CI/CD pipelines
        - Excellent problem-solving and communication skills
        
        Nice to have:
        - Experience with Python
        - Knowledge of serverless architecture
        - Experience with microservices architecture
        """,
    },
    {
        "title": "Data Scientist",
        "description": """
        We are seeking a talented Data Scientist to join our analytics team. The ideal candidate 
        will have experience in machine learning, statistical analysis, and data visualization. 
        You will work on developing predictive models and extracting insights from large datasets.
        
        Responsibilities:
        - Develop machine learning models for prediction and classification
        - Perform statistical analysis on large datasets
        - Create data visualizations to communicate findings to stakeholders
        - Collaborate with engineering teams to implement models in production
        
        Requirements:
        - 3+ years of experience in data science or related field
        - Strong proficiency in Python and data science libraries (pandas, NumPy, scikit-learn)
        - Experience with machine learning frameworks (TensorFlow, PyTorch)
        - Solid understanding of statistics and probability
        - Knowledge of SQL and database concepts
        
        Nice to have:
        - Experience with NLP or computer vision
        - Knowledge of big data technologies (Spark, Hadoop)
        - Background in finance or healthcare analytics
        """,
    },
    {
        "title": "UX/UI Designer",
        "description": """
        We are looking for a creative UX/UI Designer to create amazing user experiences. The ideal 
        candidate should have a strong portfolio demonstrating their ability to create user-centered 
        designs and a good understanding of user research methodologies.
        
        Responsibilities:
        - Conduct user research and usability testing
        - Create wireframes, prototypes, and high-fidelity designs
        - Collaborate with product managers and developers
        - Maintain and evolve our design system
        
        Requirements:
        - 3+ years of experience in UX/UI design
        - Proficiency in design tools like Figma, Adobe XD, or Sketch
        - Experience conducting user research and usability testing
        - Understanding of accessibility standards
        - Portfolio demonstrating user-centered design processes
        
        Nice to have:
        - Experience with design systems
        - Knowledge of HTML/CSS
        - Background in mobile app design
        """,
    },
]


def main():
    """Load sample resume data for demonstration purposes."""
    print("Loading sample resume and job data for demonstration...")

    # Initialize the GraphRAG predictor
    predictor = GraphRAGPredictor(
        neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
        neo4j_password=os.getenv("NEO4J_PASSWORD", "password"),
    )

    try:
        print(f"Connected to Neo4j at {predictor.neo4j_uri}")

        # First clear any existing data to avoid duplicates
        with predictor.driver.session() as session:
            # Delete all nodes and relationships
            print("Cleaning existing data...")
            session.run("MATCH (n) DETACH DELETE n")
            print("Database cleared successfully")

        # Create the vector index for candidates
        print("Creating vector index for candidates...")
        create_candidate_vector_index()

        # Load the sample resumes
        for i, resume in enumerate(SAMPLE_RESUMES):
            print(f"Adding candidate {i+1}/{len(SAMPLE_RESUMES)}: {resume['name']}...")

            # Parse and store the resume
            candidate_id = f"sample-{i+1}"
            parsed_resume = predictor.parse_and_store_resume(
                resume["text"], candidate_id=candidate_id
            )

            print(
                f"  - Added candidate {parsed_resume.get('name', resume['name'])} with ID {candidate_id}"
            )

            # Add additional data if the parser didn't catch it
            if (
                not parsed_resume.get("skills")
                or len(parsed_resume.get("skills", [])) < 3
            ):
                print(f"  - Adding skills manually: {', '.join(resume['skills'])}")
                for skill in resume["skills"]:
                    predictor.knowledge_graph.add_skill_to_candidate(
                        candidate_id, skill
                    )

            if not parsed_resume.get("location"):
                print(f"  - Adding location manually: {resume['location']}")
                predictor.knowledge_graph.add_location_to_candidate(
                    candidate_id, resume["location"]
                )

            # Add embedding to candidate for vector search
            with predictor.driver.session() as session:
                # Get candidate profile text
                candidate_text = f"{resume['name']} - {resume['location']} - {' '.join(resume['skills'])} - {resume['text'][:500]}"

                # Generate embedding
                embedding = predictor.embedder.embed_query(candidate_text)

                # Store embedding
                session.run(
                    """
                    MATCH (c:Candidate {id: $id})
                    SET c.embedding = $embedding
                    """,
                    id=candidate_id,
                    embedding=embedding,
                )
                print(f"  - Added embedding to candidate {candidate_id}")

        # Verify data was loaded correctly
        print("\nVerifying data load...")
        with predictor.driver.session() as session:
            # Check candidates
            result = session.run("MATCH (c:Candidate) RETURN count(c) as count")
            candidate_count = result.single()["count"] if result.peek() else 0
            print(f"- Found {candidate_count} candidates in Neo4j")

            # Check skills
            result = session.run("MATCH (s:Skill) RETURN count(s) as count")
            skill_count = result.single()["count"] if result.peek() else 0
            print(f"- Found {skill_count} unique skills in Neo4j")

            # Check candidate-skill relationships
            result = session.run(
                "MATCH (:Candidate)-[r:HAS_SKILL]->(:Skill) RETURN count(r) as count"
            )
            skill_rel_count = result.single()["count"] if result.peek() else 0
            print(f"- Found {skill_rel_count} candidate-skill relationships in Neo4j")

            # Check locations
            result = session.run("MATCH (l:Location) RETURN count(l) as count")
            location_count = result.single()["count"] if result.peek() else 0
            print(f"- Found {location_count} unique locations in Neo4j")

            # Run a test vector search
            print("\nTesting vector search functionality...")
            try:
                vector_results = predictor.score_candidates_vector(
                    "software engineer", top_k=2
                )
                print(f"- Vector search test: Found {len(vector_results)} candidates")
                if vector_results:
                    print(
                        f"  Top result: {vector_results[0].get('name', vector_results[0].get('id'))} with score {vector_results[0].get('similarity_score', 0):.2f}"
                    )
            except Exception as e:
                print(f"- Vector search test failed: {e}")

            # Run a test graph search
            print("\nTesting graph search functionality...")
            try:
                graph_results = predictor.score_candidates_graph(
                    "software engineer with React experience"
                )
                print(f"- Graph search test: Found {len(graph_results)} candidates")
                if graph_results:
                    print(
                        f"  Top result: {graph_results[0].get('name', '')} with {graph_results[0].get('match_count', 0)} matches"
                    )
            except Exception as e:
                print(f"- Graph search test failed: {e}")

        print("\nSample data loaded successfully!")
        print("\nYou can now use the GraphRAG resume-job matching system:")
        print("1. Open the frontend at http://localhost:8501")
        print("2. Go to the 'Job Search' tab")
        print("3. Enter a job description or use one of these sample queries:")
        for job in SAMPLE_JOBS:
            print(f"   - '{job['title']}': {job['description'].strip()[:100]}...'")

        print("\nTo debug the system, open this URL in your browser:")
        print("http://localhost:8000/debug/candidates")

    except Exception as e:
        print(f"Error loading sample data: {str(e)}")
        import traceback

        traceback.print_exc()
    finally:
        predictor.close()


if __name__ == "__main__":
    main()
