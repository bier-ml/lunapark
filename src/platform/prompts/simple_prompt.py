from textwrap import dedent

FEW_SHOT_EXAMPLES = dedent("""
    <vacancy_description>
    We are seeking a Senior Software Engineer with expertise in Java and Spring Framework. The ideal candidate should have experience with microservices architecture, RESTful APIs, and cloud platforms like AWS or Azure. Responsibilities include designing scalable applications, leading a team of developers, and collaborating with cross-functional teams to deliver high-quality software solutions. Our company, Tech Innovators Inc., is a leader in providing cutting-edge technology solutions to clients worldwide.
    </vacancy_description>

    <candidate_description>
    John Doe is a seasoned software engineer with over 8 years of experience in software development. He holds a Bachelor’s degree in Computer Science from XYZ University. John has extensive experience with Java, Spring Boot, and has worked on projects involving microservices architecture. He is proficient in AWS and has led a team of developers at his previous company, Innovatech Solutions, where he was instrumental in developing scalable applications and improving system performance.
    </candidate_description>
                          
    <thought>
    John's experience closely aligns with the job requirements. His proficiency in Java and Spring Framework, along with his experience in microservices and AWS, make him a strong candidate. His leadership experience and history of developing scalable applications further enhance his fit for the role.
    </thought>
                          
    <score>
    90
    </score>
    
    ---
                          
    <vacancy_description>
    We are looking for a Digital Marketing Specialist with expertise in SEO, content marketing, and social media management. The candidate should have at least 3 years of experience in digital marketing and a strong understanding of Google Analytics and SEO tools. Responsibilities include developing and implementing marketing strategies, optimizing content for search engines, and managing social media campaigns. Bright Minds Agency is a dynamic marketing firm specializing in innovative digital solutions.
    </vacancy_description>

    <candidate_description>
    Jane Smith is a digital marketing professional with 4 years of experience. She holds a Master’s degree in Marketing from ABC University. Jane has a proven track record in SEO optimization, content creation, and managing successful social media campaigns. At her previous job with Creative Marketing Co., she increased organic traffic by 50% and improved search engine rankings for key clients. She is proficient in Google Analytics, SEMrush, and Hootsuite.
    </candidate_description>

    <thought>
    Jane's background matches the requirements perfectly. Her experience in SEO, content marketing, and social media management aligns with the role's focus. Her proven results in increasing organic traffic and improving search rankings demonstrate her capability to deliver impactful marketing strategies.
    </thought>
                                    
    <score>
    95
    </score>

    ---
                          
    <vacancy_description>
    We are seeking a Quantitative Research Analyst with a strong background in statistical modeling, machine learning, and financial markets. The ideal candidate should have at least 5 years of experience in quantitative research, proficiency in Python, R, and experience with algorithmic trading strategies. Responsibilities include developing predictive models, analyzing large financial datasets, and collaborating with traders to implement strategies. Global Finance Corp. is a leader in innovative financial solutions and algorithmic trading.
    </vacancy_description>

    <candidate_description>
    A professional with 6 years of experience as a Software Engineer specializing in backend development using Java and C++. They hold a Master’s degree in Computer Engineering and have worked on developing enterprise-level software applications. Their expertise includes database management, API development, and optimizing system performance. They have limited experience with Python and no background in finance or quantitative analysis.
    </candidate_description>

    <thought>
    The candidate has solid experience in software engineering but lacks the necessary background in quantitative research and financial markets. They do not have experience with statistical modeling, machine learning, or algorithmic trading strategies required for the role. Their skills in Java and C++ are not directly applicable to the position that emphasizes Python, R, and financial analysis.
    </thought>
                          
    <score>
    30
    </score>
""")


PROMPT = dedent("""
    You are a helpful assistant that can help me to match candidates with vacancies.
    You are given a vacancy description and a candidate description.
    You need to return a score between 0 and 100 that indicates how well the candidate matches the vacancy.
    You also need to return a detailed description of the match analysis. 
    Think step by step about how well the candidate matches the vacancy and only then return the final score from 0 to 100.
                
    At the start of your response you should first provide your step-by-step reasoning on matching the candidate with the vacancy.
    Your thought process should be enclosed using "<thought>" tag, for example: 
    <thought>
    He has 5 years of experience in Java and Spring Framework, which is more than required. 
    </thought>.
    After that, you must provide the final score. The score should be enclosed using "<score>" tag, for example: 
    <score> 
    65 
    </score>.
    Here are examples of how to structure your response using the requested vanacy and candidate descriptions:

    {few_shot_examples}
    ---
 
    <vacancy_description>
    {vacancy_description}
    </vacancy_description>
    
    <candidate_description>
    {candidate_description}
    </candidate_description>
    <thought>
""")


if __name__ == "__main__":
    print(
        PROMPT.format(
            few_shot_examples=FEW_SHOT_EXAMPLES,
            vacancy_description="kek",
            candidate_description="kekos",
        )
    )
