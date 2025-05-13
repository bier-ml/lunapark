import argparse
import json
import os
import sys
from pathlib import Path

from src.platform.rag.graph_rag_predictor import GraphRAGPredictor

# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent.parent.parent)
sys.path.insert(0, project_root)


def validate_url(url):
    """Validate and normalize URL to ensure it has a scheme."""
    if url is None:
        return None

    if not url.startswith(("http://", "https://")):
        return f"http://{url}"
    return url


def main():
    """
    Example script to parse and store resumes using the GraphRAGPredictor.
    """
    parser = argparse.ArgumentParser(description="Parse and store resumes")
    parser.add_argument(
        "resume_file", help="Path to a text file containing the resume to parse"
    )
    parser.add_argument(
        "--output", help="Path to save the parsed JSON output", default=None
    )
    parser.add_argument(
        "--candidate-id", help="Custom candidate ID (optional)", default=None
    )
    parser.add_argument(
        "--neo4j-uri", help="Neo4j URI", default="bolt://localhost:7687"
    )
    parser.add_argument("--neo4j-user", help="Neo4j username", default="neo4j")
    parser.add_argument("--neo4j-password", help="Neo4j password", default="password")
    parser.add_argument(
        "--lm-api-url", help="LM API URL", default=os.getenv("LM_API_BASE_URL")
    )
    parser.add_argument(
        "--lm-api-key", help="LM API key", default=os.getenv("LM_API_KEY")
    )
    parser.add_argument("--lm-model", help="LM model", default=os.getenv("LM_MODEL"))

    args = parser.parse_args()

    # Validate LM API URL
    lm_api_url = validate_url(args.lm_api_url)
    if not lm_api_url:
        lm_api_url = "http://localhost:1234/v1"
        print(f"Using default LM API URL: {lm_api_url}")

    # Create the GraphRAGPredictor
    try:
        predictor = GraphRAGPredictor(
            neo4j_uri=args.neo4j_uri,
            neo4j_user=args.neo4j_user,
            neo4j_password=args.neo4j_password,
            lm_api_base_url=lm_api_url,
            lm_api_key=args.lm_api_key,
            lm_model=args.lm_model,
        )
    except Exception as e:
        print(f"Error initializing GraphRAGPredictor: {e}")
        sys.exit(1)

    # Read the resume file
    try:
        with open(args.resume_file, "r", encoding="utf-8") as f:
            resume_text = f.read()
    except Exception as e:
        print(f"Error reading resume file: {e}")
        sys.exit(1)

    print(f"Parsing resume: {args.resume_file}")
    print(f"Using LM API URL: {lm_api_url}")

    # Parse and store the resume
    try:
        result = predictor.parse_and_store_resume(
            resume_text=resume_text, candidate_id=args.candidate_id
        )
    except Exception as e:
        print(f"Error during resume parsing and storage: {e}")
        sys.exit(1)

    # Print the result
    if result.get("status") == "success":
        print("Resume parsing successful!")
        print(f"Candidate ID: {result.get('candidate_id')}")

        if result.get("storage_status") == "success":
            print("Resume stored in knowledge graph and vector store successfully")
        else:
            print(f"Error storing resume: {result.get('storage_error')}")
    else:
        print(f"Error parsing resume: {result.get('error')}")

    # Save the result to a file if requested
    if args.output:
        try:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)
            print(f"Saved parsed resume to {args.output}")
        except Exception as e:
            print(f"Error saving output file: {e}")

    # Close the predictor to release resources
    try:
        predictor.close()
    except Exception as e:
        print(f"Error closing predictor: {e}")


if __name__ == "__main__":
    main()
