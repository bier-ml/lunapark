import os
import subprocess
import sys
import time

# Add src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.platform.rag.setup_vector_search import setup_vector_search


def main():
    """Run the setup for vector search and then start the FastAPI application."""
    print("Starting setup process...")

    # Step 1: Set up vector search
    print("\n=======================================")
    print("Setting up vector search capabilities")
    print("=======================================\n")
    setup_vector_search()

    # Wait a moment for any pending operations to complete
    time.sleep(2)

    # Step 2: Start the FastAPI application
    print("\n=======================================")
    print("Starting FastAPI application")
    print("=======================================\n")
    subprocess.run(
        ["uvicorn", "src.service.app:app", "--host", "0.0.0.0", "--port", "8000"]
    )


if __name__ == "__main__":
    main()
