import os

import neo4j
from neo4j_graphrag.indexes import create_vector_index


def create_candidate_vector_index():
    """Create the vector index for candidate embeddings in Neo4j."""

    neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD", "password")

    print(f"Connecting to Neo4j at {neo4j_uri}")
    driver = neo4j.GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

    try:
        # Check if index already exists
        with driver.session() as session:
            result = session.run("SHOW INDEXES")
            indexes = [record["name"] for record in result]

            if "candidate-vector-index" in indexes:
                print("Index 'candidate-vector-index' already exists")
                return

        # Create the vector index (default dimension for most embedding models is 768 or 1536)
        # Using 1536 for OpenAI embedding compatibility
        print("Creating candidate-vector-index...")
        create_vector_index(
            driver,
            "candidate-vector-index",
            label="Candidate",
            embedding_property="embedding",
            dimensions=1536,  # Adjust based on your embedding model
            similarity_fn="cosine",
        )
        print("Vector index 'candidate-vector-index' created successfully")

    except Exception as e:
        print(f"Error creating vector index: {e}")
    finally:
        driver.close()


if __name__ == "__main__":
    create_candidate_vector_index()
