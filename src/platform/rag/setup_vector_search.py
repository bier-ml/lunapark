import os

import neo4j
from neo4j_graphrag.indexes import create_vector_index

from src.platform.rag.graph_rag_predictor import GraphRAGPredictor


def setup_vector_search():
    """
    Set up the vector search capabilities for the system:
    1. Create the candidate-vector-index if it doesn't exist
    2. Ensure all candidates have embeddings
    3. Verify the search is working
    """
    # Initialize connection to Neo4j
    neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD", "password")

    print(f"Connecting to Neo4j at {neo4j_uri}")
    driver = neo4j.GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

    try:
        # Initialize predictor for embedding generation
        predictor = GraphRAGPredictor(
            neo4j_uri=neo4j_uri, neo4j_user=neo4j_user, neo4j_password=neo4j_password
        )

        # 1. Check if index exists, create if not
        with driver.session() as session:
            try:
                result = session.run("SHOW INDEXES")
                indexes = [record["name"] for record in result]

                if "candidate-vector-index" in indexes:
                    print("Index 'candidate-vector-index' already exists")
                else:
                    print("Creating candidate-vector-index...")

                    # First drop if it exists with a different configuration
                    try:
                        session.run("DROP INDEX candidate-vector-index IF EXISTS")
                    except Exception:
                        pass

                    # Create the vector index
                    create_vector_index(
                        driver,
                        "candidate-vector-index",
                        label="Candidate",
                        embedding_property="embedding",
                        dimensions=768,  # Using default for sentence-transformers models
                        similarity_fn="cosine",
                    )
                    print("Vector index 'candidate-vector-index' created successfully")
            except Exception as e:
                print(f"Error checking/creating index: {e}")

        # 2. Check if candidates have embeddings, add if missing
        with driver.session() as session:
            # Get candidates without embeddings
            result = session.run(
                """
                MATCH (c:Candidate)
                WHERE c.embedding IS NULL
                RETURN c.id as id, c.name as name
                """
            )

            candidates_without_embeddings = list(result)
            print(
                f"Found {len(candidates_without_embeddings)} candidates without embeddings"
            )

            # Generate and add embeddings
            for record in candidates_without_embeddings:
                candidate_id = record["id"]
                candidate_name = record["name"]

                # Get candidate details for embedding
                details_result = session.run(
                    """
                    MATCH (c:Candidate {id: $id})
                    OPTIONAL MATCH (c)-[:HAS_SKILL]->(s:Skill)
                    OPTIONAL MATCH (c)-[:LOCATED_IN]->(l:Location)
                    RETURN c, collect(distinct s.name) as skills, collect(distinct l.name) as locations
                    """,
                    id=candidate_id,
                )

                candidate_info = details_result.single()
                if candidate_info:
                    skills = (
                        candidate_info["skills"] if candidate_info["skills"] else []
                    )
                    locations = (
                        candidate_info["locations"]
                        if candidate_info["locations"]
                        else []
                    )

                    # Create text representation for embedding
                    candidate_text = f"{candidate_name} - {', '.join(locations)} - {', '.join(skills)}"

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

        # 3. Test vector search
        print("\nTesting vector search functionality...")
        try:
            vector_results = predictor.score_candidates_vector(
                "software engineer", top_k=2
            )
            print(f"Vector search test: Found {len(vector_results)} candidates")
            if vector_results:
                print(
                    f"Top result: {vector_results[0].get('name', vector_results[0].get('id'))} with score {vector_results[0].get('similarity_score', 0):.2f}"
                )
                print("Vector search is working correctly!")
            else:
                print("Vector search returned no results, but no errors were raised")
        except Exception as e:
            print(f"Vector search test failed: {e}")

    except Exception as e:
        print(f"Error setting up vector search: {e}")
    finally:
        if "driver" in locals():
            driver.close()
        if "predictor" in locals():
            predictor.close()


if __name__ == "__main__":
    setup_vector_search()
