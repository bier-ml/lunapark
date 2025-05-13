import os

import neo4j

# Configure Neo4j connection
neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
neo4j_user = os.getenv("NEO4J_USER", "neo4j")
neo4j_password = os.getenv("NEO4J_PASSWORD", "password")

print(f"Connecting to Neo4j at {neo4j_uri}")
driver = neo4j.GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

try:
    # First check if the index exists
    with driver.session() as session:
        print("Checking if vector index exists...")
        try:
            result = session.run("SHOW INDEXES WHERE name = 'candidate-vector-index'")
            if result.peek() is not None:
                print("Index already exists!")
            else:
                print("Index doesn't exist, creating it...")

                # First check Neo4j version
                version_result = session.run(
                    "CALL dbms.components() YIELD versions RETURN versions[0] as version"
                )
                version_record = (
                    version_result.single() if version_result.peek() else None
                )
                version = version_record["version"] if version_record else "unknown"
                print(f"Neo4j version: {version}")

                # Create vector index - syntax may differ by version
                if version.startswith("5"):
                    # Neo4j 5.x syntax
                    create_index_query = """
                    CREATE VECTOR INDEX `candidate-vector-index` IF NOT EXISTS
                    FOR (c:Candidate) ON (c.embedding)
                    OPTIONS {indexConfig: {
                        `vector.dimensions`: 768,
                        `vector.similarity_function`: 'cosine'
                    }}
                    """
                else:
                    # Neo4j 4.x or other syntax
                    create_index_query = """
                    CALL db.index.vector.createNodeIndex(
                        'candidate-vector-index',
                        'Candidate',
                        'embedding',
                        768,
                        'cosine'
                    )
                    """

                try:
                    session.run(create_index_query)
                    print("Vector index created successfully!")
                except Exception as e:
                    print(f"Error creating vector index: {e}")

                    # Try alternate method
                    try:
                        print("Trying alternate method...")
                        alternate_query = """
                        CALL db.index.vector.createNodeIndex(
                            'candidate-vector-index',
                            'Candidate',
                            'embedding',
                            768,
                            'cosine'
                        )
                        """
                        session.run(alternate_query)
                        print(
                            "Vector index created successfully with alternate method!"
                        )
                    except Exception as alt_e:
                        print(f"Alternate method also failed: {alt_e}")
        except Exception as e:
            print(f"Error checking for index: {e}")

            # Try to create anyway
            try:
                print("Trying to create index anyway...")
                create_query = """
                CREATE VECTOR INDEX `candidate-vector-index` IF NOT EXISTS
                FOR (c:Candidate) ON (c.embedding)
                OPTIONS {indexConfig: {
                    `vector.dimensions`: 768,
                    `vector.similarity_function`: 'cosine'
                }}
                """
                session.run(create_query)
                print("Vector index created!")
            except Exception as create_e:
                print(f"Failed to create index: {create_e}")

    print("Done!")
finally:
    driver.close()
