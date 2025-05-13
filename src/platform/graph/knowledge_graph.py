import os
from typing import Dict, List, Optional

from neo4j import GraphDatabase


class KnowledgeGraph:
    """
    Neo4j-based knowledge graph manager for candidates and their attributes.
    Stores entities like candidates, skills, experience, location, etc. as a graph.
    """

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "password",
    ):
        """
        Initialize the knowledge graph with Neo4j connection.

        Args:
            uri: URI for the Neo4j database
            user: Neo4j username
            password: Neo4j password
        """
        # Use environment variables if provided
        self.uri = os.getenv("NEO4J_URI", uri)
        self.user = os.getenv("NEO4J_USER", user)
        self.password = os.getenv("NEO4J_PASSWORD", password)

        # Initialize the Neo4j driver
        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))

        # Initialize the database with constraints and indexes
        self._initialize_db()

    def _initialize_db(self) -> None:
        """Create necessary constraints and indexes for the database."""
        with self.driver.session() as session:
            # Create constraints for uniqueness
            session.run(
                "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Candidate) REQUIRE c.id IS UNIQUE"
            )
            session.run(
                "CREATE CONSTRAINT IF NOT EXISTS FOR (s:Skill) REQUIRE s.name IS UNIQUE"
            )
            session.run(
                "CREATE CONSTRAINT IF NOT EXISTS FOR (l:Location) REQUIRE l.name IS UNIQUE"
            )
            session.run(
                "CREATE CONSTRAINT IF NOT EXISTS FOR (r:Role) REQUIRE r.title IS UNIQUE"
            )
            session.run(
                "CREATE CONSTRAINT IF NOT EXISTS FOR (e:Experience) REQUIRE e.id IS UNIQUE"
            )
            session.run(
                "CREATE CONSTRAINT IF NOT EXISTS FOR (v:Vacancy) REQUIRE v.id IS UNIQUE"
            )

    def close(self) -> None:
        """Close the Neo4j driver connection."""
        self.driver.close()

    def add_candidate(
        self, candidate_id: str, name: str, metadata: Optional[Dict] = None
    ) -> None:
        """
        Add a candidate to the knowledge graph.

        Args:
            candidate_id: Unique identifier for the candidate
            name: Name of the candidate
            metadata: Additional metadata for the candidate (e.g., email, phone, etc.)
        """
        with self.driver.session() as session:
            metadata = metadata or {}
            properties = {"id": candidate_id, "name": name, **metadata}

            # Create a Candidate node with all properties
            query = """
            MERGE (c:Candidate {id: $id})
            SET c += $properties
            RETURN c
            """

            session.run(query, id=candidate_id, properties=properties)

    def add_vacancy(
        self, vacancy_id: str, title: str, metadata: Optional[Dict] = None
    ) -> None:
        """
        Add a vacancy to the knowledge graph.

        Args:
            vacancy_id: Unique identifier for the vacancy
            title: Title of the vacancy
            metadata: Additional metadata for the vacancy
        """
        with self.driver.session() as session:
            metadata = metadata or {}
            properties = {"id": vacancy_id, "title": title, **metadata}

            # Create a Vacancy node with all properties
            query = """
            MERGE (v:Vacancy {id: $id})
            SET v += $properties
            RETURN v
            """

            session.run(query, id=vacancy_id, properties=properties)

    def add_skill_to_candidate(
        self, candidate_id: str, skill_name: str, level: Optional[str] = None
    ) -> None:
        """
        Add a skill to a candidate.

        Args:
            candidate_id: ID of the candidate
            skill_name: Name of the skill
            level: Optional skill level (e.g., 'beginner', 'intermediate', 'expert')
        """
        with self.driver.session() as session:
            query = """
            MATCH (c:Candidate {id: $candidate_id})
            MERGE (s:Skill {name: $skill_name})
            MERGE (c)-[r:HAS_SKILL]->(s)
            SET r.level = $level
            RETURN c, r, s
            """

            session.run(
                query, candidate_id=candidate_id, skill_name=skill_name, level=level
            )

    def add_experience_to_candidate(
        self,
        candidate_id: str,
        company: str,
        role: str,
        years: float,
        metadata: Optional[Dict] = None,
    ) -> str:
        """
        Add work experience to a candidate.

        Args:
            candidate_id: ID of the candidate
            company: Name of the company
            role: Job title or role
            years: Years of experience
            metadata: Additional metadata about the experience

        Returns:
            ID of the created experience node
        """
        with self.driver.session() as session:
            # Generate a unique ID for the experience
            experience_id = f"{candidate_id}_{company}_{role}"

            metadata = metadata or {}
            properties = {
                "id": experience_id,
                "company": company,
                "years": years,
                **metadata,
            }

            query = """
            MATCH (c:Candidate {id: $candidate_id})
            MERGE (r:Role {title: $role})
            CREATE (e:Experience {id: $experience_id})
            SET e += $properties
            MERGE (c)-[:HAS_EXPERIENCE]->(e)
            MERGE (e)-[:AT_ROLE]->(r)
            RETURN e.id
            """

            result = session.run(
                query,
                candidate_id=candidate_id,
                role=role,
                experience_id=experience_id,
                properties=properties,
            )

            return result.single()[0]

    def add_location_to_candidate(
        self, candidate_id: str, location: str, is_current: bool = True
    ) -> None:
        """
        Add a location to a candidate.

        Args:
            candidate_id: ID of the candidate
            location: Name of the location (city, country, etc.)
            is_current: Whether this is the current location
        """
        with self.driver.session() as session:
            query = """
            MATCH (c:Candidate {id: $candidate_id})
            MERGE (l:Location {name: $location})
            MERGE (c)-[r:LOCATED_IN]->(l)
            SET r.is_current = $is_current
            RETURN c, r, l
            """

            session.run(
                query,
                candidate_id=candidate_id,
                location=location,
                is_current=is_current,
            )

    def add_required_skill_to_vacancy(
        self, vacancy_id: str, skill_name: str, importance: Optional[str] = None
    ) -> None:
        """
        Add a required skill to a vacancy.

        Args:
            vacancy_id: ID of the vacancy
            skill_name: Name of the required skill
            importance: Optional importance level (e.g., 'required', 'preferred')
        """
        with self.driver.session() as session:
            query = """
            MATCH (v:Vacancy {id: $vacancy_id})
            MERGE (s:Skill {name: $skill_name})
            MERGE (v)-[r:REQUIRES_SKILL]->(s)
            SET r.importance = $importance
            RETURN v, r, s
            """

            session.run(
                query,
                vacancy_id=vacancy_id,
                skill_name=skill_name,
                importance=importance,
            )

    def add_location_to_vacancy(self, vacancy_id: str, location: str) -> None:
        """
        Add a location to a vacancy.

        Args:
            vacancy_id: ID of the vacancy
            location: Name of the location (city, country, etc.)
        """
        with self.driver.session() as session:
            query = """
            MATCH (v:Vacancy {id: $vacancy_id})
            MERGE (l:Location {name: $location})
            MERGE (v)-[r:LOCATED_IN]->(l)
            RETURN v, r, l
            """

            session.run(query, vacancy_id=vacancy_id, location=location)
