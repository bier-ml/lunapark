import json
import os
import re
from typing import Dict, List, Optional, Tuple, Union

from neo4j import GraphDatabase
from neo4j_graphrag.embeddings import SentenceTransformerEmbeddings
from neo4j_graphrag.generation import GraphRAG
from neo4j_graphrag.indexes import create_vector_index
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.retrievers import VectorRetriever

from src.platform.base_predictor import BasePredictor
from src.platform.graph.knowledge_graph import KnowledgeGraph
from src.platform.lm_predictor import LMPredictor
from src.platform.rag.resume_parser import ResumeParser


class GraphRAGPredictor(BasePredictor):
    """
    Graph RAG-based predictor that uses Neo4j GraphRAG for job-candidate matching.
    
    IMPROVED SCORING PIPELINE:
    
    1. Enhanced Query Analysis:
       - Extracts seniority level (junior/mid/senior) from job descriptions
       - Identifies priority skills (required, must-have, essential)
       - Better role and experience requirement parsing
    
    2. Context-Aware Vector Scoring:
       - Normalizes vector similarity scores based on query context
       - Penalizes suspiciously high scores (potential duplicates)
       - Adjusts thresholds based on seniority level and technical complexity
    
    3. Improved Graph Scoring:
       - Advanced skill matching with synonyms and fuzzy matching
       - Weighted scoring for priority vs. regular skills
       - Seniority-aware experience evaluation
       - Adaptive component weights based on query characteristics
    
    4. Intelligent Score Combination:
       - Quality assessment for both vector and graph scores
       - Adaptive weighting based on score reliability and data richness
       - Final ranking adjustments for priority skill matches
    
    5. Better Score Interpretation:
       - Scores now better reflect actual candidate relevance
       - Reduced false positives from high but irrelevant similarity
       - Improved ranking for candidates with balanced skill/experience profiles
    """

    def __init__(
        self,
        neo4j_uri: str = "bolt://localhost:7687",
        neo4j_user: str = "neo4j",
        neo4j_password: str = "password",
        lm_api_base_url: Optional[str] = None,
        lm_api_key: Optional[str] = None,
        lm_model: Optional[str] = None,
        embedding_model: str = "sentence-transformers/all-mpnet-base-v2",
    ):
        """
        Initialize the Graph RAG predictor.

        Args:
            neo4j_uri: URI for Neo4j connection
            neo4j_user: Neo4j username
            neo4j_password: Neo4j password
            lm_api_base_url: Base URL for LM API
            lm_api_key: API key for LM
            lm_model: Model identifier for LM
            embedding_model: Model to use for embeddings
        """
        super().__init__()

        # Use environment variables if provided
        self.neo4j_uri = os.environ.get("NEO4J_URI", neo4j_uri)
        self.neo4j_user = os.environ.get("NEO4J_USER", neo4j_user)
        self.neo4j_password = os.environ.get("NEO4J_PASSWORD", neo4j_password)

        # Initialize Neo4j driver
        self.driver = GraphDatabase.driver(
            self.neo4j_uri, auth=(self.neo4j_user, self.neo4j_password)
        )

        # Initialize the embedding model
        self.embedder = SentenceTransformerEmbeddings(model=embedding_model)

        # Set default LM API URL if not provided
        if lm_api_base_url is None:
            lm_api_base_url = os.environ.get(
                "LM_API_BASE_URL", "http://localhost:1234/v1"
            )

        # Ensure the URL has a scheme
        if lm_api_base_url and not lm_api_base_url.startswith(("http://", "https://")):
            lm_api_base_url = f"http://{lm_api_base_url}"

        # Initialize LLM for GraphRAG (using standard OpenAI format)
        self.llm = OpenAILLM(
            model_name=lm_model or os.environ.get("LM_MODEL", "local-model"),
            api_key=lm_api_key or os.environ.get("LM_API_KEY", "not-needed"),
            base_url=lm_api_base_url,
        )

        # Initialize LM predictor for enhanced scoring and explanation
        self.lm_predictor = LMPredictor(
            api_base_url=lm_api_base_url,
            api_key=lm_api_key or os.environ.get("LM_API_KEY", "not-needed"),
            model=lm_model or os.environ.get("LM_MODEL", "local-model"),
        )

        # Initialize hybrid retriever for candidates
        self.candidate_retriever = VectorRetriever(
            self.driver,
            index_name="candidate-vector-index",
            embedder=self.embedder,
        )

        # Initialize the knowledge graph for graph-based retrieval
        self.knowledge_graph = KnowledgeGraph(
            uri=self.neo4j_uri, user=self.neo4j_user, password=self.neo4j_password
        )

        # Initialize resume parser
        self.resume_parser = ResumeParser(
            lm_api_base_url=lm_api_base_url,
            lm_api_key=lm_api_key or os.environ.get("LM_API_KEY", "not-needed"),
            lm_model=lm_model or os.environ.get("LM_MODEL", "local-model"),
        )

    def get_available_models(self) -> List[str]:
        """Get list of available models."""
        return ["graph-rag-neo4j"]

    def extract_query_components(self, query: str) -> Dict:
        """
        Extract structured components from a job query or description.

        Args:
            query: Job query or description text

        Returns:
            Dictionary containing extracted role, skills, location, etc.
        """
        # Handle very short queries
        if len(query) < 10:
            # Simple handling - assume the whole thing is the role
            return {"role": query, "skills": [], "location": "", "years": 0, "seniority": "mid", "priority_skills": []}

        try:
            # First try a direct skills extraction by keyword matching
            skills = []
            priority_skills = []  # Skills that are explicitly emphasized

            # Common programming languages and technologies to explicitly check for
            common_skills = [
                "python", "java", "javascript", "typescript", "c++", "c#", "ruby", "go", "rust",
                "sql", "postgresql", "mysql", "mongodb", "redis", "react", "angular", "vue",
                "node.js", "django", "flask", "spring", "kubernetes", "docker", "aws", "azure", "gcp",
                "machine learning", "ai", "pytorch", "tensorflow", "figma", "sketch", "photoshop",
                "illustrator", "css", "html", "git", "devops", "cicd", "jenkins", "terraform"
            ]

            # Check for these skills in the query
            query_lower = query.lower()
            for skill in common_skills:
                if skill in query_lower:
                    skills.append(skill)
                    # Check if skill is emphasized (required, must have, essential, etc.)
                    skill_context = self._get_skill_context(query_lower, skill)
                    if any(keyword in skill_context for keyword in ["required", "must", "essential", "critical", "mandatory"]):
                        priority_skills.append(skill)

            # Extract role and seniority level
            role = ""
            seniority = "mid"  # default
            
            # Enhanced role patterns with seniority detection
            role_patterns = [
                r"(senior|sr\.?|lead|principal|staff|chief|head of|director of)?\s*(software|frontend|backend|fullstack|full-stack|data|ml|ai|devops|cloud|mobile|web|ios|android|system)?\s*(developer|engineer|architect|designer|scientist|analyst)"
            ]

            for pattern in role_patterns:
                matches = re.findall(pattern, query_lower)
                if matches:
                    # Take the first match and join its components
                    role_parts = [part for part in matches[0] if part]
                    role = " ".join(role_parts)
                    
                    # Determine seniority level
                    seniority_indicators = matches[0][0] if matches[0] else ""
                    if any(level in seniority_indicators for level in ["senior", "sr", "lead", "principal", "staff", "chief", "head", "director"]):
                        if any(level in seniority_indicators for level in ["principal", "staff", "chief", "head", "director"]):
                            seniority = "senior"
                        else:
                            seniority = "senior"
                    elif "junior" in seniority_indicators or "jr" in seniority_indicators:
                        seniority = "junior"
                    break

            # Extract years of experience if mentioned
            years = 0
            years_patterns = [
                r"(\d+)\+?\s*years?\s*(of)?\s*experience",
                r"experience\s*:\s*(\d+)\+?\s*years?",
                r"(\d+)\+?\s*years?\s*(of)?\s*work",
            ]

            for pattern in years_patterns:
                matches = re.findall(pattern, query_lower)
                if matches:
                    try:
                        # Extract the first number found
                        years = float(matches[0][0])
                        # Adjust seniority based on years if not already determined
                        if seniority == "mid":
                            if years >= 7:
                                seniority = "senior"
                            elif years <= 2:
                                seniority = "junior"
                        break
                    except (ValueError, IndexError):
                        pass

            # Extract location if mentioned
            location = ""
            location_patterns = [
                r"in\s+([A-Z][a-z]+(\s+[A-Z][a-z]+)*)",
                r"location\s*:\s*([A-Z][a-z]+(\s+[A-Z][a-z]+)*)",
                r"based\s+in\s+([A-Z][a-z]+(\s+[A-Z][a-z]+)*)",
            ]

            for pattern in location_patterns:
                matches = re.findall(pattern, query)
                if matches:
                    location = matches[0][0]
                    break

            # Ensure we at least have some minimal data
            if not role and len(query.split()) < 5:
                role = query  # Use the whole query if it's short and no role was detected

            # Combine results
            components = {
                "role": role,
                "skills": skills,
                "priority_skills": priority_skills,
                "location": location,
                "years": years,
                "seniority": seniority,
            }

            print(f"Extracted components: {components}")
            return components

        except Exception as e:
            print(f"Error extracting query components: {str(e)}")
            # Return empty values on error
            return {"role": "", "skills": [], "priority_skills": [], "location": "", "years": 0, "seniority": "mid"}

    def _get_skill_context(self, text: str, skill: str) -> str:
        """Extract context around a skill mention to determine its importance."""
        skill_index = text.find(skill)
        if skill_index == -1:
            return ""
        
        # Get 50 characters before and after the skill mention
        start = max(0, skill_index - 50)
        end = min(len(text), skill_index + len(skill) + 50)
        return text[start:end]

    def _normalize_vector_score(self, score: float, query_context: Dict) -> float:
        """
        Normalize vector similarity score based on query context and score quality.
        
        Args:
            score: Raw vector similarity score (0-1)
            query_context: Query components for context-aware normalization
            
        Returns:
            Normalized score (0-1)
        """
        if score <= 0:
            return 0.0
            
        # Apply context-aware normalization
        normalized_score = score
        
        # Penalize very high scores that might be false positives
        if score > 0.95:
            # Very high similarity might indicate duplicate or near-duplicate content
            # rather than genuine job relevance
            normalized_score = score * 0.9
            
        # Boost scores for queries with specific technical requirements
        if query_context.get("skills") and len(query_context["skills"]) > 3:
            # Complex technical queries should have higher confidence in vector matching
            normalized_score = min(1.0, normalized_score * 1.1)
            
        # Apply seniority-based adjustments
        seniority = query_context.get("seniority", "mid")
        if seniority == "senior" and score < 0.7:
            # Senior roles require higher similarity thresholds
            normalized_score = score * 0.8
        elif seniority == "junior" and score > 0.6:
            # Junior roles can be more flexible
            normalized_score = min(1.0, score * 1.2)
            
        return max(0.0, min(1.0, normalized_score))

    def _calculate_skill_relevance_score(self, candidate_skills: List[str], query_components: Dict) -> Tuple[float, List[str], Dict]:
        """
        Calculate skill relevance score with improved matching and weighting.
        
        Args:
            candidate_skills: List of candidate's skills
            query_components: Extracted query components
            
        Returns:
            Tuple of (skill_score, matched_skills, skill_details)
        """
        query_skills = query_components.get("skills", [])
        priority_skills = query_components.get("priority_skills", [])
        query_role = query_components.get("role", "").lower()
        seniority = query_components.get("seniority", "mid")
        
        if not query_skills and not query_role:
            return 0.0, [], {}
            
        matched_skills = []
        priority_matches = []
        skill_details = {}
        
        # Enhanced skill matching with fuzzy matching and synonyms
        skill_synonyms = {
            "javascript": ["js", "ecmascript", "node.js", "nodejs"],
            "python": ["py", "python3"],
            "sql": ["database", "postgresql", "mysql", "sqlite", "oracle"],
            "react": ["reactjs", "react.js"],
            "angular": ["angularjs", "angular.js"],
            "machine learning": ["ml", "ai", "artificial intelligence", "deep learning"],
            "devops": ["ci/cd", "cicd", "deployment", "infrastructure"],
        }
        
        for candidate_skill in candidate_skills:
            if not isinstance(candidate_skill, str):
                continue
                
            candidate_skill_lower = candidate_skill.lower()
            best_match_score = 0.0
            matched_query_skill = None
            is_priority = False
            
            # Direct matching against query skills
            for query_skill in query_skills:
                if not isinstance(query_skill, str):
                    continue
                    
                query_skill_lower = query_skill.lower()
                match_score = 0.0
                
                # Exact match
                if query_skill_lower == candidate_skill_lower:
                    match_score = 1.0
                # Substring match
                elif query_skill_lower in candidate_skill_lower or candidate_skill_lower in query_skill_lower:
                    match_score = 0.8
                # Synonym match
                elif query_skill_lower in skill_synonyms:
                    if any(syn in candidate_skill_lower for syn in skill_synonyms[query_skill_lower]):
                        match_score = 0.9
                # Check reverse synonyms
                else:
                    for main_skill, synonyms in skill_synonyms.items():
                        if candidate_skill_lower in synonyms and main_skill == query_skill_lower:
                            match_score = 0.9
                            break
                
                if match_score > best_match_score:
                    best_match_score = match_score
                    matched_query_skill = query_skill
                    is_priority = query_skill in priority_skills
            
            # Role-based matching if no specific skills in query
            if best_match_score == 0.0 and query_role:
                role_keywords = query_role.split()
                for keyword in role_keywords:
                    if keyword in candidate_skill_lower:
                        best_match_score = 0.6  # Lower score for role-based matches
                        matched_query_skill = f"role:{keyword}"
                        break
            
            if best_match_score > 0.5:  # Threshold for considering a match
                matched_skills.append(candidate_skill)
                if is_priority:
                    priority_matches.append(candidate_skill)
                    
                skill_details[candidate_skill] = {
                    "match_score": best_match_score,
                    "matched_query_skill": matched_query_skill,
                    "is_priority": is_priority
                }
        
        # Calculate overall skill score
        if not query_skills:
            # Role-based scoring
            skill_score = min(len(matched_skills) / 3, 0.8) if matched_skills else 0.0
        else:
            # Calculate weighted score based on matches
            total_weight = 0.0
            achieved_weight = 0.0
            
            for query_skill in query_skills:
                weight = 2.0 if query_skill in priority_skills else 1.0
                total_weight += weight
                
                # Find best matching candidate skill for this query skill
                best_candidate_match = 0.0
                for candidate_skill, details in skill_details.items():
                    if details["matched_query_skill"] == query_skill:
                        best_candidate_match = max(best_candidate_match, details["match_score"])
                
                achieved_weight += weight * best_candidate_match
            
            skill_score = achieved_weight / total_weight if total_weight > 0 else 0.0
            
            # Apply seniority adjustments
            if seniority == "senior":
                # Senior roles require higher skill coverage
                if skill_score < 0.7:
                    skill_score *= 0.8
            elif seniority == "junior":
                # Junior roles can be more flexible
                skill_score = min(1.0, skill_score * 1.2)
        
        return skill_score, matched_skills, skill_details

    def score_candidates_vector(self, query: str, top_k: int = 10) -> List[Dict]:
        """
        Score candidates based on vector similarity.

        Args:
            query: Vacancy query text
            top_k: Number of top candidates to return

        Returns:
            List of candidates with vector similarity scores
        """
        try:
            print(f"Performing vector search for query: '{query}'")
            
            # Extract query components for context-aware normalization
            query_components = self.extract_query_components(query)

            # Create query embedding
            query_embedding = self.embedder.embed_query(query)

            # Use direct Cypher query for vector search
            with self.driver.session() as session:
                # First check if the index exists
                try:
                    index_check = session.run(
                        "SHOW INDEXES WHERE name = 'candidate-vector-index'"
                    )
                    index_exists = index_check.peek() is not None
                    if not index_exists:
                        raise ValueError(
                            "No index with name candidate-vector-index found"
                        )
                except Exception as e:
                    # Handle older Neo4j versions that don't support SHOW INDEXES
                    print(f"Error checking for index existence: {e}")

                    # Try to check if candidates have embeddings at least
                    embedding_check = session.run(
                        "MATCH (c:Candidate) WHERE c.embedding IS NOT NULL RETURN count(c) AS count"
                    )
                    record = (
                        embedding_check.single() if embedding_check.peek() else None
                    )
                    has_embeddings = record and record.get("count", 0) > 0

                    if not has_embeddings:
                        print("No candidates with embeddings found")
                        # Fall back to basic candidate retrieval with dummy scores
                        return self._get_candidates_fallback(top_k)

                # Check if vectorSimilarity function exists
                try:
                    check_query = """
                    RETURN exists(function('vectorSimilarity', [0.1, 0.2], [0.1, 0.2])) as has_vector_similarity
                    """
                    check_result = session.run(check_query)
                    record = check_result.single() if check_result.peek() else None
                    has_vector_similarity = record and record.get(
                        "has_vector_similarity", False
                    )
                except Exception:
                    has_vector_similarity = False

                if has_vector_similarity:
                    print("Using vectorSimilarity function for candidate search")
                    try:
                        # Use vectorSimilarity() in Cypher directly
                        cypher_query = """
                        MATCH (c:Candidate)
                        WHERE c.embedding IS NOT NULL
                        WITH c, vectorSimilarity(c.embedding, $query_embedding) AS raw_score
                        RETURN c.id AS id, c.name AS name, raw_score
                        ORDER BY raw_score DESC
                        LIMIT $limit
                        """

                        result = session.run(
                            cypher_query, query_embedding=query_embedding, limit=top_k * 2  # Get more for filtering
                        )

                        candidates = []
                        for record in result:
                            raw_score = record["raw_score"]
                            # Apply context-aware normalization
                            normalized_score = self._normalize_vector_score(raw_score, query_components)
                            
                            # Only include candidates with meaningful scores
                            if normalized_score > 0.1:  # Minimum threshold
                                candidates.append(
                                    {
                                        "id": record["id"],
                                        "name": record["name"],
                                        "similarity_score": normalized_score,
                                        "raw_similarity_score": raw_score,
                                        "metadata": {"name": record["name"]},
                                    }
                                )

                        # Re-sort by normalized score and limit to top_k
                        candidates.sort(key=lambda x: x["similarity_score"], reverse=True)
                        candidates = candidates[:top_k]

                        if candidates:
                            print(
                                f"Found {len(candidates)} candidates using vectorSimilarity (normalized)"
                            )
                            return candidates
                    except Exception as cypher_err:
                        print(
                            f"Vector search with vectorSimilarity failed: {cypher_err}"
                        )

                # Fall back to getting all candidates with descending scores if vector search is not available
                return self._get_candidates_fallback(top_k)

        except Exception as e:
            print(f"Error in vector search: {e}")
            import traceback

            traceback.print_exc()
            return []  # Return empty list on error

    def _get_candidates_fallback(self, limit: int) -> List[Dict]:
        """Fallback method to get candidates when vector search is not available.

        Args:
            limit: Maximum number of candidates to return

        Returns:
            List of candidates with dummy similarity scores
        """
        print("Using basic candidate retrieval with manual scoring")
        try:
            with self.driver.session() as session:
                basic_query = """
                MATCH (c:Candidate)
                RETURN c.id AS id, c.name AS name
                LIMIT $limit
                """

                result = session.run(basic_query, limit=limit)
                candidates = []
                for record in result:
                    # Decrease score gradually for each position to create a ranking
                    position_score = 1.0 - (
                        len(candidates) * 0.1
                    )  # Score decreases by 0.1 for each position
                    similarity_score = max(0.5, position_score)  # Minimum score of 0.5

                    candidates.append(
                        {
                            "id": record["id"],
                            "name": record["name"],
                            "similarity_score": similarity_score,
                            "metadata": {"name": record["name"]},
                        }
                    )

                print(f"Found {len(candidates)} candidates via basic retrieval")
                return candidates
        except Exception as e:
            print(f"Error in fallback candidate retrieval: {e}")
            return []

    def score_candidates_graph(self, query: str, top_k: int = 10) -> List[Dict]:
        """
        Score candidates based on graph features.

        Args:
            query: Vacancy query text
            top_k: Number of top candidates to return

        Returns:
            List of candidates with graph-based scores
        """
        print(f"Performing graph search for query: '{query}'")
        try:
            # Parse query to extract components (skills, etc.)
            query_components = self.extract_query_components(query)
            query_skills = query_components.get("skills", [])
            priority_skills = query_components.get("priority_skills", [])
            query_role = query_components.get("role", "")
            query_location = query_components.get("location", "")
            query_years = query_components.get("years", 0)
            seniority = query_components.get("seniority", "mid")

            # Ensure query_skills is a list, not a float (fixing type error)
            if not isinstance(query_skills, list):
                query_skills = []

            print(
                f"Extracted query components: role='{query_role}', location='{query_location}', years={query_years}, seniority='{seniority}', skills={query_skills}, priority_skills={priority_skills}"
            )

            candidates = []

            # Get all candidates directly using Neo4j
            with self.driver.session() as session:
                candidate_query = """
                MATCH (c:Candidate)
                RETURN c.id as id, c.name as name
                LIMIT $limit
                """
                candidate_result = session.run(
                    candidate_query, limit=top_k * 3
                )  # Get more candidates than needed for better filtering

                # Process each candidate
                for record in candidate_result:
                    candidate_id = record["id"]
                    candidate_name = record["name"]

                    print(
                        f"Evaluating candidate: {candidate_name} (ID: {candidate_id})"
                    )

                    # Get all skills for this candidate
                    skills_query = """
                    MATCH (c:Candidate {id: $candidate_id})-[:HAS_SKILL]->(s:Skill)
                    RETURN s.name as skill_name
                    """
                    skills_result = session.run(skills_query, candidate_id=candidate_id)
                    candidate_skills = [
                        record["skill_name"] for record in skills_result
                    ]

                    print(
                        f"  - Found {len(candidate_skills)} skills: {', '.join(candidate_skills[:5])}{'...' if len(candidate_skills) > 5 else ''}"
                    )

                    # Calculate skill relevance using improved method
                    skill_score, matched_skills, skill_details = self._calculate_skill_relevance_score(
                        candidate_skills, query_components
                    )

                    print(
                        f"  - Skill score: {skill_score:.2f}, matched {len(matched_skills)} skills: {', '.join(matched_skills[:5])}{'...' if len(matched_skills) > 5 else ''}"
                    )

                    # Get experience for this candidate
                    exp_query = """
                    MATCH (c:Candidate {id: $candidate_id})-[:HAS_EXPERIENCE]->(e)
                    RETURN e.company as company, e.title as title, e.years as years
                    """
                    exp_result = session.run(exp_query, candidate_id=candidate_id)
                    experiences = []
                    total_years = 0

                    for exp_record in exp_result:
                        company = exp_record.get("company", "")
                        title = exp_record.get("title", "")
                        years = exp_record.get("years", 0)

                        if isinstance(years, (int, float)):
                            total_years += years

                        experiences.append(
                            {"company": company, "title": title, "years": years}
                        )

                    print(
                        f"  - Found {len(experiences)} experiences, total {total_years} years"
                    )

                    # Calculate experience score with improved logic
                    exp_score = self._calculate_experience_score(
                        total_years, experiences, query_components
                    )

                    # Get location for this candidate
                    loc_query = """
                    MATCH (c:Candidate {id: $candidate_id})-[:LOCATED_IN]->(l)
                    RETURN l.name as location
                    """
                    loc_result = session.run(loc_query, candidate_id=candidate_id)
                    locations = []

                    for loc_record in loc_result:
                        location = loc_record.get("location", "")
                        locations.append(location)

                    print(f"  - Locations: {', '.join(locations)}")

                    # Calculate location score
                    loc_score = self._calculate_location_score(locations, query_components)

                    # Calculate adaptive weights based on query characteristics
                    weights = self._calculate_adaptive_weights(query_components)

                    # Combine scores with adaptive weights
                    graph_score = (
                        (weights["skill"] * skill_score) + 
                        (weights["experience"] * exp_score) + 
                        (weights["location"] * loc_score)
                    )

                    # Apply minimum score threshold
                    if len(matched_skills) > 0 or total_years > 0:
                        graph_score = max(graph_score, 0.05)  # Lower minimum threshold

                    print(
                        f"  - Final graph score: {graph_score:.2f} (skill: {skill_score:.2f}*{weights['skill']:.1f}, exp: {exp_score:.2f}*{weights['experience']:.1f}, loc: {loc_score:.2f}*{weights['location']:.1f})"
                    )

                    # Build graph matches for explanation
                    graph_matches = []
                    
                    # Add skill matches with details
                    for skill, details in skill_details.items():
                        graph_matches.append({
                            "type": "skill",
                            "skill": {"name": skill},
                            "relevance": details["match_score"],
                            "is_priority": details.get("is_priority", False),
                            "matched_query_skill": details.get("matched_query_skill", "")
                        })
                    
                    # Add experience matches
                    for exp in experiences:
                        graph_matches.append({
                            "type": "experience",
                            "experience": {
                                "company": exp["company"],
                                "role": exp["title"],
                                "years": exp["years"],
                            }
                        })
                    
                    # Add location matches
                    for location in locations:
                        graph_matches.append({
                            "type": "location",
                            "location": {"name": location}
                        })

                    candidates.append(
                        {
                            "id": candidate_id,
                            "name": candidate_name,
                            "graph_score": graph_score,
                            "graph_matches": graph_matches,
                            "matched_skills": matched_skills,  # Only relevant/matched skills
                            "all_skills": candidate_skills,  # All skills the candidate has
                            "skill_details": skill_details,  # Detailed skill matching info
                            "relevant_skill_count": len(matched_skills),
                            "total_skill_count": len(candidate_skills),
                            "total_years_experience": total_years,
                            "locations": locations,
                            "component_scores": {
                                "skill_score": skill_score,
                                "experience_score": exp_score,
                                "location_score": loc_score
                            },
                            "weights": weights
                        }
                    )

            # Sort by graph score and limit to top_k
            candidates.sort(key=lambda x: x["graph_score"], reverse=True)
            candidates = candidates[:top_k]

            print(f"Graph search found {len(candidates)} candidates with scores")
            return candidates

        except Exception as e:
            print(f"Error in graph search: {str(e)}")
            import traceback

            traceback.print_exc()
            return []  # Return empty list on error

    def _calculate_experience_score(self, total_years: float, experiences: List[Dict], query_components: Dict) -> float:
        """Calculate experience score with improved logic."""
        query_years = query_components.get("years", 0)
        seniority = query_components.get("seniority", "mid")
        query_role = query_components.get("role", "").lower()
        
        if total_years <= 0:
            return 0.0
        
        # Base score from years of experience
        exp_score = 0.0
        
        if query_years > 0:
            # Specific years requirement
            if total_years >= query_years:
                exp_score = 0.4  # Full score for meeting requirement
                # Bonus for exceeding requirement (but with diminishing returns)
                excess_years = total_years - query_years
                bonus = min(0.1, excess_years * 0.02)  # Max 0.1 bonus
                exp_score += bonus
            else:
                # Partial credit based on how close to requirement
                exp_score = 0.4 * (total_years / query_years)
        else:
            # No specific years requirement, score based on seniority expectations
            if seniority == "junior":
                # Junior roles: 0-3 years is ideal
                if total_years <= 3:
                    exp_score = 0.3
                elif total_years <= 5:
                    exp_score = 0.25  # Slightly overqualified
                else:
                    exp_score = 0.2  # Significantly overqualified
            elif seniority == "senior":
                # Senior roles: 5+ years expected
                if total_years >= 5:
                    exp_score = 0.4
                    # Bonus for very senior candidates
                    if total_years >= 10:
                        exp_score = min(0.5, 0.4 + (total_years - 10) * 0.01)
                else:
                    exp_score = 0.4 * (total_years / 5)  # Partial credit
            else:
                # Mid-level roles: 2-7 years is good
                if 2 <= total_years <= 7:
                    exp_score = 0.35
                elif total_years < 2:
                    exp_score = 0.35 * (total_years / 2)
                else:
                    exp_score = 0.3  # Overqualified but still good
        
        # Bonus for relevant role experience
        if query_role and experiences:
            role_keywords = query_role.split()
            for exp in experiences:
                exp_title = exp.get("title", "").lower()
                if any(keyword in exp_title for keyword in role_keywords):
                    exp_score = min(1.0, exp_score * 1.2)  # 20% bonus for relevant role
                    break
        
        return min(1.0, exp_score)

    def _calculate_location_score(self, locations: List[str], query_components: Dict) -> float:
        """Calculate location score."""
        query_location = query_components.get("location", "")
        
        if not query_location or not locations:
            return 0.0
        
        query_loc_lower = query_location.lower()
        for location in locations:
            if (
                location
                and isinstance(location, str)
                and (
                    query_loc_lower in location.lower()
                    or location.lower() in query_loc_lower
                )
            ):
                return 1.0  # Perfect location match
        
        return 0.0

    def _calculate_adaptive_weights(self, query_components: Dict) -> Dict[str, float]:
        """Calculate adaptive weights based on query characteristics."""
        skills = query_components.get("skills", [])
        priority_skills = query_components.get("priority_skills", [])
        years = query_components.get("years", 0)
        location = query_components.get("location", "")
        seniority = query_components.get("seniority", "mid")
        
        # Base weights
        skill_weight = 0.6
        exp_weight = 0.3
        loc_weight = 0.1
        
        # Adjust based on query characteristics
        if len(skills) > 5 or len(priority_skills) > 2:
            # Skill-heavy queries - increase skill weight
            skill_weight = 0.75
            exp_weight = 0.2
            loc_weight = 0.05
        elif years > 0 or seniority in ["senior", "junior"]:
            # Experience-focused queries
            skill_weight = 0.5
            exp_weight = 0.4
            loc_weight = 0.1
        elif location:
            # Location-specific queries
            skill_weight = 0.5
            exp_weight = 0.25
            loc_weight = 0.25
        
        # Seniority adjustments
        if seniority == "senior":
            # Senior roles care more about experience
            exp_weight = min(0.5, exp_weight * 1.3)
            skill_weight = max(0.4, 1.0 - exp_weight - loc_weight)
        elif seniority == "junior":
            # Junior roles care more about skills and potential
            skill_weight = min(0.8, skill_weight * 1.2)
            exp_weight = max(0.1, 1.0 - skill_weight - loc_weight)
        
        # Normalize to ensure weights sum to 1.0
        total = skill_weight + exp_weight + loc_weight
        return {
            "skill": skill_weight / total,
            "experience": exp_weight / total,
            "location": loc_weight / total
        }

    def combine_scores(
        self, vector_results: List[Dict], graph_results: List[Dict]
    ) -> List[Dict]:
        """
        Combine vector and graph scores using intelligent weighted averaging.

        Args:
            vector_results: Results from vector similarity search
            graph_results: Results from graph-based search

        Returns:
            Combined results with overall scores
        """
        print(
            f"Combining scores from {len(vector_results)} vector results and {len(graph_results)} graph results"
        )

        # Create a map for vector results
        vector_map = (
            {result["id"]: result for result in vector_results}
            if vector_results
            else {}
        )

        # Create a map for graph results
        graph_map = (
            {result["id"]: result for result in graph_results} if graph_results else {}
        )

        # Combine results
        combined_results = []

        # Process all candidates from either result set
        all_candidate_ids = set(vector_map.keys()) | set(graph_map.keys())
        print(f"Found {len(all_candidate_ids)} unique candidates to combine")

        for cid in all_candidate_ids:
            vector_item = vector_map.get(cid, {})
            graph_item = graph_map.get(cid, {})

            vector_score = vector_item.get("similarity_score", 0)
            graph_score = graph_item.get("graph_score", 0)
            raw_vector_score = vector_item.get("raw_similarity_score", vector_score)

            # Assess score quality and reliability
            vector_quality = self._assess_vector_score_quality(vector_score, raw_vector_score, vector_item)
            graph_quality = self._assess_graph_score_quality(graph_score, graph_item)

            # Calculate adaptive weights based on score quality and availability
            weights = self._calculate_combination_weights(
                vector_score, graph_score, vector_quality, graph_quality, graph_item
            )

            # Apply weighted average with quality adjustments
            combined_score = (weights["vector"] * vector_score) + (weights["graph"] * graph_score)

            # Apply quality-based adjustments
            quality_factor = (vector_quality * weights["vector"]) + (graph_quality * weights["graph"])
            combined_score = combined_score * quality_factor

            # Get candidate details from either result
            candidate_details = vector_map.get(cid, graph_map.get(cid, {}))
            name = (
                candidate_details.get("name", "")
                or vector_item.get("metadata", {}).get("name", "")
                or graph_item.get("name", "")
                or "Unknown Candidate"
            )

            # Get matched skills from graph results
            matched_skills = graph_item.get("matched_skills", [])
            all_skills = graph_item.get("all_skills", [])
            skill_details = graph_item.get("skill_details", {})

            # Get additional data from graph results
            total_years_experience = graph_item.get("total_years_experience", 0)
            locations = graph_item.get("locations", [])
            component_scores = graph_item.get("component_scores", {})
            adaptive_weights = graph_item.get("weights", {})

            # Construct detailed result
            result = {
                "id": cid,
                "name": name,
                "vector_score": vector_score,
                "raw_vector_score": raw_vector_score,
                "graph_score": graph_score,
                "combined_score": combined_score,
                "vector_quality": vector_quality,
                "graph_quality": graph_quality,
                "combination_weights": weights,
                "quality_factor": quality_factor,
                "graph_matches": graph_item.get("graph_matches", []),
                "matched_skills": matched_skills,
                "skill_details": skill_details,
            }

            # Add additional data if available
            if all_skills:
                result["all_skills"] = all_skills

            if "relevant_skill_count" in graph_item:
                result["relevant_skill_count"] = graph_item["relevant_skill_count"]

            if "total_skill_count" in graph_item:
                result["total_skill_count"] = graph_item["total_skill_count"]

            if total_years_experience:
                result["total_years_experience"] = total_years_experience

            if locations:
                result["locations"] = locations

            if component_scores:
                result["component_scores"] = component_scores

            if adaptive_weights:
                result["adaptive_weights"] = adaptive_weights

            if "match_count" in graph_item:
                result["match_count"] = graph_item["match_count"]

            combined_results.append(result)

        # Sort by combined score
        combined_results.sort(key=lambda x: x["combined_score"], reverse=True)

        # Apply final ranking adjustments
        combined_results = self._apply_ranking_adjustments(combined_results)

        # Print summary
        top_score = combined_results[0]["combined_score"] if combined_results else 0
        print(f"Combined {len(combined_results)} results, top score: {top_score:.3f}")

        return combined_results

    def _assess_vector_score_quality(self, normalized_score: float, raw_score: float, vector_item: Dict) -> float:
        """Assess the quality/reliability of a vector similarity score."""
        if normalized_score <= 0:
            return 0.0
        
        quality = 1.0
        
        # Penalize very high raw scores that might be false positives
        if raw_score > 0.98:
            quality *= 0.7  # Likely duplicate or near-duplicate
        elif raw_score > 0.95:
            quality *= 0.85  # Possibly too similar
        
        # Penalize very low scores
        if normalized_score < 0.3:
            quality *= 0.6  # Low confidence in match
        
        # Boost moderate-high scores that are likely genuine matches
        if 0.6 <= normalized_score <= 0.9:
            quality *= 1.1  # Sweet spot for genuine relevance
        
        return min(1.0, quality)

    def _assess_graph_score_quality(self, graph_score: float, graph_item: Dict) -> float:
        """Assess the quality/reliability of a graph-based score."""
        if graph_score <= 0:
            return 0.0
        
        quality = 1.0
        
        # Check if we have meaningful skill matches
        matched_skills = graph_item.get("matched_skills", [])
        skill_details = graph_item.get("skill_details", {})
        component_scores = graph_item.get("component_scores", {})
        
        # Quality based on skill match quality
        if matched_skills and skill_details:
            avg_skill_match_score = sum(
                details.get("match_score", 0) for details in skill_details.values()
            ) / len(skill_details)
            
            if avg_skill_match_score > 0.8:
                quality *= 1.2  # High-quality skill matches
            elif avg_skill_match_score < 0.6:
                quality *= 0.8  # Lower-quality skill matches
        
        # Quality based on component score distribution
        skill_score = component_scores.get("skill_score", 0)
        exp_score = component_scores.get("experience_score", 0)
        
        # Prefer candidates with balanced scores rather than single high component
        if skill_score > 0 and exp_score > 0:
            quality *= 1.1  # Balanced candidate
        elif skill_score > 0.8 and exp_score == 0:
            quality *= 0.9  # Skills-only match
        elif exp_score > 0.5 and skill_score == 0:
            quality *= 0.7  # Experience-only match (less reliable)
        
        # Penalize artificially inflated scores
        if graph_score > 0.9 and len(matched_skills) < 2:
            quality *= 0.6  # High score with few matches is suspicious
        
        return min(1.0, quality)

    def _calculate_combination_weights(
        self, vector_score: float, graph_score: float, 
        vector_quality: float, graph_quality: float, graph_item: Dict
    ) -> Dict[str, float]:
        """Calculate adaptive weights for combining vector and graph scores."""
        
        # Base weights
        vector_weight = 0.4
        graph_weight = 0.6
        
        # Adjust based on score availability and quality
        has_vector = vector_score > 0
        has_graph = graph_score > 0
        
        if not has_vector and not has_graph:
            return {"vector": 0.0, "graph": 0.0}
        elif not has_vector:
            return {"vector": 0.0, "graph": 1.0}
        elif not has_graph:
            return {"vector": 1.0, "graph": 0.0}
        
        # Adjust weights based on quality
        quality_ratio = vector_quality / (vector_quality + graph_quality) if (vector_quality + graph_quality) > 0 else 0.5
        
        # Adjust based on graph match richness
        matched_skills = graph_item.get("matched_skills", [])
        skill_details = graph_item.get("skill_details", {})
        
        if len(matched_skills) >= 3 and skill_details:
            # Rich graph data - trust graph more
            graph_weight = min(0.8, graph_weight * 1.3)
        elif len(matched_skills) <= 1:
            # Sparse graph data - trust vector more
            vector_weight = min(0.8, vector_weight * 1.3)
        
        # Adjust based on score magnitudes
        if vector_score > 0.8 and graph_score < 0.3:
            # High vector, low graph - trust vector more
            vector_weight = min(0.8, vector_weight * 1.4)
        elif graph_score > 0.7 and vector_score < 0.4:
            # High graph, low vector - trust graph more
            graph_weight = min(0.8, graph_weight * 1.4)
        
        # Apply quality-based final adjustment
        vector_weight = vector_weight * (0.5 + 0.5 * vector_quality)
        graph_weight = graph_weight * (0.5 + 0.5 * graph_quality)
        
        # Normalize
        total = vector_weight + graph_weight
        if total > 0:
            return {
                "vector": vector_weight / total,
                "graph": graph_weight / total
            }
        else:
            return {"vector": 0.5, "graph": 0.5}

    def _apply_ranking_adjustments(self, results: List[Dict]) -> List[Dict]:
        """Apply final ranking adjustments to improve result quality."""
        if not results:
            return results
        
        # Calculate percentile-based adjustments
        scores = [r["combined_score"] for r in results]
        if len(scores) > 1:
            score_range = max(scores) - min(scores)
            
            for i, result in enumerate(results):
                # Boost candidates with high-quality skill matches
                skill_details = result.get("skill_details", {})
                if skill_details:
                    priority_matches = sum(
                        1 for details in skill_details.values() 
                        if details.get("is_priority", False)
                    )
                    if priority_matches > 0:
                        # Boost for priority skill matches
                        boost = min(0.1, priority_matches * 0.03)
                        result["combined_score"] = min(1.0, result["combined_score"] + boost)
                
                # Penalize candidates with very low component scores
                component_scores = result.get("component_scores", {})
                if component_scores:
                    skill_score = component_scores.get("skill_score", 0)
                    if skill_score < 0.1 and result["combined_score"] > 0.5:
                        # High combined score but very low skill score is suspicious
                        result["combined_score"] *= 0.8
        
        # Re-sort after adjustments
        results.sort(key=lambda x: x["combined_score"], reverse=True)
        
        return results

    def predict(
        self,
        candidate_description: str,
        vacancy_description: str,
        hr_comment: str = "",
        temperature: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> Tuple[float, str]:
        """
        Predict the match score between a candidate and a vacancy.

        Args:
            candidate_description: Text description of the candidate (resume)
            vacancy_description: Text description of the vacancy (job posting)
            hr_comment: Optional HR comment or additional context
            temperature: Optional temperature for LLM (higher = more creative)
            seed: Optional seed for LLM (for reproducibility)

        Returns:
            Tuple of (score, explanation) where score is a float between 0 and 5
            and explanation is a string describing the reasoning
        """
        # Parse the resume if it's not already in the database
        resume_id = None

        try:
            # Try to find the resume by content first
            vector_results = self.score_candidates_vector(
                candidate_description, top_k=1
            )

            if (
                vector_results
                and len(vector_results) > 0
                and vector_results[0]["similarity_score"] > 0.95
            ):
                # Found a very similar resume, use its ID
                resume_id = vector_results[0]["id"]
            else:
                # Parse and store the resume
                parsed_resume = self.parse_and_store_resume(candidate_description)
                resume_id = parsed_resume.get("id")

        except Exception as e:
            print(f"Error handling resume: {str(e)}")
            # Continue without resume parsing, we'll use the raw text
            pass

        # Combine vector and graph-based scoring
        vector_results = self.score_candidates_vector(vacancy_description)
        graph_results = self.score_candidates_graph(vacancy_description)
        combined_results = self.combine_scores(vector_results, graph_results)

        # Check if this candidate is in the top results
        candidate_rank = -1
        candidate_score = 0.0

        for i, result in enumerate(combined_results):
            if resume_id and result["id"] == resume_id:
                candidate_rank = i + 1
                candidate_score = result["combined_score"]
                break

        # If candidate wasn't found in combined results or no resume_id,
        # we need to evaluate based on the text directly
        if candidate_rank == -1:
            # Create a GraphRAG object for evaluation
            # Use only the vector retriever for RAG evaluation
            rag = GraphRAG(retriever=self.candidate_retriever, llm=self.llm)

            # Create the evaluation prompt
            query_text = f"""
            I need to evaluate how well a candidate matches a job description.
            
            Job Description:
            {vacancy_description}
            
            Candidate Description:
            {candidate_description}
            
            {hr_comment if hr_comment else ""}
            
            Please provide:
            1. A match score from 1 to 5 (where 5 is perfect match)
            2. A detailed explanation of why this score was given
            3. List the pros and cons of this candidate for the position
            
            Return in the following format:
            Score: [numerical score 1-5]
            Explanation: [detailed explanation]
            Pros: [bullet points]
            Cons: [bullet points]
            """

            # Get RAG response
            response = rag.search(query_text=query_text)

            # Parse the response to extract score and explanation
            try:
                lines = response.answer.strip().split("\n")
                score_line = next(
                    (line for line in lines if line.startswith("Score:")), ""
                )
                score_text = score_line.replace("Score:", "").strip()
                score = (
                    float(score_text) if score_text.replace(".", "").isdigit() else 0.0
                )

                # Extract explanation (everything after "Explanation:" until the next section)
                explanation_start = response.answer.find("Explanation:")
                pros_start = response.answer.find("Pros:")

                if explanation_start != -1 and pros_start != -1:
                    explanation = response.answer[
                        explanation_start + 12 : pros_start
                    ].strip()
                else:
                    explanation = response.answer

                return score, explanation

            except Exception as e:
                print(f"Error parsing RAG response: {str(e)}")
                # Use LM predictor as fallback
                pass

        # If we have a score from the database matching, convert to 1-5 scale
        if candidate_score > 0:
            score = 1 + (candidate_score * 4)  # Scale 0-1 to 1-5

            # Get explanation using LM
            prompt = f"""<|im_start|>system
You are an expert HR assistant that evaluates how well a candidate matches a job description.
You will be given information about a candidate, a job description, and some additional context.
Your task is to provide a detailed explanation of why the candidate received a particular match score.
<|im_end|>

<|im_start|>user
Job Description:
{vacancy_description}

Candidate Description:
{candidate_description}

{hr_comment if hr_comment else ""}

The candidate received a match score of {score:.1f} out of 5.
- The candidate was ranked #{candidate_rank} in the search results.
- The match was based on both vector similarity and graph-based feature matching.

Please provide:
1. A detailed explanation of why this score is appropriate
2. List the pros and cons of this candidate for the position
<|im_end|>"""

            explanation = self.lm_predictor._call_api(prompt)
            return score, explanation

        # If all else fails, use LM predictor
        prompt = f"""<|im_start|>system
You are an expert HR assistant that evaluates how well a candidate matches a job description.
You will be given a candidate description and a job description, and you need to:
1. Score the match from 1 to 5 (where 5 is a perfect match, 1 is a poor match)
2. Provide a detailed explanation of your reasoning
3. List the pros and cons of this candidate for the position

IMPORTANT: Structure your response using the following XML format:
<score>X</score> (where X is a number between 1 and 5)
<explanation>Your detailed explanation of why the score makes sense</explanation>
<pros>
- First specific strength
- Second specific strength
</pros>
<cons>
- First specific weakness or gap
- Second specific weakness or gap
</cons>

You may also include a <thought> tag if you want to show your reasoning, but this is optional.
<|im_end|>

<|im_start|>user
Job Description:
{vacancy_description}

Candidate Description:
{candidate_description}

{hr_comment if hr_comment else ""}

Please evaluate how well this candidate matches the job description using the specified XML format.
<|im_end|>"""

        # Store the original temperature and seed values
        original_temperature = self.lm_predictor.temperature
        original_seed = self.lm_predictor.seed

        try:
            # Set the provided temperature and seed
            if temperature is not None:
                self.lm_predictor.temperature = temperature
            if seed is not None:
                self.lm_predictor.seed = seed

            # Call the API
            response = self.lm_predictor._call_api(prompt)

            # Restore original values
            if temperature is not None:
                self.lm_predictor.temperature = original_temperature
            if seed is not None:
                self.lm_predictor.seed = original_seed

            # Extract score from XML tag
            score_match = re.search(r"<score>(.*?)</score>", response, re.DOTALL)
            if score_match:
                score_text = score_match.group(1).strip()
                score = (
                    float(score_text) if score_text.replace(".", "").isdigit() else 0.0
                )
            else:
                # Fallback to traditional score extraction
                lines = response.strip().split("\n")
                score_line = next(
                    (line for line in lines if line.startswith("Score:")), ""
                )
                score_text = score_line.replace("Score:", "").strip()
                score = (
                    float(score_text) if score_text.replace(".", "").isdigit() else 0.0
                )

            # Already extracted all necessary information in XML tags, return the full response
            # The Streamlit app will parse and format it properly
            return score, response

        except Exception as e:
            # Always restore original values in case of error
            if temperature is not None:
                self.lm_predictor.temperature = original_temperature
            if seed is not None:
                self.lm_predictor.seed = original_seed

            print(f"Error parsing LM response: {str(e)}")
            # Default values on error
            return 0.0, "Error evaluating candidate match."

    def close(self) -> None:
        """Clean up resources."""
        if hasattr(self, "driver") and self.driver:
            self.driver.close()

    def parse_and_store_resume(
        self, resume_text: str, candidate_id: Optional[str] = None
    ) -> Dict:
        """
        Parse resume text and store structured data in the knowledge graph.

        Args:
            resume_text: Raw resume text
            candidate_id: Optional candidate ID (generated if not provided)

        Returns:
            Dictionary with parsed resume data
        """
        # Generate candidate ID if not provided
        if not candidate_id:
            import uuid

            candidate_id = str(uuid.uuid4())

        # Traditional resume parsing and storage (legacy approach)
        print("Using legacy resume parser...")
        # Parse resume to extract structured information
        parsed_data = self.resume_parser.parse_resume(resume_text)

        # Extract name or use a default
        candidate_name = "Unknown Candidate"
        if parsed_data.get("basic_info") and parsed_data["basic_info"].get("name"):
            candidate_name = parsed_data["basic_info"]["name"]

        # Add candidate to knowledge graph
        self.knowledge_graph.add_candidate(
            candidate_id=candidate_id,
            name=candidate_name,
            metadata={"summary": parsed_data.get("summary", "")},
        )

        # Add skills to candidate
        for skill in parsed_data.get("skills", []):
            skill_name = skill if isinstance(skill, str) else skill.get("name")
            if skill_name:
                self.knowledge_graph.add_skill_to_candidate(
                    candidate_id=candidate_id, skill_name=skill_name
                )

        # Add experience to candidate
        for exp in parsed_data.get("experience", []):
            self.knowledge_graph.add_experience_to_candidate(
                candidate_id=candidate_id,
                company=exp.get("company", "Unknown Company"),
                role=exp.get("role", "Unknown Role")
                or exp.get("title", "Unknown Role"),
                years=exp.get("years", 0.0),
                metadata={
                    "start_date": exp.get("start_date", ""),
                    "end_date": exp.get("end_date", ""),
                    "description": exp.get("description", ""),
                },
            )

        # Add location to candidate
        location = None
        if parsed_data.get("location"):
            location = parsed_data["location"]
        elif parsed_data.get("basic_info") and parsed_data["basic_info"].get(
            "location"
        ):
            location = parsed_data["basic_info"]["location"]

        if location:
            self.knowledge_graph.add_location_to_candidate(
                candidate_id=candidate_id, location=location
            )

        # Add parsed data to the result for reference
        parsed_data["id"] = candidate_id
        parsed_data["name"] = candidate_name

        return parsed_data

    def generate_match_explanation(
        self, candidate_id: str, job_description: str, combined_score: float
    ) -> Tuple[str, Dict[str, List[str]]]:
        """
        Generate a detailed explanation of the match between a candidate and job.

        Args:
            candidate_id: ID of the candidate to explain
            job_description: Job description or query text
            combined_score: Combined match score (0-1)

        Returns:
            Tuple of (explanation text, dictionary with pros/cons lists)
        """
        # Retrieve candidate details from the database
        with self.driver.session() as session:
            # Get candidate node with all properties
            query = """
            MATCH (c:Candidate {id: $candidate_id})
            RETURN c
            """

            candidate_result = session.run(query, candidate_id=candidate_id)
            record = candidate_result.single() if candidate_result.peek() else None
            candidate_data = record.get("c") if record else None

            if not candidate_data:
                return "Candidate not found in database.", {"pros": [], "cons": []}

            # Get candidate skills
            skills_query = """
            MATCH (c:Candidate {id: $candidate_id})-[r:HAS_SKILL]->(s:Skill)
            RETURN s.name as skill
            """

            skills_result = session.run(skills_query, candidate_id=candidate_id)
            skills = [{"name": record["skill"]} for record in skills_result]

            # Get candidate experience
            experience_query = """
            MATCH (c:Candidate {id: $candidate_id})-[:HAS_EXPERIENCE]->(e:Experience)-[:AT_ROLE]->(r:Role)
            RETURN e.company as company, r.title as role, e.years as years, 
                   e.start_date as start_date, e.end_date as end_date, e.description as description
            """

            experience_result = session.run(experience_query, candidate_id=candidate_id)
            experience = [dict(record) for record in experience_result]

            # Get candidate location
            location_query = """
            MATCH (c:Candidate {id: $candidate_id})-[r:LOCATED_IN]->(l:Location)
            RETURN l.name as location, r.is_current as is_current
            """

            location_result = session.run(location_query, candidate_id=candidate_id)
            locations = [dict(record) for record in location_result]

        # Ensure combined_score is between 0 and 1
        combined_score = max(0.0, min(combined_score, 1.0))

        # Normalize score to 1-5 scale for human readability
        normalized_score = 1 + (combined_score * 4)  # Scale 0-1 to 1-5

        # Construct candidate summary
        candidate_summary = f"Name: {candidate_data.get('name', 'Unknown')}\n"

        if locations:
            locations_text = ", ".join([loc["location"] for loc in locations])
            candidate_summary += f"Location: {locations_text}\n"

        if skills:
            # Get relevant skills (those that match what's sought in the job)
            query_components = self.extract_query_components(job_description)
            query_role = query_components.get("role", "")
            query_skills = query_components.get("skills", [])

            # Ensure types are correct
            query_role = query_role.lower() if isinstance(query_role, str) else ""
            query_skills = query_skills if isinstance(query_skills, list) else []

            # Mark skills as relevant based on job description
            relevant_skills = []
            for skill in skills:
                skill_name = skill["name"]
                is_relevant = False

                # Check if skill is in query skills
                for qs in query_skills:
                    if (
                        isinstance(qs, str)
                        and isinstance(skill_name, str)
                        and (
                            qs.lower() in skill_name.lower()
                            or skill_name.lower() in qs.lower()
                        )
                    ):
                        is_relevant = True
                        break

                # Check if skill is related to role
                if not is_relevant and query_role and isinstance(skill_name, str):
                    is_relevant = query_role in skill_name.lower()

                # Add to the right list
                if is_relevant:
                    relevant_skills.append(skill_name)

            if relevant_skills:
                candidate_summary += f"Relevant Skills: {', '.join(relevant_skills)}\n"

            all_skills_text = ", ".join([skill["name"] for skill in skills])
            candidate_summary += f"All Skills: {all_skills_text}\n"

        if experience:
            candidate_summary += "Experience:\n"
            for exp in experience:
                years = exp.get("years", "Unknown")
                role = exp.get("role", "Unknown role")
                company = exp.get("company", "Unknown company")
                candidate_summary += f"- {role} at {company} ({years} years)\n"

        # Use the LLM to generate the explanation
        prompt = f"""<|im_start|>system
You are an expert HR talent matcher that evaluates how well a candidate matches a job description.
You will be given information about a candidate, a job description, and a match score.
Your task is to:
1. Explain the match between the candidate and the job
2. Identify specific strengths (pros) and weaknesses (cons) of the candidate for this position

Focus on concrete factors like skills, experience, and qualifications.

IMPORTANT: Structure your response using the following XML format:
<explanation>Your detailed explanation of why the score makes sense (2-3 sentences)</explanation>
<pros>
- First specific strength
- Second specific strength
</pros>
<cons>
- First specific weakness or gap
- Second specific weakness or gap
</cons>

Do not include the score in the explanation. The score will be displayed separately.
<|im_end|>

<|im_start|>user
Job Description:
{job_description}

Candidate Information:
{candidate_summary}

The candidate received a match score of {normalized_score:.1f} out of 5.

Please analyze the match between this candidate and the job description using the specified XML format.
<|im_end|>"""

        response = self.lm_predictor._call_api(prompt)

        # Extract the explanation and pros/cons using XML parsing
        explanation = ""
        pros = []
        cons = []
        llm_score = None

        # Parse the response
        try:
            # Try to extract content within XML tags
            # Extract explanation
            explanation_match = re.search(
                r"<explanation>(.*?)</explanation>", response, re.DOTALL
            )
            if explanation_match:
                explanation = explanation_match.group(1).strip()

            # Extract score if present
            score_match = re.search(r"<score>(.*?)</score>", response, re.DOTALL)
            if score_match:
                try:
                    llm_score = float(score_match.group(1).strip())
                except ValueError:
                    pass

            # Extract pros
            pros_match = re.search(r"<pros>(.*?)</pros>", response, re.DOTALL)
            if pros_match:
                pros_text = pros_match.group(1).strip()
                pros = [
                    line.strip()[2:].strip()
                    for line in pros_text.split("\n")
                    if line.strip().startswith("-")
                ]

            # Extract cons
            cons_match = re.search(r"<cons>(.*?)</cons>", response, re.DOTALL)
            if cons_match:
                cons_text = cons_match.group(1).strip()
                cons = [
                    line.strip()[2:].strip()
                    for line in cons_text.split("\n")
                    if line.strip().startswith("-")
                ]

            # Fallback to traditional parsing if no XML tags are found
            if not explanation:
                # Try traditional parsing with "Pros:" and "Cons:" labels
                pros_index = response.find("Pros:")
                if pros_index != -1:
                    explanation = response[:pros_index].strip()

                    # Extract pros
                    cons_index = response.find("Cons:")
                    if cons_index != -1:
                        pros_text = response[pros_index + 5 : cons_index].strip()
                        if not pros:  # Only replace if XML parsing didn't get any pros
                            pros = [
                                line.strip()[2:].strip()
                                for line in pros_text.split("\n")
                                if line.strip().startswith("-")
                            ]

                        # Extract cons
                        cons_text = response[cons_index + 5 :].strip()
                        if not cons:  # Only replace if XML parsing didn't get any cons
                            cons = [
                                line.strip()[2:].strip()
                                for line in cons_text.split("\n")
                                if line.strip().startswith("-")
                            ]
                else:
                    explanation = response.strip()

        except Exception as e:
            print(f"Error parsing explanation: {str(e)}")
            explanation = response.strip()

        result = {"pros": pros, "cons": cons}

        # Include the LLM's score if it provided one
        if llm_score is not None:
            # Create a new dictionary that includes the original result plus the score
            result = {**result, "llm_score": llm_score}

        return explanation, result
