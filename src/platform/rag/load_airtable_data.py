#!/usr/bin/env python3
"""
Script to load data from Airtable into  Neo4j for candidate-vacancy matching.
This is a one-time process to initialize the databases.
"""

import argparse
import logging
import os
from typing import Optional

from dotenv import load_dotenv

from src.platform.rag.airtable_loader import AirtableLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main(
    airtable_token: Optional[str] = None,
    airtable_base: Optional[str] = None,
    airtable_table: Optional[str] = None,
    neo4j_uri: str = "bolt://localhost:7687",
    neo4j_user: str = "neo4j",
    neo4j_password: str = "password",
):
    """
    Load data from Airtable into Neo4j.

    Args:
        airtable_token: Airtable API token
        airtable_base: Airtable base ID
        airtable_table: Airtable table ID
        neo4j_uri: URI for Neo4j database
        neo4j_user: Neo4j username
        neo4j_password: Neo4j password
    """
    # Load environment variables
    load_dotenv()

    # Override environment variables with command-line arguments if provided
    if airtable_token:
        os.environ["AIRTABLE_TOKEN"] = airtable_token
    if airtable_base:
        os.environ["AIRTABLE_BASE"] = airtable_base
    if airtable_table:
        os.environ["AIRTABLE_TABLE"] = airtable_table

    # Set Neo4j environment variables
    os.environ["NEO4J_URI"] = neo4j_uri
    os.environ["NEO4J_USER"] = neo4j_user
    os.environ["NEO4J_PASSWORD"] = neo4j_password

    try:
        logger.info("Starting data load from Airtable to Neo4j")

        # Initialize loader
        loader = AirtableLoader()

        # Load data to knowledge graph
        logger.info("Loading data to knowledge graph (Neo4j)")
        loader.load_to_knowledge_graph()

        # Close connections
        loader.close()

        logger.info("Data loading completed successfully")

    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Load data from Airtable to Neo4j")

    parser.add_argument("--airtable-token", help="Airtable API token")
    parser.add_argument("--airtable-base", help="Airtable base ID")
    parser.add_argument("--airtable-table", help="Airtable table ID")
    parser.add_argument(
        "--neo4j-uri", default="bolt://localhost:7687", help="Neo4j URI"
    )
    parser.add_argument("--neo4j-user", default="neo4j", help="Neo4j username")
    parser.add_argument("--neo4j-password", default="password", help="Neo4j password")

    args = parser.parse_args()

    main(
        airtable_token=args.airtable_token,
        airtable_base=args.airtable_base,
        airtable_table=args.airtable_table,
        neo4j_uri=args.neo4j_uri,
        neo4j_user=args.neo4j_user,
        neo4j_password=args.neo4j_password,
    )
