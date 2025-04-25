import os
from tortoise import Tortoise
from tortoise.contrib.fastapi import register_tortoise

from src.service.models import MatchResult

DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "lunapark")

DB_URL = f"postgres://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

TORTOISE_ORM = {
    "connections": {"default": DB_URL},
    "apps": {
        "models": {
            "models": ["src.service.models", "aerich.models"],
            "default_connection": "default",
        }
    },
}


async def init_db():
    """Initialize the database connection."""
    await Tortoise.init(
        db_url=DB_URL,
        modules={"models": ["src.service.models"]},
    )
    await Tortoise.generate_schemas()


def init_db_for_fastapi(app):
    """Initialize Tortoise ORM for FastAPI application."""
    register_tortoise(
        app,
        config=TORTOISE_ORM,
        generate_schemas=True,
        add_exception_handlers=True,
    ) 