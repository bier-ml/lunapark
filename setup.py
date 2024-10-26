from setuptools import setup, find_packages

setup(
    name="cv-matcher",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "fastapi",
        "uvicorn",
        "pydantic",
        "streamlit",
        "requests",
    ],
)
