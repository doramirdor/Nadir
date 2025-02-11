from setuptools import setup, find_packages

setup(
    name="nadir-llm", 
    version="0.2.0",
    description="A package for dynamically selecting and interacting with LLMs.",
    author="Dor Amir",
    author_email="amirdor@gmail.com",
    url="https://github.com/doramirdor/nadir",
    license="MIT",
    packages=find_packages(where="src"),  
    package_dir={"": "src"}, 
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "tiktoken",
        "transformers",
        "google-generativeai",
        "anthropic",
        "pydantic",
    ],
    extras_require={
        "dev": [
            "pytest",
            "flake8",
            "black",
        ],
    },
)
