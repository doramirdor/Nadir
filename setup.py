from setuptools import setup, find_packages

setup(
    name="nadir",
    version="0.1.0",
    description="A package for dynamically selecting and interacting with LLMs based on complexity and cost.",
    author="Dor Amir",
    author_email="amirdor@gmail.com",
    url="https://github.com/doramirdor/nadir",
    license="AGPL",
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Affero General Public License v3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
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
