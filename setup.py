from setuptools import setup, find_packages

setup(
    name="dynamic-llm-selector",
    version="0.1.0",
    packages=find_packages(where='src'),
    package_dir={'': 'src'}
)