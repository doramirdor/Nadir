# LLMOpt: A Modular Framework for LLM Selection & Prompt Processing

LLMOpt is a flexible Python framework that allows you to:

- **Measure prompt complexity** via a custom **ComplexityAnalyzer**.
- **Dynamically select** an appropriate LLM from multiple providers (OpenAI, Hugging Face, etc.) based on that complexity.
- **Optionally compress** large or complex prompts before sending them to an LLM.

The project is organized into a set of modules that can be composed or extended as needed.

---

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage Examples](#usage-examples)
  - [Complexity Analysis](#complexity-analysis)
  - [Model Selection & Generation](#model-selection--generation)
  - [Prompt Compression](#prompt-compression)
- [Available Models](#available-models)
- [Advanced Configuration](#advanced-configuration)
- [Contributing](#contributing)
- [License](#license)

---

## Features

1. **Complexity Analysis**  
   - Calculate an overall complexity score (0–100) based on *token complexity*, *linguistic complexity*, and *structural complexity*.  
   - Easily extendable for custom metrics.

2. **LLM Selection**  
   - **LLMSelector** automatically decides which model to use based on a prompt’s complexity.  
   - Register multiple models (GPT-3.5, GPT-4, GPT-2, etc.) with individual **complexity thresholds**.

3. **Providers**  
   - **OpenAIProvider**: Uses OpenAI’s ChatCompletion API (e.g. GPT-3.5, GPT-4).  
   - **HuggingFaceProvider**: Uses locally (or remotely) hosted Hugging Face transformers (e.g. GPT-2, T5, etc.).

4. **Prompt Compression** *(Optional)*  
   - **BaseCompression**: A no-op compressor.  
   - **PromptCompressor**: A configurable compressor that can truncate text, extract keywords, or invoke an LLM for “high-quality” compression.

5. **Caching**  
   - Optional usage of `lru_cache` on compression or complexity analysis for repeated calls on the same data.

---

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/YourUsername/LLMOpt.git
   cd LLMOpt
Create a virtual environment (recommended):

bash
Copy
python3 -m venv .venv
source .venv/bin/activate
Install dependencies:

bash
Copy
pip install -r requirements.txt
(Typical libraries: openai, tiktoken, transformers, torch, etc.)

Set up environment variables (if using OpenAI):

bash
Copy
export OPENAI_API_KEY="YOUR_OPENAI_KEY"
or put this key in a .env file that your project can load.

Project Structure
bash
Copy
LLMOpt/
├── src/
│   ├── llm_selector/
│   │   ├── core.py             # LLMSelector class
│   │   ├── complexity_analyzer.py
│   │   ├── model_registry.py
│   │   ├── providers/
│   │   │   ├── __init__.py
│   │   │   ├── openai.py       # OpenAIProvider
│   │   │   └── huggingface.py  # HuggingFaceProvider
│   │   ├── compression/
│   │   │   ├── base_compression.py
│   │   │   └── prompt_compressor.py
│   │   └── ...
│   └── ...
├── README.md
├── requirements.txt
└── ...
complexity_analyzer.py – Contains ComplexityAnalyzer to compute token, linguistic, and structural complexity.
core.py – Contains LLMSelector, the main class that picks a model and generates responses.
model_registry.py – Defines ModelConfig and ModelRegistry, storing model info (name, threshold, provider instance, etc.).
providers/ – Subpackage containing different LLM providers (OpenAI, Hugging Face, or custom).
compression/ – Subpackage with BaseCompression (no-op) and more advanced compressors like PromptCompressor.
Usage Examples
Complexity Analysis
python
Copy
from src.llm_selector.complexity_analyzer import ComplexityAnalyzer

analyzer = ComplexityAnalyzer()
prompt = "Explain the differences between supervised and unsupervised learning."
details = analyzer.get_complexity_details(prompt)

print("Complexity Details:", details)
# {
#   'overall_complexity': 45.6,
#   'token_complexity': 42.2,
#   'linguistic_complexity': 48.8,
#   'structural_complexity': 39.0,
#   'token_count': 23
# }
Model Selection & Generation
python
Copy
import logging
from src.llm_selector.core import LLMSelector
from src.llm_selector.model_registry import ModelRegistry, ModelConfig
from src.llm_selector.providers.openai import OpenAIProvider
from src.llm_selector.providers.huggingface import HuggingFaceProvider

# Create a logger (optional)
logger = logging.getLogger("LLMOpt")

# Set up a model registry
model_registry = ModelRegistry(models=[
    ModelConfig(
        name="gpt-3.5-turbo",
        complexity_threshold=60.0,
        model_instance=OpenAIProvider("gpt-3.5-turbo")
    ),
    ModelConfig(
        name="gpt2-medium",
        complexity_threshold=20.0,
        model_instance=HuggingFaceProvider("gpt2-medium", max_length=100, temperature=0.8)
    )
])

# Create the selector
llm_selector = LLMSelector(
    model_registry=model_registry,
    logger=logger
)

# Generate a response
prompt = "Hello, how are you?"
response = llm_selector.generate_response(prompt)
print("Response:", response)
Prompt Compression
python
Copy
from src.llm_selector.core import LLMSelector
from src.llm_selector.model_registry import ModelRegistry, ModelConfig
from src.llm_selector.providers.openai import OpenAIProvider
from src.llm_selector.compression.prompt_compressor import PromptCompressor

# Initialize a PromptCompressor
compressor = PromptCompressor(
    openai_model="gpt-3.5-turbo",    # The LLM used for "high_quality" compression
    hf_keyword_model="dslim/bert-base-NER"
)

model_registry = ModelRegistry(models=[
    ModelConfig(
        name="gpt-3.5-turbo",
        complexity_threshold=80.0,
        model_instance=OpenAIProvider("gpt-3.5-turbo")
    )
])

llm_selector = LLMSelector(
    model_registry=model_registry,
    compression=compressor
)

long_prompt = """This is a very lengthy piece of text that might exceed token limits..."""
response = llm_selector.generate_response(long_prompt, max_tokens_for_compression=100)
print("Compressed + Model Response:", response)
Available Models
You can register as many models as you like in ModelRegistry(models=[]). Each entry is a ModelConfig with:

name: A unique string identifier.
complexity_threshold: A numeric value indicating the maximum complexity that model can handle.
model_instance: An instance of a provider (e.g., OpenAIProvider, HuggingFaceProvider, etc.).
Advanced Configuration
Environment Variables

OPENAI_API_KEY: Required if using OpenAIProvider.
GPU usage for Hugging Face models is automatically handled if CUDA is available.
Extended Complexity Analysis

Modify the ComplexityAnalyzer or create your own subclass to adjust weights, thresholds, or incorporate new metrics.
Prompt Compression

The default is BaseCompression, which returns the prompt unmodified.
For more advanced logic, use PromptCompressor or your own custom class.
Pass max_tokens_for_compression in generate_response to control the compression limit.
Logging

The LLMSelector accepts an optional logger. It defaults to a logger named after __name__.
Contributing
Fork this repository and create a new feature branch.
Make your changes (and add tests where applicable).
Submit a pull request. For major changes, discuss via an issue first.
We appreciate all forms of contribution, whether bug reports, code improvements, or feature requests.

License
Private License
All rights reserved.
You are not permitted to distribute or modify this code without explicit permission from the owner.