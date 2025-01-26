<h1 align="center">
     Nadir
  <br>
</h1>


<div align="center">

<img src="https://private-user-images.githubusercontent.com/167151565/406756998-83ec52f5-7310-44f3-a2c8-bfcafee6345d.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3Mzc5MTg0OTIsIm5iZiI6MTczNzkxODE5MiwicGF0aCI6Ii8xNjcxNTE1NjUvNDA2NzU2OTk4LTgzZWM1MmY1LTczMTAtNDRmMy1hMmM4LWJmY2FmZWU2MzQ1ZC5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjUwMTI2JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI1MDEyNlQxOTAzMTJaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT0yMjgzMzE0NDNlN2NkOGYzYTM0NDI4MTYwMDYzZjI1ZDdlNTEyZTMxMjE0ZGQyOWRjN2Q1OTA3OTBjMjY4OGE1JlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.xk3vjEjxK6r8Kga3jtyixFtWEert83Uq3pgIW22ESSg" width="25%" height="25%">

</div>

<p align="center">
  <b>You Do You, We Optimize the Rest</b> <br />
  <b>Dynamic Model Selection • Cost-Efficient Compression • Multi-Provider Support</b> <br />
</p>

---

[![AGPL License](https://img.shields.io/badge/license-AGPL-blue.svg)](http://www.gnu.org/licenses/agpl-3.0)
![example workflow](https://github.com/doramirdor/Nadir/actions/workflows/python-package.yml/badge.svg)
![version](https://img.shields.io/badge/version-0.1.0-blue)

---

## Introduction

**Nadir** is a cutting-edge package that dynamically selects and interacts with Large Language Models (LLMs) based on prompt complexity, cost constraints, and performance goals. Whether you're working with OpenAI, Anthropic, Gemini, or Hugging Face models, Nadir ensures you always pick the right tool for the job.

From **cost-efficient compression** to **insightful complexity analysis**, Nadir brings intelligence to your LLM workflows, helping you save resources while maximizing output quality.

---

## Why Choose Nadir?

- **Dynamic Model Selection**: Automatically choose the best LLM for any given task based on complexity and cost thresholds.
- **Cost Optimization**: Minimize token usage and costs with intelligent prompt compression.
- **Multi-Provider Support**: Seamless integration with OpenAI, Anthropic, Google Gemini, and Hugging Face.
- **Extensible Design**: Add your own complexity analyzers, compression strategies, or new providers effortlessly.
- **Rich Insights**: Generate detailed metrics on token usage, costs, and model performance.

---

## Installation

Install Nadir using pip:

```bash
pip install nadir
```

---

## Getting Started

Here’s a quick example to get you up and running with Nadir:

### Dynamic Model Selection
```python
from nadir import LLMSelector
from nadir.config import ModelConfig

# Register custom model configurations
custom_model_config = ModelConfig(
    name="gemini-1.5-flash-8b",
    provider="gemini",
    complexity_threshold=10.0,
    cost_per_1k_tokens_input=0.003,
    cost_per_1k_tokens_output=0.005,
    model_instance=GeminiProvider("gemini-1.5-flash-8b")
)

# Initialize the model registry
model_registry = ModelRegistry()
model_registry.register_models([custom_model_config])

# Initialize the LLMSelector with the custom model
selector = LLMSelector(model_registry=model_registry)

# Example prompt
prompt = "Explain the theory of relativity in simple terms."

# Dynamically select a model and generate a response
details = selector.get_complexity_details(prompt)
print("Complexity Details:", details)

# Generate response
response = selector.generate_response(prompt)
print(response)
```

---

## Advanced Examples

### Prompt Compression
Nadir allows you to compress prompts dynamically to save costs without losing key context.

#### Example: Compressing a Detailed Prompt
```python
from nadir import PromptCompressor

# Initialize the PromptCompressor
compressor = PromptCompressor()

# Original prompt
original_prompt = (
    "Please provide a detailed analysis of quantum mechanics, focusing on its historical development, \
    key contributors like Einstein and Planck, and its implications for modern physics. Additionally, \
    explain how it relates to general relativity and string theory."
)

# Compress the prompt
compressed_prompt = compressor.compress(original_prompt)

# Output the compressed version
print("Original Prompt:", original_prompt)
print("Compressed Prompt:", compressed_prompt)

# Pass the compressed prompt to the LLM selector
response = selector.generate_response(compressed_prompt)
print("Response:", response)
```

### Custom Complexity Analyzer
You can add your own custom complexity analyzers to tailor Nadir to your unique use cases.

```python
from nadir import LLMSelector
from nadir.analysis import ComplexityAnalyzer

class CustomAnalyzer(ComplexityAnalyzer):
    def analyze(self, prompt):
        # Custom logic for complexity scoring
        return len(prompt.split()) * 0.1

# Initialize selector with a custom analyzer
selector = LLMSelector(complexity_analyzer=CustomAnalyzer())

prompt = "Summarize the history of the Roman Empire."
complexity_score = selector.get_complexity_score(prompt)
print("Custom Complexity Score:", complexity_score)
```

### Multi-Model Usage
Leverage Nadir to interact with multiple models in a single workflow.

#### Example 1: Basic Multi-Model Workflow
```python
from nadir import LLMSelector

# Example prompts
prompts = [
    "Summarize the latest research on black holes.",
    "Write a short poem about the ocean.",
    "What are the benefits of renewable energy?"
]

# Iterate through prompts and dynamically select models
for prompt in prompts:
    response = selector.generate_response(prompt)
    print(f"Prompt: {prompt}\nResponse: {response}\n")
```

#### Example 2: Advanced Multi-Model Configuration
```python
from nadir import LLMSelector
from nadir.config import DynamicLLMSelectorConfig, ModelConfig

# Create custom configuration
custom_config = DynamicLLMSelectorConfig(
    models={
        # Gemini models
        "gemini-1.5-flash": ModelConfig(
            name="gemini-1.5-flash",
            provider="gemini",
            complexity_threshold=20.0,
            cost_per_1k_tokens_input=0.0025,
            cost_per_1k_tokens_output=0.0035,
            model_instance=GeminiProvider("gemini-1.5-flash")
        ),
        "gemini-1.5-flash-8b": ModelConfig(
            name="gemini-1.5-flash-8b",
            provider="gemini",
            complexity_threshold=10.0,
            cost_per_1k_tokens_input=0.003,
            cost_per_1k_tokens_output=0.005,
            model_instance=GeminiProvider("gemini-1.5-flash-8b")
        ),
        # OpenAI models
        "gpt-3.5-turbo": ModelConfig(
            name="gpt-3.5-turbo",
            provider="openai",
            complexity_threshold=40.0,
            cost_per_1k_tokens_input=0.0015,
            cost_per_1k_tokens_output=0.002,
            model_instance=OpenAIProvider("gpt-3.5-turbo")
        ),
        "gpt-4": ModelConfig(
            name="gpt-4",
            provider="openai",
            complexity_threshold=75.0,
            cost_per_1k_tokens_input=0.03,
            cost_per_1k_tokens_output=0.06,
            model_instance=OpenAIProvider("gpt-4")
        )
    }
)

# Initialize the LLMSelector with the custom configuration
selector = LLMSelector(config=custom_config)

# Example prompt
prompt = "Describe the impact of artificial intelligence on healthcare."

# Dynamically select a model and generate a response
response = selector.generate_response(prompt)
print(f"Selected Model Response: {response}")
```

---

## Community & Support

Join the conversation and get support in our **[Discord Community](https://discord.gg/nadir)**. You can also find examples, documentation, and updates on our **[Website](https://nadir.ai)**.

---

## License

Nadir is released under the **AGPL License**. It is strictly restricted for commercial use unless a specific agreement is obtained. Contributions are welcome! Open a pull request or reach out via the community channels.

---
