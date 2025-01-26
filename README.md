<div align="center">

# Nadir

<<img src="https://private-user-images.githubusercontent.com/167151565/406702548-e67fd6b8-47f4-4351-ab50-a547bb049bcc.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3Mzc4NjA4MDMsIm5iZiI6MTczNzg2MDUwMywicGF0aCI6Ii8xNjcxNTE1NjUvNDA2NzAyNTQ4LWU2N2ZkNmI4LTQ3ZjQtNDM1MS1hYjUwLWE1NDdiYjA0OWJjYy5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjUwMTI2JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI1MDEyNlQwMzAxNDNaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT01MzljOWNjN2EwYjk5MzFlYWFmZDBhZWU2ZjU2ODFiZjVlNWVjNWU1ZjhiMTNmZjk4NmY1OGU2NmU5YWMzMDRkJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.HUKuYMOfJpiLkG4tGMKK64joLOcjRQ7EZ4cbbQfeAiI" width="25%" height="25%">
</div>

---

[![AGPL License](https://img.shields.io/badge/license-AGPL-blue.svg)](http://www.gnu.org/licenses/agpl-3.0)
![example workflow](https://github.com/doramirdor/Nadir/actions/workflows/python-package.yml/badge.svg)

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
model_registry.register_models([custom_config])

# Initialize the LLMSelector with the custom model
selector = LLMSelector(model_registry=model_registry)

# Example prompt
prompt = "Explain the theory of relativity in simple terms."

# Dynamically select a model and generate a response
details = llm_selector.get_complexity_details(prompt)
print("Complexity Details:", details)
    
# Generate response
response = llm_selector.generate_response(prompt)
print(response)
```

---

## Community & Support

Join the conversation and get support in our **[Discord Community](https://discord.gg/nadir)**. You can also find examples, documentation, and updates on our **[Website](https://nadir.ai)**.

---

## License

Nadir is released under the **AGPL License**. It is strictly restricted for commercial use unless a specific agreement is obtained. Contributions are welcome! Open a pull request or reach out via the community channels.

---
