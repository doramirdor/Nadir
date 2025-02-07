<h1 align="center">
     Nadir
  <br>
</h1>

<div align="center">
<img src="https://iili.io/2LWOIhx.md.png" width="25%" height="25%">
</div>

<p align="center">
  <b>No Overhead, Just Output</b> <br />
  <b>AI-Driven LLM Selection • Complexity Analysis  • Cost-Efficient Compression • Multi-Provider Support</b> <br />
</p>

---

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![example workflow](https://github.com/doramirdor/Nadir/actions/workflows/python-package.yml/badge.svg)
![version](https://img.shields.io/badge/version-0.1.0-blue)

---

## 🔍 **Overview**

**Nadir** is an intelligent **LLM selection framework** that dynamically chooses the best AI model for a given prompt based on:

- 🚀 **Complexity Analysis**: Evaluates text structure, difficulty, and token usage.
- ⚡ **Multi-Provider Support**: Works with OpenAI, Anthropic, Gemini, and Hugging Face models.
- 💰 **Cost & Speed Optimization**: Balances model **accuracy, response time, and pricing**.
- 🔄 **Adaptive Compression**: Reduces token usage via **truncation, keyword extraction, or AI-powered compression**.

---

## Why Choose Nadir?

- **Dynamic Model Selection**: Automatically choose the best LLM for any given task based on complexity and cost thresholds.
- **Cost Optimization**: Minimize token usage and costs with intelligent prompt compression.
- **Multi-Provider Support**: Seamless integration with OpenAI, Anthropic, Google Gemini, and Hugging Face.
- **Extensible Design**: Add your own complexity analyzers, compression strategies, or new providers effortlessly.
- **Rich Insights**: Generate detailed metrics on token usage, costs, and model performance.

<img width="857" alt="Image" src="https://github.com/user-attachments/assets/2c25a21c-eb16-48b8-a205-df82ffc77cdc" />
---

## Installation

Install Nadir using pip:

```bash
pip install nadir
```

---

## Set Up Environment Variables

Create a .env file to store your API keys:

```
# .env file
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
GEMINI_API_KEY=your_google_ai_key
HUGGINGFACE_API_KEY=your_huggingface_api_key

```

## 🚀 Usage

### 🔹 Select the Best LLM for a Prompt

```python
from src.llm_selector.selector.auto import AutoSelector

nadir = AutoSelector()
prompt = "Explain quantum entanglement in simple terms."
response = nadir.generate_response(prompt)

print(response)
```

### 🔹 Get Complexity Analysis & Recommended Model

```python
complexity_details = nadir.get_complexity_details("What is the speed of light in vacuum?")
print(complexity_details)
```

### 🔹 List Available Models

```python
models = nadir.list_available_models()
print(models)
```

## ⚙️ Advanced Usage: Using GeminiComplexityAnalyzer and Compression

### 🔹 Analyzing Code Complexity and Selecting the Best LLM

```python
from src.complexity.gemini import GeminiComplexityAnalyzer
from src.llm_selector.selector.auto import AutoSelector

# Initialize the Gemini-based complexity analyzer
complexity_analyzer = GeminiComplexityAnalyzer()

# Sample Python code
code_snippet = """
def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n-1)
"""

# Get detailed complexity metrics
complexity_details = complexity_analyzer.get_complexity_details(code_snippet)
print("Complexity Details:", complexity_details)

# Initialize Nadir and dynamically select the best model
nadir = AutoSelector(complexity_analyzer=complexity_analyzer)
selected_model = nadir.select_model(code_snippet)
print("Selected Model:", selected_model.name)


```

### 🔹 Compressing Long Prompts Before Model Selection

```python
from src.compression import GeminiCompressor
from src.llm_selector.selector.auto import AutoSelector

# Initialize Gemini-based prompt compression
compressor = GeminiCompressor()

# A very long prompt
long_prompt = """
Machine learning models require extensive preprocessing and feature engineering.
However, feature selection techniques vary widely based on the type of data.
For example, in text-based datasets, TF-IDF, word embeddings, and transformers
play a significant role, whereas in tabular data, methods like PCA, correlation
analysis, and decision tree-based feature selection are preferred.
"""

# Compress the prompt
compressed_prompt = compressor.compress(long_prompt, method="auto", max_tokens=100)
print("Compressed Prompt:", compressed_prompt)

# Use Nadir to select the best model for the compressed prompt
nadir = AutoSelector()
selected_model = nadir.select_model(compressed_prompt)
print("Selected Model:", selected_model.name)
```

### 🔹 Combining Compression & Complexity Analysis

```python
from src.compression import GeminiCompressor
from src.complexity.gemini import GeminiComplexityAnalyzer
from src.llm_selector.selector.auto import AutoSelector

# Initialize complexity analyzer and compressor
complexity_analyzer = GeminiComplexityAnalyzer()
compressor = GeminiCompressor()

# A long, complex prompt
long_prompt = """
Deep learning models often suffer from overfitting when trained on small datasets.
To combat this, techniques such as dropout, batch normalization, and L2 regularization
are widely used. Furthermore, transfer learning from pre-trained models has become
a popular method for reducing the need for large labeled datasets.
"""

# Step 1: Compress the prompt
compressed_prompt = compressor.compress(long_prompt, method="auto", max_tokens=80)
print("Compressed Prompt:", compressed_prompt)

# Step 2: Analyze complexity
complexity_details = complexity_analyzer.get_complexity_details(compressed_prompt)
print("Complexity Details:", complexity_details)

# Step 3: Select the best model
nadir = AutoSelector(complexity_analyzer=complexity_analyzer)
selected_model = nadir.select_model(compressed_prompt, complexity_details)
print("Selected Model:", selected_model.name)

```

---

## ⚙️ How It Works

### 1️⃣ Complexity Analysis

Uses `GeminiComplexityAnalyzer` to evaluate **token usage**, **linguistic difficulty**, and **structural complexity**.
Assigns a **complexity score** (0-100).

### 2️⃣ Intelligent Model Selection

Compares **complexity scor**e with **pre-configured LLM models**.
Chooses **the best trade-off between cost, accuracy, and speed**.

### 3️⃣ Efficient Response Generation

**Compresses long prompts** when necessary.
Calls **the selected model** and **tracks token usage & cost**.

---

## Community & Support

## 🛠 Development & Contributions

### 💡 We welcome contributions! Follow these steps:

### 1️⃣ Fork the Repository

```
git clone https://github.com/your-username/Nadir.git
cd Nadir
```
### 2️⃣ Create a Feature Branch
```
git checkout -b feature-improvement
```
### 3️⃣ Make Changes & Run Tests
```
pytest tests/
```
### 4️⃣ Commit & Push Changes
```
git add .
git commit -m "Added a new complexity metric"
git push origin feature-improvement
```
### 5️⃣ Submit a Pull Request
Open a PR on GitHub 🚀


Join the conversation and get support in our **[Discord Community](https://discord.gg/nadir)**. You can also find examples, documentation, and updates on our **[Website](https://nadir.ai)**.

---

## 📢 Connect with Us
💬 Have questions or suggestions? **Create an Issue** or **Start a Discussion** on GitHub.

🔥 Happy coding with **Nadir!** 🚀
