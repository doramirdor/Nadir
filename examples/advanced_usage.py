import os
from src.llm_selector.core import LLMSelector
from src.config.settings import ModelConfig, DynamicLLMSelectorConfig

def main():
    # Create custom configuration
    custom_config = DynamicLLMSelectorConfig(
        models={
            "gpt-3.5": ModelConfig(
                name="gpt-3.5-turbo",
                provider="openai",
                complexity_threshold=40.0
            ),
            "gpt-4": ModelConfig(
                name="gpt-4",
                provider="openai",
                complexity_threshold=75.0
            )
        }
    )

    # Initialize LLM Selector with custom configuration
    llm_selector = LLMSelector()

    # Complex prompts with varying complexity
    prompts = [
        "Hello, how are you?",
        "Explain the basics of machine learning",
        "Provide a comprehensive analysis of quantum computing's impact on cryptography"
    ]

    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        details = llm_selector.get_complexity_details(prompt)
        print("Complexity Details:", details)
        
        # Generate response
        response = llm_selector.generate_response(prompt)
        print("Response:", response[:100] + "...")

if __name__ == "__main__":
    main()