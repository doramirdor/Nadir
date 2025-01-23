from src.llm_selector.core import LLMSelector

def main():
    # Initialize LLM Selector
    llm_selector = LLMSelector()

    # Simple prompt
    simple_prompt = "Write a short greeting."
    print("Simple Prompt Complexity:")
    print(llm_selector.get_complexity_details(simple_prompt))

    # Generate response
    response = llm_selector.generate_response(simple_prompt)
    print("\nResponse:", response)

if __name__ == "__main__":
    main()