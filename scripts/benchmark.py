import time
import statistics
from src.nadir.llm_selector.selector.auto import AutoSelector

def benchmark_model_selection():
    """
    Benchmark model selection performance
    """
    llm_selector = AutoSelector()
    
    # Test prompts with varying complexity
    prompts = [
        "Hello",
        "Explain machine learning algorithms",
        "Provide a comprehensive analysis of quantum computing's cryptographic implications"
    ]
    
    selection_times = []
    
    for prompt in prompts:
        start_time = time.time()
        llm_selector.select_model(prompt)
        end_time = time.time()
        
        selection_times.append(end_time - start_time)
    
    print("Model Selection Performance:")
    print(f"Average Selection Time: {statistics.mean(selection_times):.4f} seconds")
    print(f"Min Selection Time: {min(selection_times):.4f} seconds")
    print(f"Max Selection Time: {max(selection_times):.4f} seconds")

def benchmark_response_generation():
    """
    Benchmark response generation performance
    """
    llm_selector = AutoSelector()
    
    prompts = [
        "Write a short greeting",
        "Explain a complex scientific concept",
        "Generate a detailed technical report"
    ]
    
    generation_times = []
    
    for prompt in prompts:
        start_time = time.time()
        llm_selector.generate_response(prompt)
        end_time = time.time()
        
        generation_times.append(end_time - start_time)
    
    print("\nResponse Generation Performance:")
    print(f"Average Generation Time: {statistics.mean(generation_times):.4f} seconds")
    print(f"Min Generation Time: {min(generation_times):.4f} seconds")
    print(f"Max Generation Time: {max(generation_times):.4f} seconds")

if __name__ == "__main__":
    benchmark_model_selection()
    benchmark_response_generation()