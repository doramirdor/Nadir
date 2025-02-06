from litellm import completion

class LLMEvaluator:
    def __init__(self, model: str = "gemini-1.5-flash-8b"):
        """
        Initializes the evaluator with an API key and a model.
        """
        self.model = model

    def evaluate_responses(self, prompt: str, response_1: str, response_2: str):
        """
        Evaluates two responses to a given prompt.
        Returns similarity score, relevance scores, and an explanation of differences.
        """
        evaluation_prompt = f"""
            You are an evaluator analyzing two responses to a given prompt. Evaluate the following:
            1. How similar are the two responses? Provide a similarity score from 0 to 100.
            2. How relevant is each response to the original prompt? Provide a relevance score (0 to 100) for each response.
            3. Write a short explanation of the key differences between the two responses.

            Prompt:
            {prompt}

            Response 1:
            {response_1}

            Response 2:
            {response_2}

            Provide the result in the following format:
            - Similarity Score: <value>
            - Relevance Score for Response 1: <value>
            - Relevance Score for Response 2: <value>
            - Differences: <short explanation>
        """

        try:
            # Call the litellm API for completion
            response = completion(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant specialized in evaluating text."},
                    {"role": "user", "content": evaluation_prompt},
                ],
            )
            return response['choices'][0]['message']['content']
        except Exception as e:
            print("Error during litellm API call:", str(e))
            return None