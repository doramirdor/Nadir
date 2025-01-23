# Base class for compression, does nothing by default.
class BaseCompression:
    def compress(self, text: str, max_tokens: int = 9999) -> str:
        """
        Naive compression: does nothing and returns the original text.
        
        :param text: The original prompt text.
        :param max_tokens: (Unused here) a hint for maximum tokens.
        :return: Unmodified text.
        """
        return text