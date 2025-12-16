import requests


class Groq:
    def __init__(self, api_key):
        if not api_key:
            raise ValueError("API key is required")
        self.api_key = api_key.strip()
        self.base_url = "https://api.groq.com/openai/v1"

    def chat_completions(self, messages, model, temperature=1, max_completion_tokens=1024,
                         top_p=1, stream=False, stop=None):
        """
        Makes a POST request to the Groq API for chat completions, with support for streaming responses.

        :param messages: List of messages to send to the API.
        :param model: The language model to use.
        :param temperature: Sampling temperature.
        :param max_completion_tokens: Maximum number of tokens to generate.
        :param top_p: Controls nucleus sampling.
        :param stream: Whether to stream responses.
        :param stop: Stop sequences to end completion.
        :return: The API response or a generator if streaming is enabled.
        """
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        data = {
            "messages": messages,
            "model": model,
            "temperature": temperature,
            "max_tokens": max_completion_tokens,
            "top_p": top_p,
            "stream": stream,
            "stop": stop,
        }

        try:
            response = requests.post(url, headers=headers, json=data, stream=stream)

            if stream:
                response.raise_for_status()
                for line in response.iter_lines(decode_unicode=True):
                    if line.strip():
                        yield line
            else:
                if response.status_code == 200:
                    return response.json()
                else:
                    try:
                        error_message = response.json().get("error", {}).get("message", "Unknown error")
                    except ValueError:
                        error_message = "Non-JSON response received from server."
                    raise Exception(f"Groq API Error: {error_message}")

        except requests.RequestException as e:
            raise Exception(f"Network error occurred: {e}")


def initialize_client(api_key):
    """
    Initialize and return the Groq client.

    :param api_key: API key for authentication
    :return: Groq client instance
    """
    return Groq(api_key)


def send_query(client, user_input, model="llama-3.3-70b-versatile", messages=None, **kwargs):
    """
    Send a query to the language model with support for conversation context.

    :param client: Groq client instance
    :param user_input: User's input text
    :param model: Model to use for completion
    :param messages: Optional list of previous messages for context (if chat concatenation is enabled)
    :param kwargs: Additional parameters (stream, temperature, etc.)
    :return: Response content or generator for streaming
    """
    if not user_input.strip():
        raise ValueError("Input is required")

    # If messages are provided (chat concatenation enabled), use them
    # Otherwise, create a simple message list with just the user input
    if messages is None:
        messages = [{"role": "user", "content": user_input}]

    response = client.chat_completions(messages=messages, model=model, **kwargs)

    # For non-streaming, extract the response content
    if not kwargs.get("stream", False):
        try:
            return response["choices"][0]["message"]["content"]
        except (KeyError, IndexError):
            raise Exception("Malformed response from Groq API")

    # For streaming, return the generator
    return response