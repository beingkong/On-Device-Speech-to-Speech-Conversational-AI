import re
import requests
import json

def filter_response(response: str) -> str:
    """Remove markdown and emojis from a string.

    Args:
        response: The string to filter.

    Returns:
        The filtered string.
    """
    response = re.sub(r'\*\*|__|~~|`', '', response)
    response = re.sub(r'[\U00010000-\U0010ffff]', '', response, flags=re.UNICODE)
    return response

def get_ai_response(messages: list, llm_model: str, lm_studio_url: str, max_tokens: int, temperature: float = 0.7, stream: bool = False):
    """Get response from LM Studio API.

    Args:
        messages: A list of message dictionaries.
        llm_model: The name of the LLM model.
        lm_studio_url: The URL of the LM Studio API.
        max_tokens: The maximum number of tokens to generate.
        temperature: The sampling temperature.
        stream: Whether to stream the response.

    Returns:
        The response from the API, either as an iterator of lines or a string.
    """
    try:
        response = requests.post(
            f"{lm_studio_url}/chat/completions",
            json={
                "messages": messages,
                "model": llm_model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": stream
            },
            headers={"Content-Type": "application/json"},
            stream=stream
        )
        response.raise_for_status()
        
        if stream:
            return response.iter_lines()
        else:
            return response.json()["choices"][0]["message"]["content"]
            
    except requests.RequestException as e:
        print(f"Error communicating with LM Studio: {str(e)}")
        return None

def parse_stream_chunk(chunk: bytes) -> dict | None:
    """Parse a chunk from the stream into a response object.

    Args:
        chunk: The chunk of data from the stream.

    Returns:
        A dictionary representing the parsed chunk, or None if parsing fails.
    """
    if not chunk:
        return None
        
    try:
        text = chunk.decode('utf-8').strip()
        
        if text == '[DONE]' or text == 'data: [DONE]':
            return {
                "choices": [{
                    "finish_reason": "stop",
                    "delta": {}
                }]
            }
            
        if text.startswith('data: '):
            text = text[6:]
            
        text = re.sub(r'[\U00010000-\U0010ffff]', '', text)
            
        try:
            data = json.loads(text)
            
            if "choices" in data and len(data["choices"]) > 0:
                choice = data["choices"][0]
                if "delta" in choice:
                    if "content" in choice["delta"]:
                        choice["delta"]["content"] = filter_response(choice["delta"]["content"])
                    return data
                elif "message" in choice:
                    if "content" in choice["message"]:
                        choice["message"]["content"] = filter_response(choice["message"]["content"])
                    return {
                        "choices": [{
                            "delta": choice["message"]
                        }]
                    }
                elif "finish_reason" in choice:
                    return {
                        "choices": [{
                            "finish_reason": choice["finish_reason"],
                            "delta": {}
                        }]
                    }
            return data
            
        except json.JSONDecodeError as e:
            if text and not text.startswith('{') and not text.startswith('['):
                filtered_text = filter_response(text)
                return {
                    "choices": [{
                        "delta": {
                            "content": filtered_text
                        }
                    }]
                }
            raise e
            
    except Exception as e:
        if str(e) != "Expecting value: line 1 column 2 (char 1)":
            print(f"Error parsing stream chunk: {str(e)}")
        return None