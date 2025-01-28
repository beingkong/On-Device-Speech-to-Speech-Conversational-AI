import re
import requests
import json
from src.utils.config import settings
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Persistent session with connection pooling
session = requests.Session()
adapter = HTTPAdapter(
    pool_connections=10,
    pool_maxsize=10,
    max_retries=Retry(total=3, backoff_factor=0.1)
)
session.mount('http://', adapter)
session.mount('https://', adapter)

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

def get_ai_response(messages: list, llm_model: str, lm_studio_url: str, max_tokens: int, 
                   temperature: float = 0.7, stream: bool = False):
    """Get response with streaming support and minimal overhead"""
    try:
        response = session.post(
            f"{lm_studio_url}/chat/completions",
            json={
                "messages": messages,
                "model": llm_model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": stream,
                "ttl": settings.LLM_TTL,
                "jit": False
            },
            headers={
                "Content-Type": "application/json",
                "X-Auto-Evict": str(settings.LLM_AUTO_EVICT).lower()
            },
            timeout=30,
            stream=stream  # Critical for streaming performance
        )
        response.raise_for_status()
        
        if stream:
            # Directly return the raw stream iterator
            return (parse_stream_chunk(chunk) for chunk in response.iter_content(chunk_size=None))
        else:
            return response.json()["choices"][0]["message"]["content"]
            
    except requests.RequestException as e:
        print(f"API Error: {str(e)}")
        return None

def parse_stream_chunk(chunk: bytes) -> dict | None:
    """Ultra-efficient stream parser with minimal processing"""
    try:
        # Handle keep-alive chunks first
        if chunk.startswith(b': ping'):
            return None
            
        # Decode before processing
        text = chunk.decode('utf-8').strip()
        if not text:
            return None
            
        # Handle completion marker
        if text == '[DONE]':
            return {'done': True}
            
        # Extract JSON payload
        if text.startswith('data: '):
            text = text[6:].strip()
            
        try:
            data = json.loads(text)
            if "choices" in data and data["choices"]:
                choice = data["choices"][0]
                content = (choice.get("delta") or choice.get("message", {})).get("content", "")
                return {
                    "content": content.translate(str.maketrans('', '', '*_~`')),
                    "done": choice.get("finish_reason") == "stop"
                }
        except json.JSONDecodeError:
            return {"content": text.translate(str.maketrans('', '', '*_~`')), "done": False}
            
    except Exception as e:
        if "Expecting value" not in str(e):
            print(f"Stream Error: {str(e)}")
        return None