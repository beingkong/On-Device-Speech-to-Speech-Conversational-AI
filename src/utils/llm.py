import re
import requests
import json
import time

def filter_response(response: str) -> str:
    response = re.sub(r'\*\*|__|~~|`', '', response)
    response = re.sub(r'[\U00010000-\U0010ffff]', '', response, flags=re.UNICODE)
    return response

def get_ai_response(session: requests.Session, messages: list, llm_model: str, lm_studio_url: str, max_tokens: int, temperature: float = 0.7, stream: bool = False):
    attempt = 0
    max_retries = 3
    
    session.headers.update({
        "X-LM-Studio-Retries": str(max_retries),
        "X-LM-Studio-Client-Timeout": "120s"
    })
    
    while attempt < max_retries:
        try:
            response = session.post(
                f"{lm_studio_url}/chat/completions",
                json={
                    "messages": messages,
                    "model": llm_model,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "stream": stream,
                    "seed": 42,
                    "gpu_layers": 33,
                    "n_parallel": 3,
                    "keep_alive": "5m",
                    "context_overlap": 128
                },
                headers={
                    "Content-Type": "application/json",
                    "Connection": "keep-alive",
                    "Keep-Alive": f"timeout={60*5}, max=1000"
                },
                stream=stream,
                timeout=(10.0, 60)
            )
            
            if response.raw.connection and response.raw.connection.is_closed:
                raise requests.RequestException("Connection closed unexpectedly")
                
            response.raise_for_status()
            
            if stream:
                def streaming_iterator():
                    try:
                        for chunk in response.iter_content(chunk_size=512):
                            if chunk:
                                yield chunk
                            else:
                                yield b' '
                    finally:
                        session.headers.update({'Connection': 'keep-alive'})
                return streaming_iterator()
                
            else:
                return response.json()["choices"][0]["message"]["content"]
                
        except (requests.ConnectionError, requests.Timeout) as e:
            print(f"Connection error (attempt {attempt+1}/{max_retries}): {str(e)}")
            attempt += 1
            time.sleep(1.5 ** attempt)
            continue
            
    print("Max retries exceeded")
    return None

def parse_stream_chunk(chunk: bytes) -> dict:
    """Handle empty chunks safely"""
    if not chunk:
        return {"keep_alive": True}  # Default value for empty chunks
        
    try:
        text = chunk.decode('utf-8').strip()
        if text.startswith('data: '):
            text = text[6:]
            
        if text == '[DONE]':
            return {
                "choices": [{
                    "finish_reason": "stop",
                    "delta": {}
                }]
            }
            
        if text.startswith('{'):
            data = json.loads(text)
            if "choices" in data and data["choices"]:
                choice = data["choices"][0]
                content = choice.get("delta", {}).get("content", "") or choice.get("message", {}).get("content", "")
                if content:
                    return {
                        "choices": [{
                            "delta": {
                                "content": filter_response(content)
                            }
                        }]
                    }
        return None
            
    except Exception as e:
        if str(e) != "Expecting value: line 1 column 2 (char 1)":
            print(f"Error parsing stream chunk: {str(e)}")
        return None