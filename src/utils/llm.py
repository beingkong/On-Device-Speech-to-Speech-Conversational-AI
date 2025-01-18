import re
import requests
import json

def filter_response(response):
    """Remove markdown and emojis from AI response"""
    # Remove markdown
    response = re.sub(r'\*\*|__|~~|`', '', response)  # Remove markdown symbols
    # Remove emojis
    response = re.sub(r'[\U00010000-\U0010ffff]', '', response, flags=re.UNICODE)  # Remove emojis
    return response

def get_ai_response(messages, llm_model, lm_studio_url, max_tokens, temperature=0.7, stream=False):
    """Get response from LM Studio API"""
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
            stream=stream  # Enable streaming at request level
        )
        response.raise_for_status()
        
        if stream:
            return response.iter_lines()
        else:
            return response.json()["choices"][0]["message"]["content"]
            
    except requests.RequestException as e:
        print(f"Error communicating with LM Studio: {str(e)}")
        return None

def parse_stream_chunk(chunk):
    """Parse a chunk from the stream into a response object"""
    if not chunk:
        return None
        
    try:
        # Decode the chunk
        text = chunk.decode('utf-8').strip()
        
        # Handle [DONE] messages specially
        if text == '[DONE]' or text == 'data: [DONE]':
            return {
                "choices": [{
                    "finish_reason": "stop",
                    "delta": {}
                }]
            }
            
        # Remove 'data: ' prefix if present
        if text.startswith('data: '):
            text = text[6:]
            
        # Filter out emojis and special characters before JSON parsing
        text = re.sub(r'[\U00010000-\U0010ffff]', '', text)
            
        try:
            # Parse the JSON
            data = json.loads(text)
            
            # Handle OpenAI format
            if "choices" in data and len(data["choices"]) > 0:
                choice = data["choices"][0]
                # If this is a delta message
                if "delta" in choice:
                    # Filter content if present
                    if "content" in choice["delta"]:
                        choice["delta"]["content"] = filter_response(choice["delta"]["content"])
                    return data
                # If this is a regular message
                elif "message" in choice:
                    if "content" in choice["message"]:
                        choice["message"]["content"] = filter_response(choice["message"]["content"])
                    return {
                        "choices": [{
                            "delta": choice["message"]
                        }]
                    }
                # If this is a completion message
                elif "finish_reason" in choice:
                    return {
                        "choices": [{
                            "finish_reason": choice["finish_reason"],
                            "delta": {}
                        }]
                    }
            return data
            
        except json.JSONDecodeError as e:
            # If JSON parsing fails, it might be a direct text response
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
        if str(e) != "Expecting value: line 1 column 2 (char 1)":  # Skip common streaming artifacts
            print(f"Error parsing stream chunk: {str(e)}")
        return None 