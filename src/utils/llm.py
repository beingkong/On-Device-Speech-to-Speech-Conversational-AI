import re
import requests

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
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except requests.RequestException as e:
        print(f"Error communicating with LM Studio: {str(e)}")
        return None 