import re
import requests
import json
import time
from src.utils.config import settings


def filter_response(response: str) -> str:
    response = re.sub(r"\*\*|__|~~|`", "", response)
    response = re.sub(r"[\U00010000-\U0010ffff]", "", response, flags=re.UNICODE)
    return response


def warmup_llm(session: requests.Session, llm_model: str, llm_url: str):
    try:
        # Check server status first
        health = session.get("http://localhost:11434", timeout=3)
        if health.status_code != 200:
            print("Ollama not running! Start it first.")
            return

        # Model warmup with empty context
        session.post(
            llm_url,
            json={
                "model": llm_model,
                "messages": [{"role": "user", "content": "."}],
                "context": [],
                "options": {"num_ctx": 64},
            },
            timeout=5,
        )

    except requests.RequestException as e:
        print(f"Warmup failed: {str(e)}")
        return


def get_ai_response(
    session: requests.Session,
    messages: list,
    llm_model: str,
    llm_url: str,
    max_tokens: int,
    temperature: float = 0.7,
    stream: bool = False,
):
    try:
        response = session.post(
            llm_url,
            json={
                "model": llm_model,
                "messages": messages,
                "options": {
                    "num_ctx": settings.TARGET_SIZE,  # Reduced context window
                    "num_thread": 4,  # Optimal for most CPUs
                    "repeat_penalty": 1.0,
                    "stop": ["\n"],  # Stop at newlines
                },
                "stream": True,
            },
            timeout=8,
            stream=True,
        )
        response.raise_for_status()

        def streaming_iterator():
            try:
                for chunk in response.iter_content(chunk_size=512):
                    if chunk:
                        yield chunk
                    else:
                        # Send minimal silence packet instead of space
                        yield b"\x00\x00"  # 16-bit silence sample
            except Exception as e:
                print(f"\nError: {str(e)}")
                yield b"\x00\x00"  # Ensure final silence

        return streaming_iterator()

    except Exception as e:
        print(f"\nError: {str(e)}")


def parse_stream_chunk(chunk: bytes) -> dict:
    """Handle empty chunks safely"""
    if not chunk:
        return {"keep_alive": True}
    
    try:
        text = chunk.decode("utf-8").strip()
        if text.startswith("data: "):
            text = text[6:]
        if text == "[DONE]":
            return {"choices": [{"finish_reason": "stop", "delta": {}}]}
        if text.startswith("{"):
            data = json.loads(text)
            content = ""
            if "message" in data:
                content = data["message"].get("content", "")
            elif "choices" in data and data["choices"]:
                choice = data["choices"][0]
                content = choice.get("delta", {}).get("content", "") or choice.get(
                    "message", {}
                ).get("content", "")

            if content:
                return {"choices": [{"delta": {"content": filter_response(content)}}]}
        return None

    except Exception as e:
        if str(e) != "Expecting value: line 1 column 2 (char 1)":
            print(f"Error parsing stream chunk: {str(e)}")
        return None
