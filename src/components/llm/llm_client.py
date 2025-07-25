import requests
import json
import time
from src.config.settings import settings



def warmup_llm(session: requests.Session, llm_model: str, llm_url: str):
    """Sends a warmup request to the LLM server.

    Args:
        session (requests.Session): The requests session to use.
        llm_model (str): The name of the LLM model.
        llm_url (str): The URL of the LLM server.
    """
    try:
        health = session.get("http://localhost:11434", timeout=3)
        if health.status_code != 200:
            print("Ollama not running! Start it first.")
            return

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
    """Sends a request to the LLM and returns a streaming iterator.

    Args:
        session (requests.Session): The requests session to use.
        messages (list): The list of messages to send to the LLM.
        llm_model (str): The name of the LLM model.
        llm_url (str): The URL of the LLM server.
        max_tokens (int): The maximum number of tokens to generate.
        temperature (float, optional): The temperature to use for generation. Defaults to 0.7.
        stream (bool, optional): Whether to stream the response. Defaults to False.

    Returns:
        iterator: An iterator over the streaming response.
    """
    try:
        response = session.post(
            llm_url,
            json={
                "model": llm_model,
                "messages": messages,
                "options": {
                    "num_ctx": settings.MAX_TOKENS * 2,
                    "num_thread": settings.NUM_THREADS,
                },
                "stream": stream,
            },
            timeout=3600,
            stream=stream,
        )
        response.raise_for_status()

        def streaming_iterator():
            """Iterates over the streaming response."""
            try:
                for chunk in response.iter_content(chunk_size=512):
                    if chunk:
                        yield chunk
                    else:
                        yield b"\x00\x00"
            except Exception as e:
                print(f"\nError: {str(e)}")
                yield b"\x00\x00"

        return streaming_iterator()

    except Exception as e:
        print(f"\nError: {str(e)}")


def parse_stream_chunk(chunk: bytes) -> dict:
    """Parses a chunk of data from the LLM stream.

    Args:
        chunk (bytes): The chunk of data to parse.

    Returns:
        dict: A dictionary containing the parsed data.
    """
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
                return {"choices": [{"delta": {"content": content}}]}
        return None

    except Exception as e:
        if str(e) != "Expecting value: line 1 column 2 (char 1)":
            print(f"Error parsing stream chunk: {str(e)}")
        return None
