import re

def filter_response(response: str) -> str:
    """Removes markdown formatting and unicode characters from a string.

    Args:
        response (str): The string to filter.

    Returns:
        str: The filtered string.
    """
    response = re.sub(r"\*\*|__|~~|`", "", response)
    response = re.sub(r"[\U00010000-\U0010ffff]", "", response, flags=re.UNICODE)
    return response
