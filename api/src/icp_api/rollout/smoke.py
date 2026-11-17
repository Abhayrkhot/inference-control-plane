import httpx


async def smoke_test(endpoint: str, model: str, timeout: float) -> str | None:
    """Send one tiny completion. Returns None if it worked, otherwise what went wrong."""
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                f"{endpoint.rstrip('/')}/v1/completions",
                json={"model": model, "prompt": "Hello", "max_tokens": 1},
            )
    except httpx.HTTPError as e:
        return f"request failed: {type(e).__name__}: {e}"
    if response.status_code != 200:
        return f"HTTP {response.status_code}: {response.text[:200]}"
    try:
        choices = response.json().get("choices")
    except ValueError:
        return "response was not JSON"
    return None if choices else "response had no choices"
