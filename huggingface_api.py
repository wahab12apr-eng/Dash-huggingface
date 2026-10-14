import os

import requests
from dotenv import load_dotenv


load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

# The old api-inference.huggingface.co endpoint is retired; requests go through the router now.
BASE_URL = "https://router.huggingface.co/hf-inference/models/"

MODELS = {
    "facebook/bart-large-cnn": "BART Large CNN (best quality)",
    "sshleifer/distilbart-cnn-12-6": "DistilBART CNN (faster)",
    "google/pegasus-xsum": "Pegasus XSum (one-sentence)",
}
DEFAULT_MODEL = "facebook/bart-large-cnn"

# BART-style models read at most ~1024 tokens; ~600 words keeps each chunk safely under that.
CHUNK_WORDS = 600

TIMEOUT = 60

session = requests.Session()


class SummarizerError(Exception):
    """An error with a message that is safe to show in the UI."""


def _chunks(text, size=CHUNK_WORDS):
    words = text.split()
    for i in range(0, len(words), size):
        yield " ".join(words[i:i + size])


def _query(text, model, min_length, max_length):
    try:
        response = session.post(
            BASE_URL + model,
            headers={
                "Authorization": f"Bearer {HF_TOKEN}",
                # Block until a cold model has loaded instead of returning 503.
                "x-wait-for-model": "true",
            },
            json={
                "inputs": text,
                "parameters": {
                    "min_length": min_length,
                    "max_length": max_length,
                    "do_sample": False,
                },
            },
            timeout=TIMEOUT,
        )
    except requests.Timeout:
        raise SummarizerError("The model took too long to respond. Please try again.")
    except requests.RequestException as e:
        raise SummarizerError(f"Network error: {e}")

    if response.status_code == 401:
        raise SummarizerError("Invalid Hugging Face token. Check HF_TOKEN in your .env file.")
    if response.status_code == 429:
        raise SummarizerError("Rate limit reached. Wait a moment and try again.")

    try:
        result = response.json()
    except ValueError:
        raise SummarizerError(f"Unexpected response ({response.status_code}) from Hugging Face.")

    if isinstance(result, list) and result and "summary_text" in result[0]:
        return result[0]["summary_text"].strip()

    error = result.get("error") if isinstance(result, dict) else None
    raise SummarizerError(error or f"Unexpected response from Hugging Face: {result}")


def summarize_text(text, model=DEFAULT_MODEL, min_length=30, max_length=130):
    """Summarize text, splitting long inputs into chunks the model can handle."""
    if not HF_TOKEN:
        raise SummarizerError("HF_TOKEN is not set. Add it to your .env file.")
    if model not in MODELS:
        raise SummarizerError(f"Unknown model: {model}")

    chunks = list(_chunks(text))
    if len(chunks) == 1:
        return _query(chunks[0], model, min_length, max_length)

    # Summarize each chunk, then recursively condense the joined partial summaries.
    partial = " ".join(_query(c, model, min_length, max_length) for c in chunks)
    if len(partial.split()) > max_length:
        return summarize_text(partial, model, min_length, max_length)
    return partial
