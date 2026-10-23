from dotenv import load_dotenv
import os
import httpx
from openai import OpenAI
load_dotenv()
import json as _json
import time
from typing import cast
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from collections import Counter

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
BIFROST_BASE_URL = os.getenv("BIFROST_BASE_URL", "http://localhost:8080")
BIFROST_GROQ_VIRTUAL_KEY = os.getenv("BIFROST_GROQ_VIRTUAL_KEY", "")
BIFROST_MISTRAL_VIRTUAL_KEY = os.getenv("BIFROST_MISTRAL_VIRTUAL_KEY", "")

GROQ_MODEL = "groq/openai/gpt-oss-120b"
MISTRAL_MODEL = "mistral/ministral-8b-latest"
MISTRAL_FALLBACK_MODEL = "mistral/ministral-3b-latest"

TEST_PROMPTS = {
    "simple": "What is the capital of America?",
    "reasoning": "Explain the difference between RAG and fine-tuning in 3 bullet points.",
    "code": "Write a Python function that validates an email address using regex.",
    "duplicate1": "What is Langchain used for?",
    "duplicate2": "What is Langchain primarily used for?",
    "deepwiki": "What are the stream modes in the new langgraph version?",
    "tavily": "Search the web for the latest news about GROQ AI and summarize the top results."
}

try:
    health = httpx.get(f"{BIFROST_BASE_URL}/health", timeout=3)
    bifrost_ok = health.status_code == 200
except Exception as e:
    print(f"Error connecting to Bifrost: {e}")
    bifrost_ok = False

print("Bifrost:", "Online" if bifrost_ok else "Offline")

# 1. Pre-configure specialized clients with their required virtual keys
groq_client = OpenAI(
    api_key="bifrost",
    base_url=f"{BIFROST_BASE_URL}/v1",
    default_headers={"x-bf-vk": os.getenv("BIFROST_GROQ_VIRTUAL_KEY", "")}
)

mistral_client = OpenAI(
    api_key="bifrost",
    base_url=f"{BIFROST_BASE_URL}/v1",
    default_headers={"x-bf-vk": os.getenv("BIFROST_MISTRAL_VIRTUAL_KEY", "")}
)

# 2. A clean helper to extract metadata from the raw response
def execution_summary(raw_response) -> str:
    """Helper to extract the actual serving provider from Bifrost metadata."""
    try:
        meta = _json.loads(raw_response.text)
        return meta.get("extra_fields", {}).get("provider", "unknown")
    except Exception:
        return "unknown"


# Example 1: Standard Groq Call
raw = groq_client.chat.completions.with_raw_response.create(
    model=GROQ_MODEL,
    messages=[{"role": "user", "content": TEST_PROMPTS["code"]}],
    stream=True
)

# for chunk in raw.parse():
#     if chunk.choices[0].delta.content:
#         print(chunk.choices[0].delta.content, end="", flush=True)

# Example 2: Standard Mistral Call
raw = mistral_client.chat.completions.with_raw_response.create(
    model=MISTRAL_MODEL,
    messages=[{"role": "user", "content": "What is the capital of America?"}],
)
print(f"[{execution_summary(raw)}] {raw.parse().choices[0].message.content}\n")

# Example 3: Groq Call with Mistral Fallback
raw = groq_client.chat.completions.with_raw_response.create(
    model=GROQ_MODEL,
    messages=[{"role": "user", "content": "What is Langchain primarily used for?"}],
    extra_headers={"x-bifrost-fallback-models": MISTRAL_MODEL} # Inline fallback rule
)
print(f"[{execution_summary(raw)}] {raw.parse().choices[0].message.content}")

def get_virtual_key(model: str) -> str:
    """Extracts the appropriate virtual key depending on model routing path."""
    if model.startswith("groq/"):
        return BIFROST_GROQ_VIRTUAL_KEY
    elif model.startswith("mistral/"):
        return BIFROST_MISTRAL_VIRTUAL_KEY
    return ""

def bifrost_key_used(model: str, prompt: str = "Hello") -> str:
    # 1. Dynamically find the right virtual key based on the model prefix
    vk = get_virtual_key(model) 
    
    headers = {
        "Content-Type": "application/json"
    }
    if vk:
        headers["x-bf-vk"] = vk  # Attach the required Bifrost authorization key

    response = httpx.post(
        f"{BIFROST_BASE_URL}/v1/chat/completions",
        json={"model": model, "messages": [{"role": "user", "content": prompt}]},
        headers=headers,  # Ensure headers are passed!
        timeout=30,
        follow_redirects=True
    )
    
    # 2. Check for safe status before parsing
    if response.status_code != 200:
        print(f"🚨 Request failed with status {response.status_code}: {response.text}")
        return "error_status"
        
    # 3. Fixed nested dictionary lookup parentheses
    data = response.json()
    return data.get("extra_fields", {}).get("routing_info", {}).get("key", "unknown")


# Run your distribution counter
counts = Counter(bifrost_key_used(f"{GROQ_MODEL}") for _ in range(10))
print("Requests per key (10 calls):", dict(counts))


def run_observed_prompt(model: str, prompt: str, stream: bool = False):
    start_time = time.time()
    
    # 1. Fire the request using raw response to catch headers
    raw = groq_client.chat.completions.with_raw_response.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        stream=stream
    )
    
    # 2. Calculate latency and grab Bifrost metadata from headers
    latency = time.time() - start_time
    provider = raw.headers.get("x-bifrost-provider", "unknown")
    routing_key = raw.headers.get("x-bifrost-routing-key", "unknown")
    
    print(f"\n📊 [METRICS] Provider: {provider} | Key: {routing_key} | Latency: {latency:.2f}s | Stream: {stream}")
    print("💬 Response: ", end="", flush=True)
    
    # 3. Print the text content
    if stream:
        response_stream = raw.parse()
        for chunk in response_stream:
            chunk = cast(ChatCompletionChunk, chunk)
            if chunk.choices[0].delta.content:
                print(chunk.choices[0].delta.content, end="", flush=True)
    else:
        response = cast(ChatCompletion, raw.parse())
        print(response.choices[0].message.content)
    print("\n" + "-"*40)

run_observed_prompt(GROQ_MODEL, "What is 2+2?", stream=False)