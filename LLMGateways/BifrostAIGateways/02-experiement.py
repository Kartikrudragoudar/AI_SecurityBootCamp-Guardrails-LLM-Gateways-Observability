from dotenv import load_dotenv
import os
import httpx
from openai import OpenAI
load_dotenv()
import json as _json
import time

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
BIFROST_BASE_URL = os.getenv("BIFROST_BASE_URL", "http://localhost:8080")
BIFROST_GROQ_VIRTUAL_KEY = os.getenv("BIFROST_GROQ_VIRTUAL_KEY", "")
BIFROST_MISTRAL_VIRTUAL_KEY = os.getenv("BIFROST_MISTRAL_VIRTUAL_KEY", "")
BIFROST_SETUP_TOKEN = os.getenv("BIFROST_SETUP_TOKEN", "mySecretToken123")

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

def call_with_virtual_key(virtual_key: str, model:str, prompt:str):
    if not virtual_key:
       print("No virtual key set for this provider - create one at http://localhost:8080")
       return
    client = OpenAI(api_key=virtual_key, base_url=f"{BIFROST_BASE_URL}/v1")
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role":"user", "content":prompt}],
        )
        print(response.choices[0].message.content)
    except Exception as e:
        print("Call failed - check the virtual key is valid and not over budget:", e)


# call_with_virtual_key(BIFROST_GROQ_VIRTUAL_KEY, f"{GROQ_MODEL}", TEST_PROMPTS['simple'])
# call_with_virtual_key(BIFROST_MISTRAL_VIRTUAL_KEY, f"{MISTRAL_MODEL}", TEST_PROMPTS['simple'])

# REGISTER MCP TO BIFROST
headers = {"X-Bifrost-Setup-Token": BIFROST_SETUP_TOKEN}
response = httpx.get(f"{BIFROST_BASE_URL.rstrip('/')}/api/mcp/clients", headers=headers, timeout=5)
data = response.json()
clients = data.get("clients", []) if isinstance(data, dict) else data
print(f"Found {len(clients)} MCP clients:")
for client in clients:
    name = client.get("config", {}).get("name") or client.get("name")
    tools = [t.get("name") for t in client.get("tools", [])]
    print(f" - {name} ({len(tools)} tools: {', '.join(tools)})")


def call_with_tools(prompt: str, servers: list[str]) -> dict:
    headers = {"x-bf-vk": BIFROST_MISTRAL_VIRTUAL_KEY}
    response = httpx.post(
        f"{BIFROST_BASE_URL.rstrip('/')}/v1/chat/completions",
        headers=headers,
        json={
            "model": f"{MISTRAL_MODEL}",
            "messages": [{"role": "user", "content": prompt}],
            "bifrost": {"mcp": {"enabled": True, "code_mode": True, "servers": servers}},
        },
        timeout=60,
    )
    if response.status_code != 200:
        raise RuntimeError(f"Bifrost error {response.status_code}: {response.text}")
    return response.json()["choices"][0]["message"]

message = call_with_tools(TEST_PROMPTS["deepwiki"], ["deepwiki"])
print(message.get("content") or message.get("tool_calls"))

message = call_with_tools(TEST_PROMPTS["tavily"], ["tavily"])
print(message.get("content") or message.get("tool_calls"))