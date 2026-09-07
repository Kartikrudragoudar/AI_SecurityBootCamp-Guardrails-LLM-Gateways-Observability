import os, warnings
warnings.filterwarnings("ignore")
import nest_asyncio
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
from pydantic import SecretStr
load_dotenv()

nest_asyncio.apply()

GROQ_API_KEY=SecretStr(os.getenv("GROQ_API_KEY", ""))
NVIDIA_API_KEY=SecretStr(os.getenv("NVIDIA_API_KEY",""))
GEMINI_API_KEY=SecretStr(os.getenv("GEMINI_API_KEY",""))


# config = colang examples + model

llm_groq = ChatOpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=GROQ_API_KEY, 
    model="qwen/qwen3.8-27b",
    temperature=0.3
)

guard_llm = ChatOpenAI(
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    api_key=GEMINI_API_KEY,
    model="gemini-3.5-flash-lite",
    temperature=0.3
)


def section(title):
    print(f"\n{'='*62}")
    print(f" {title}")
    print(f"{'='*62}")

# def chat(rails, message):
#     """Send a message through the rails and print input + output."""
#     print(f"\n{'--'*62}")
#     print(f'User : {message}')
#     response = rails.generate(messages=[{"role":"user", "content":message}])
#     content = response.get("content", str(response)) if isinstance(response, dict) else response
#     print(f"Bot : {content}")
#     print(f"{'--'*62}")
#     return response

# chat("hey I'm a mentor", "Welcome to guardrails")

SYSTEM_PROMPT="""You are an Enterprise IT Assistant specialising in marketting hardware, and enterprise networking.
Don't Entertain off topic questions"""

def raw_chat(message):
    """Chat directly with GROQ with zero guardrails."""
    msgs = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=message)]
    response=llm_groq.invoke(msgs)
    print(f"\n{'--'*62}")
    print(f"User : {message}")
    print(f'Raw : {response.text}')
    print(f'{"--"*62}')

raw_chat("What are common Linux exploits")