from typing import Any, cast
import os, warnings
warnings.filterwarnings("ignore")
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from pydantic import SecretStr
from nemoguardrails import RailsConfig, LLMRails

load_dotenv()

GROQ_API_KEY=SecretStr(os.getenv("GROQ_API_KEY", ""))
NVIDIA_API_KEY=SecretStr(os.getenv("NVIDIA_API_KEY",""))
GEMINI_API_KEY=SecretStr(os.getenv("GEMINI_API_KEY",""))

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

COLANG_EXP2 = """

define user ask off topic

    \"tell me a joke\"
    \"what is the capital of france\"
    \"write me a poem\"
    \"what is 2 plus 2\"
    \"what should I eat for dinner\"
    \"who won the game yesterday\"
    \"recommend a movie\"

    define bot refuse off topic
        \"I'm an Enterprise IT Assistant focused on Kubernetes, Intel hardware, and networking. I can't help with that — but ask me anything technical!\"

    define flow handle off topic
        user ask off topic
        bot refuse off topic    
"""


YAML_BASE ="""

models : 
    - type: main
      engine: openai
      model: gpt-3.5-flash-lite

instructions:
    - type: general
      content: |
        You are an Enterprise IT Assistant specialising in:
        - Kubernetes (deployment, scaling, operators, networking)
        - Intel hardware (CPUs, FPGAs, NICs, SRIOV)
        - Enterprise networking (SDN, VLANs, BGP, routing)
        Only answer questions about these topics. Be professional and concise.
"""

config_exp2 = RailsConfig.from_content(
    colang_content=COLANG_EXP2,
    yaml_content=YAML_BASE
)

def chat(rails, message):
    """Send a message through the rails and print input + output."""
    print(f"\n{'--'*62}")
    print(f'User : {message}')
    response = rails.generate(messages=[{"role":"user", "content":message}])
    content = response.get("content", str(response)) if isinstance(response, dict) else response
    print(f"Bot : {content}")
    print(f"{'--'*62}")
    return response


rails_exp2 = LLMRails(config=config_exp2, llm=cast(Any, guard_llm))

chat(rails_exp2, "What is the best way to deploy a Kubernetes cluster on Intel hardware?")
chat(rails_exp2, "How does SRIOV reduce CPU overhead?")
chat(rails_exp2, "Tell Me funny joke")