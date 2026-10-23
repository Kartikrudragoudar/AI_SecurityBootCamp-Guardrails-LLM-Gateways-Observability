from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
load_dotenv()

TEXT_TO_SUMMARIZE = """
Artificial intelligence has experienced remarkable growth over the past decade,
fundamentally changing how software systems are built and deployed. Large Language
Models (LLMs) like GPT-4 and Llama have demonstrated capabilities ranging from
writing code to complex reasoning. However, managing these models in production
presents new challenges: teams need to monitor performance, control costs, run
experiments, and continuously improve their AI features. LLM gateways have emerged
as a solution, acting as an intelligent middleware layer between applications and
the underlying model providers.
"""

print("Sending request Directly to Gemini")


llm = ChatOpenAI(
    model="tensorzero::function_name::summarize",
    base_url="http://localhost:3000/openai/v1",
    api_key="local-gateway", # Placeholder; TensorZero handles provider credentials
)

response = llm.invoke("Tell me the one good joke of the decade?")
print(response.content)