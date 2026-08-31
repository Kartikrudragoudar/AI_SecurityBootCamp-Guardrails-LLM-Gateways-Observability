import os, warnings
warnings.filterwarnings("ignore")
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()

api_key=os.getenv("GROQ_API_KEY")
langsmith_tracing=os.getenv("LANGSMITH_TRACING")
langsmith_api_key=os.getenv("LANGSMITH_API_KEY")
langsmith_project=os.getenv("LANGSMITH_PROJECT")

llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.7, max_tokens=512)
response = llm.invoke("What is the a langsmith run? Answer in 2 sentences only.")
print(response.content)


