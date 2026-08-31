import json
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage
from langchain.tools import tool
from langchain.agents import create_agent
import os, warnings
warnings.filterwarnings("ignore")
import logfire
from dotenv import load_dotenv

load_dotenv()
logfire.configure()
logfire.instrument_openai()

with open("/home/kartik-rudragoudar/Kartik_DOC/Python_Programs/learnings/Data/documents.json") as f:
    raw_docs = json.load(f)

DOCS = [
    Document(page_content=d["content"], metadata={"topic": d["topic"], "source":d["source"]})
    for d in raw_docs
]


embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    api_key=os.getenv("GEMINI_API_KEY")
)

llm_groq = ChatOpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY"),
    model="qwen/qwen3.8-27b",
    temperature=0.3
)

vectorstore = FAISS.from_documents(DOCS, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k":2})

@tool
def search_knowledge_base(query:str)-> str:
    """Search the knowledge base for LLM production topics: RAG, guardrails,
    gateways, observability, evaluations, and fine-tuning"""
    docs = vectorstore.similarity_search(query, k=2)
    return "\n\n".join(
        f"[{d.metadata['topic']}] {d.page_content}" for d in docs
    )

agent = create_agent(
    model=llm_groq,
    tools=[search_knowledge_base],
    system_prompt=(
        """You are a helpful assistant. Use search_knowledge_base for any question
            about LLM production topics. Answer directly for general knowledge questions.
        """
    )
)

def run_agent(question: str, user_id:str="anonymous"):
    with logfire.span("agent_run", question=question, user_id=user_id):
        result = agent.invoke({"messages": [HumanMessage(content=question)]})

        last_ai = next(
            (m for m in reversed(result["messages"]) if isinstance(m, AIMessage)),
            None,
        )
        answer = last_ai.text if last_ai else ""
        used_tool = any(isinstance(m, ToolMessage) for m in result['messages'])

        logfire.info("agent_done", used_tool=used_tool, answer_length=len(answer))
        return answer, used_tool

queries = [
    ("What is LLM observability and which tools provide it?", "priya"),
    ("How do LLM guardrails work?", "bhavesh"),
    ("What is the capital of France?", 'kunal'),
]

for q, uid in queries:
    print(f"{'='*55}")
    answer, used_tool = run_agent(q,user_id=uid)
    print(f"Q: {q}")
    print(f"Tool used: {used_tool} {"<- retrieved from KB" if used_tool else '<- answered directly'}")
    print(f"A: {answer[:300]}")
