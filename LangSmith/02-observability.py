import os, warnings
warnings.filterwarnings("ignore")
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import re
from langsmith import traceable
load_dotenv()

api_key=os.getenv("GROQ_API_KEY")
langsmith_tracing=os.getenv("LANGSMITH_TRACING")
langsmith_api_key=os.getenv("LANGSMITH_API_KEY")
langsmith_project=os.getenv("LANGSMITH_PROJECT")

llm = ChatGroq(model="openai/gpt-oss-120b", max_tokens=512)

@traceable(run_type="tool", name="doc_keyword_search")
def search_document(query: str, top_k: int = 3) -> list[str]:
    """Searches llm_production_guide.txt by keyword overlap. Visible as a tool run."""
    with open("/home/kartik-rudragoudar/Kartik_DOC/Python_Programs/learnings/Data/llm_production_guide.txt", "r") as f:
        text = f.read()

    paragraphs = [p.strip() for p in text.split("\n\n") if len (p.strip()) > 80]
    keywords = set(re.findall(r"\b\w{4,}\b", query.lower()))
    ranked = sorted(paragraphs,
                key=lambda p: sum(1 for kw in keywords if kw in p.lower()), reverse=True)
    return ranked[:top_k]

@traceable(run_type="chain", name="doc_qa_pipeline")
def doc_qa_pipeline(question: str) -> str:
    """Parent chain. LangSmith shows: doc_qa_pipeline -> doc_keyword_search + Chatgroq."""
    sections = search_document(question)
    context = "\n\n".join(sections)
    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer concisely:"
    response = llm.invoke(prompt)
    return response.text

answer = doc_qa_pipeline("What are the main LLM Security threats?")
print(f"Answer: {answer}")