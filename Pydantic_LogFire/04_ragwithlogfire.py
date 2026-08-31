import json
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
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

print(f"loaded {len(DOCS)} documents: {[d.metadata["topic"] for d in DOCS]}")

llm_groq = ChatOpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY"),
    model="qwen/qwen3.8-27b",
    temperature=0.3
)

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    api_key=os.getenv("GEMINI_API_KEY")
)

vectorstore = FAISS.from_documents(DOCS, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k":2})
print("FAISS index ready")


def rag(question: str, user_id: str='anonymous') -> str:
    with logfire.span('rag_pipeline', question=question, user_id=user_id):
        docs = retriever.invoke(question)
        logfire.info("docs_retrieved",
                     topics=[d.metadata['topic'] for d in docs],
                     nums_docs=len(docs))
        context = "\n\n".join(
            f"[{d.metadata['topic']}] {d.page_content}" for d in docs
        )
        prompt = (
            f"Answer the question based only on the context below. \n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\nAnswer concisely:"
        )
        return llm_groq.invoke(prompt).text

answer = rag("How does a RAG reduce hallucination", user_id="student1")
print(f"\nA: {answer}")