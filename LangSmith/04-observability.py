import os, warnings
warnings.filterwarnings("ignore")
from langsmith import get_current_run_tree
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langsmith import traceable
load_dotenv()

embedding_key=os.getenv("GEMINI_API_KEY")
llm_api_key=os.getenv("GROQ_API_KEY")
langsmith_tracing=os.getenv("LANGSMITH_TRACING")
langsmith_api_key=os.getenv("LANGSMITH_API_KEY")
langsmith_project=os.getenv("LANGSMITH_PROJECT")

llm = ChatGroq(model="openai/gpt-oss-120b", api_key=llm_api_key, max_tokens=512)

# --LOAD + Split the real guide document
loader = TextLoader("/home/kartik-rudragoudar/Kartik_DOC/Python_Programs/learnings/Data/llm_production_guide.txt", encoding="utf-8")
raw_docs = loader.load()
print(f"Loaded : {len(raw_docs[0].page_content):,}")


# SPLIT
splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80)
chunks = splitter.split_documents(raw_docs)
print(f"Chunks: {len(chunks)} (avg {sum(len(c.page_content) for c in chunks)//len(chunks)})")

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001", api_key=embedding_key)
vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k":3})
print("FAISS index ready")


@traceable(run_type="chain", name="production_guide_rag")
def rag(question: str, user_id: str = "anonymous") -> str:
    docs = retriever.invoke(question)
    context = "\n\n".join(f"[chunk {i+1}] {d.page_content}" for i, d in enumerate(docs))
    prompt = (
        f"Answer based ONLY on the context below.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\nAnswer concisely:"
    )
    run = get_current_run_tree()
    if run:
        run.metadata.update({"user_id": user_id, "chunks_retrieved": len(docs)})
    return llm.invoke(prompt).text

for q, uid in [
    ("What are the main LLM Security risks in production?", "student__001"),
    ("How should we evaluate LLM Outputs for quality?", "student__002"),
]:
    answer = rag(q, user_id=uid)
    print(f"\nQ: {q}")
    print(f"A: {answer[:250]}...")