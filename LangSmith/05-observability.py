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
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage
from langchain_community.utilities import GoogleSerperAPIWrapper
load_dotenv()

embedding_key=os.getenv("GEMINI_API_KEY")
llm_api_key=os.getenv("GROQ_API_KEY")
langsmith_tracing=os.getenv("LANGSMITH_TRACING")
langsmith_api_key=os.getenv("LANGSMITH_API_KEY")
langsmith_project=os.getenv("LANGSMITH_PROJECT")

llm = ChatGroq(model="openai/gpt-oss-120b", api_key=llm_api_key, max_tokens=512)
serper = GoogleSerperAPIWrapper(serper_api_key=os.getenv("SERPER_API_KEY"))


loader = TextLoader("/home/kartik-rudragoudar/Kartik_DOC/Python_Programs/learnings/Data/llm_production_guide.txt", encoding="utf-8")
raw_docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80)
chunks = splitter.split_documents(raw_docs)

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001", api_key=embedding_key)
vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k":3})

@tool
def search_local_docs(query: str)-> str:
    """
    Search the internal LLM production guide.

    IMPORTANT:
    - Use only once per question.
    - After receiving results, answer the user.
    - Do not call repeatedly.
    """

    docs = vectorstore.similarity_search(query, k=3)

    if not docs:
        return "No relevant documents found."

    response = "\n\n".join(
        f"[Chunk {i+1}\n{doc.page_content[:700]}...]" for i, doc in enumerate(docs)
    )

    return response[:2500]

@tool
def google_search(query: str) -> str:
    """
    Search the web for recent information.

    IMPORTANT:
    - Use only once per question.
    - After receiving results, answer the user.
    - Do not search again unless absolutely required.
    """

    try:
        result = serper.run(query)

        if not result:
            return "No search results found."

        return str(result)[:2500]

    except Exception as e:
        return f"Search failed: {str(e)}"

agent = create_agent(
    model=llm,
    tools=[search_local_docs, google_search],
    system_prompt="""
    You are a research assistant.
    
    You have two tools:

    1. search_local_docs
    - Use for RAG, security, evaluation, monitoring,
      prompt engineering, guardrails, deployment.
    
    2. google_search
    - Use for current events, news,
      regulations, recent AI developments.
    
    Rules:

    1. Call a tool ONLY if needed.
    2. Never call the same tool more than once.
    3. Maximum TWO total tool calls.
    4. After receiving tool results, provide the final answer.
    5. Do NOT continue searching if enough information exists.
    6. Do NOT loop.
    7. If one tool gives sufficient information,
       answer immediately.
    
    """,
)

def run_agent(question:str):
    result = agent.invoke(
        {
            "messages": [
                HumanMessage(content=question)
            ]
        },
        config={"recursion_limit":10}
    )

    tools_used = []

    for msg in result["messages"]:
        if isinstance(msg, ToolMessage):
            tools_used.append(msg.name)

    final_answer = ""

    for msg in reversed(result["messages"]):
        if isinstance(msg, AIMessage):

            final_answer = msg.content
            break

    return final_answer, list(dict.fromkeys(tools_used))

queries = [
    (
        "What are LLM prompt injection attacks and how do we defend against them?",
        "search_local_docs"
    ),
    (
        "What are the latest AI regulations passed in 2025?",
        "google_search"
    ),
    (
        "How does RAG work and what are the latest open-source RAG frameworks in 2025?",
        "both"
    )
]

for question, expected in queries:
    print("\n" + "=" * 80)
    print("QUESTION:")
    print(question)

    print("\nEXPECTED:")
    print(expected)

    answer, tools = run_agent(question)

    print("\nTOOLS USED:")
    print(tools)

    print("\nANSWER")
    print(answer[:500])

print("\n Completed Successfully")