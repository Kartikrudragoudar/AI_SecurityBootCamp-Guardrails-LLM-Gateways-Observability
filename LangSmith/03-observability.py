from langsmith import get_current_run_tree
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

llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.7, max_tokens=512)

@traceable(run_type="chain", name="support-query")
def support_qa(question: str, user_id: str, session_id: str) -> str:
    run = get_current_run_tree()
    if run:
        run.metadata.update({
            "user_id": user_id,
            "session_id": session_id,
            "feature": "customer-support",
            "env": "production"
        })
        run.tags = ["production", "support-bot", "groq"]
    return llm.invoke(question).text

list_of_queriers = [
    ("priya", "sess_001", "What is prompt injection and how do we prevent it?"),
    ("aditi", "sess_002", "What are the main LLM Security threats?"),
    ("kartik", "sess_003", "What are the best practices for LLM observability?"),
]

for user, session, q in list_of_queriers:
    answer = support_qa(q, user, session)
    print(f"Answer for {user} (session: {session}): {answer}")