import os
import time
import uuid
from dotenv import load_dotenv
from portkey_ai import Portkey


load_dotenv()

PORTKEY_API_KEY = os.getenv("PORTKEY_API_KEY")
portkey = Portkey(api_key=PORTKEY_API_KEY)


GROQ_SLUG = "gkey"
GROQ_MODEL = f"@{GROQ_SLUG}/openai/gpt-oss-20b"

session = str(uuid.uuid4())[:8]

scenarios = [
    ("alice", "enterprise-rag", "What is Kubernetes RBAC?"),
    ("bob", "docs-chatbot", "How does BGP path selection work?"),
    ("carol", "enterprise-rag", "What is the difference between a pod and a container?"),
    ("alice", "support-bot", "What is SRIOV virtulization?"),    
]

for user, feature, q in scenarios:
    t0 = time.time()
    r = portkey.with_options(
        metadata={
            "_user": user,
            "session_id": session,
            "feature":feature,
            "environment":"notebook"
        }
    ).chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": q}],
    )
    ms = (time.time() - t0) * 1000
    print(f"\n {user:8s} | {feature:18s} | {ms:.0f}ms")
    print(f" Q: {q}") 