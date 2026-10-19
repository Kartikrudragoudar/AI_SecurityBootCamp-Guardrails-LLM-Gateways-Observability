import os
import time
import json
from dotenv import load_dotenv
from portkey_ai import Portkey, createHeaders, PORTKEY_GATEWAY_URL


load_dotenv()

PORTKEY_API_KEY = os.getenv("PORTKEY_API_KEY")
portkey = Portkey(api_key=PORTKEY_API_KEY)


GROQ_SLUG = "gkey"
GROQ_MODEL = f"@{GROQ_SLUG}/openai/gpt-oss-20b"

def section(title):
    print(f"\n{"="*62}")
    print(f" {title}")
    print(f"{"="*62}")

section("HELLO")

def show(q, answer, ms, label=""):
    bar = chr(9472) * 62
    print(f"\n{bar}")
    print(f"Q: {q}")
    print(f"A: {answer[:260]}{'...' if len(answer) > 260 else ''}")
    note = f" |  {label}" if label else ""
    print(f" {ms:.0f}ms{note}")
    print(bar)

show("Why is the SKy blue", "nature", 5, "GOD")

questions = [
    "What is AI and Gen AI?",
    "What is latest new in the ai."
]

for q in questions:
    t0 = time.time()
    r = portkey.chat.completions.create(
        model = GROQ_MODEL,
        messages = [{"role": "user", "content": q}],
        stream = False
    )
    choices = getattr(r, "choices", []) or []
    message = choices[0].message if choices else None
    answer = message.content if message is not None else ""
    show(q, answer or "", (time.time()-t0)*1000,
         label=f"routed via Portkey gateway")

print("\n Check portkey.ai for --> Logs to see both requests fully logged!")
print(" Token count, cost, latency, --- all tracked. Zero extra code. ")