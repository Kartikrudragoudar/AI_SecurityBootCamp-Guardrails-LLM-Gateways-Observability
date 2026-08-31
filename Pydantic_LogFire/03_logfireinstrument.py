from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import os, time, warnings
warnings.filterwarnings("ignore")
import logfire
from dotenv import load_dotenv

load_dotenv()
logfire.configure()
logfire.instrument_openai()

LOGFIRE_TOKEN = os.getenv("LOGFIRE_TOKEN")

llm_groq = ChatOpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY"),
    model="qwen/qwen3.8-27b",
    temperature=0.3
)

# print('Calling Groq (qwen/qwen3.8-27b)...')
# response=llm_groq.invoke([
#     HumanMessage(content="Explain what an observability 'span' is, in exactly 2 sentences.")
# ])

# print(response.content)


llm_gemini = ChatOpenAI(
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    api_key=os.getenv("GEMINI_API_KEY"),
    model="gemini-3.5-flash-lite",
    temperature=0.3
)

# print("Calling Gemini (gemini-3.5-flash-lite)...")
# try:
#     response=llm_gemini.invoke([
#         HumanMessage(content='Explain what an observability "span" is, in exactly 2 sentences.')
#     ])
#     print(f"\n Gemini Response:\n{response.content}")
# except Exception as e:
#     print(f" Gemini Call failed; {e}")
#     print("    Check your GEMINI_API_KEY in .env")


query = "What is the difference between RAG and fine? Give 3 bullet points."

with logfire.span("model_comparison", query=query, num_models=2):

    with logfire.span('groq_call', model='qwen/qwen3.8-27b', providers='groq'):
        t0 = time.time()
        r_groq = llm_groq.invoke([HumanMessage(content=query)])
        groq_ms = round((time.time() - t0) * 1000, 1)
        logfire.info("groq done", latency_ms=groq_ms, len=len(r_groq.content))

    with logfire.span('gemini_call', model='gemini-3.5-flash-lite', provider='google'):
        t0 = time.time()
        try:
            r_gemini = llm_gemini.invoke([HumanMessage(content=query)])
            gemini_ms = round((time.time() - t0) * 1000, 1)
            logfire.info("gemini_done", latency_ms=gemini_ms, answer_len=len(r_gemini.content))
            gemini_answer= r_gemini.content
        except Exception as e:
            logfire.warning('gemini_failed', error=str(e))
            gemini_ms = 0
            gemini_answer = f"[Error: {e}]"

print(f" Groq ({groq_ms}ms): \n{r_groq.content}")
print(f" Gemini ({gemini_ms}ms): \n{gemini_answer}")