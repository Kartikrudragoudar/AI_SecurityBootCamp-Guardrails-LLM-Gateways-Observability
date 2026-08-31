import os, time, warnings
warnings.filterwarnings("ignore")
import logfire
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import Optional
load_dotenv()

LOGFIRE_TOKEN = os.getenv("LOGFIRE_TOKEN")
GROQ_API_KEY=os.getenv("GROQ_API_KEY")
GEMINI_API_KEY=os.getenv("GEMINI_API_KEY")

logfire.configure()

class LLMRequest(BaseModel):
    user_id: str
    session_id: str
    query: str
    model: str
    temperature:  float = 0.7
    max_tokens: Optional[int] = None

class LLMResponse(BaseModel):
    answer: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    model_used:str


request = LLMRequest(
    user_id='priya',
    session_id='sess_abc123',
    query='what is reterival-agumented generation?',
    model='qwen/qwen3.8-27b',
    max_tokens=500
)

with logfire.span('llm_CALL', user_id=request.user_id, session_id=request.session_id, model_used=request.model):
    logfire.info('request_recevied', **request.model_dump())
    time.sleep(0.1)

    response = LLMResponse(answer='RAG is a technique that retrieves relevant documents...', 
                           input_tokens=18,
                           output_tokens=120,
                           latency_ms=342.5,
                           model_used='qwen/qwen3.8-27b')
    logfire.info('response_sent', **response.model_dump())

print(response)




