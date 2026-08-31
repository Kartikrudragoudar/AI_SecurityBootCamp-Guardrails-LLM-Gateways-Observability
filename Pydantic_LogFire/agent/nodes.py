import logfire
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_tavily import TavilySearch
from dotenv import load_dotenv
from .state import AgentState
from .llms import groq_llm
load_dotenv()

# setup model and websearch class
_groq = groq_llm()
_tavily = TavilySearch(max_results=4)

# ── Prompts ───────────────────────────────────────────────────────────────

_PLANNER_PROMPT = """Classify the user query into EXACTLY one word from this list:
- explain  → "what is", "how does", "why", "tell me about", "describe"
- analyze  → "compare", "pros and cons", "evaluate", "difference", "vs", "tradeoffs"
- create   → "write", "generate", "brainstorm", "create", "ideas for", "draft"
- search   → "latest", "current", "news", "today", "recent", "2024", "2025", "search for"

Reply with ONLY one word. No punctuation, no explanation."""

_EXPLAINER_PROMPT = """You are an expert teacher.
Explain the topic clearly using simple language, real-world analogies, and bullet points.
Be concise — aim for 150-250 words."""

_ANALYST_PROMPT = """You are a sharp analytical expert.
Provide structured analysis: key points, pros/cons or comparisons, and a clear recommendation.
Use markdown headers and bullet points. Aim for 200-300 words."""

_CREATOR_PROMPT = """You are a creative content specialist.
Generate fresh, specific, and actionable ideas or content.
Format as a numbered list with brief explanations. Aim for 150-250 words."""

_FORMATTER_PROMPT = """You are a formatting assistant.
Lightly polish the response below for clarity and readability.
Do NOT change the content or add new information — only improve structure.
Keep the same approximate length."""

def planner(state: AgentState) -> dict:
    with logfire.span("planner", 
                        question=state["question"],
                        session_id=state.get("session_id", "")):
        response = _groq.invoke([
            SystemMessage(content=_PLANNER_PROMPT),
            HumanMessage(content=state['question'])
        ])
        intent=response.text.strip().lower().split()[0]
        if intent not in ("explain", "analyze", "create", "search"):
            intent = "explain"

        logfire.info("intent_classified",
                     intent=intent,
                     question=state["question"])

        return {
            "intent":intent,
            "node_path":state.get("node_path", []) + ["planner"],
        }

def explainer(state:AgentState) -> dict:
    with logfire.span("explainer", 
                      question=state["question"],
                      model="qwen/qwen3.8-27b"):
            response = _groq.invoke([
                 SystemMessage(content=_EXPLAINER_PROMPT),
                 HumanMessage(content=state["question"]),
            ])
            logfire.info("explanation_ready",
                         answer_length=len(response.text))

            return {
                 "specialist_output" : response.text,
                 "model_used": "qwen/qwen3.8-27b",
                 "node_path": state.get("node_path", []) + ["explainer"],
            }


def analyst(state:AgentState) -> dict:
    with logfire.span("explainer", 
                      question=state["question"],
                      model="qwen/qwen3.8-27b"):
            response = _groq.invoke([
                 SystemMessage(content=_ANALYST_PROMPT),
                 HumanMessage(content=state["question"]),
            ])
            logfire.info("explanation_ready",
                         answer_length=len(response.text))

            return {
                 "specialist_output" : response.text,
                 "model_used": "qwen/qwen3.8-27b",
                 "node_path": state.get("node_path", []) + ["analyst"],
            }

def creator(state:AgentState) -> dict:
    with logfire.span("explainer", 
                      question=state["question"],
                      model="qwen/qwen3.8-27b"):
            response = _groq.invoke([
                 SystemMessage(content=_CREATOR_PROMPT),
                 HumanMessage(content=state["question"]),
            ])
            logfire.info("explanation_ready",
                         answer_length=len(response.text))

            return {
                 "specialist_output" : response.text,
                 "model_used": "qwen/qwen3.8-27b",
                 "node_path": state.get("node_path", []) + ["creater"],
            }

def web_searcher(state:AgentState) -> dict:
     with logfire.span("web_searcher",
                       query=state["question"]):

        with logfire.span("tavily_search", query=state["question"]):
            _raw=_tavily.invoke(state["question"])
            sources = []
            # TavilySearch may return a list[dict] or a plan string
            if isinstance(_raw, list):
                results = _raw
                raw = "\n\n".join(
                     f"[{i+1} {r.get('url', '')}\n{r.get('content', '')}]"
                     for i, r in enumerate(results)
                )
                sources = [r.get("url", "") for r in results]
            else:
                results = []
                raw = str(_raw)
                sources=[]
            logfire.info("search_complete",
                        num_results=len(results),
                        sources=sources)

        with logfire.span("synthesize_results",
                          model="qwen/qwen3.8-27b"):
            summary = _groq.invoke([
                SystemMessage(content=(
                    "Synthesize the search results into a clear, concise answer. "
                    "Include key facts and mention sources where relevant."
                )),
                HumanMessage(content=(
                    f"Question: {state['question']}\n\n"
                    f"Search Results:\n{raw}"
                )),
            ])
            logfire.info("synthesis_ready",
                        answer_length=len(summary.text))

        return {
            "specialist_output": summary.text,
            "search_results":raw,
            "model_used": "tavily + qwen/qwen3.8-27b",
            "node_path": state.get("node_path", []) + ["web_searcher"]
        }

def formatter(state:AgentState) -> dict:
    with logfire.span("formatter",
                       intent=state["intent"],
                       model="qwen/qwen3.8-27b"):
        response = _groq.invoke([
            SystemMessage(content=_FORMATTER_PROMPT),
            HumanMessage(content=state['specialist_output']),
        ])
        final = response.text
        model_tag = state.get("model_used", "") + " -> qwen/qwen3.8-27b (formatter)"

        logfire.info("formatter_done",
                    input_length=len(state["specialist_output"]),
                    output_length=len(final))
        return {
            "final_answer":final,
            "model_used": model_tag,
            "node_path": state.get("node_path", []) + ["formatter"],
        }     