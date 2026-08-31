import os
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_community.utilities import GoogleSerperAPIWrapper
from pydantic import SecretStr

from .state import AgentState
from .tools import search_document, extract_text

load_dotenv()


def _safe_secret(name: str):
    value = os.getenv(name)
    return SecretStr(value) if value else None


## LLM Clients
_groq = ChatGroq(
    model="qwen/qwen3.6-27b",
    api_key=_safe_secret("GROQ_API_KEY"),
)

_gemini = ChatOpenAI(
    base_url="https://generativelanguage.googleapis.com/v1beta/models",
    api_key=_safe_secret("GEMINI_API_KEY"),
    model="gemini-2.0-flash",
    temperature=0.3,
)

# Search tool
_serper = GoogleSerperAPIWrapper(serper_api_key=os.getenv("SERPER_API_KEY"))


def planner(state: AgentState) -> dict:
    """Rewrites the user question for clarity and precision."""
    response = _groq.invoke([
        SystemMessage(content=(
            "You are a research question refiner. "
            "Rewrite the user question into ONE concise search query. "
            "Return ONLY the rewritten query text. Do NOT include explanations, thinking, or quotes."
        )),
        HumanMessage(content=state["question"])
    ])
    cleaned_query = extract_text(response).strip().strip('"').strip("'")
    # Take only the last non-empty line in case there is preamble
    lines = [line.strip() for line in cleaned_query.splitlines() if line.strip()]
    final_query = lines[-1] if lines else state["question"]

    return {
        "refined_question": final_query,
        "steps_taken": state.get("steps_taken", []) + ["planner"],
    }


def document_reader(state: AgentState) -> dict:
    """Searches the local knowledge-base document for relevant sections."""
    sections = search_document(state["refined_question"], top_k=3)
    return {
        "doc_sections": sections,
        "steps_taken": state.get("steps_taken", []) + ["document_reader"],
    }


def web_enricher(state: AgentState) -> dict:
    """Fetches the latest information from the web using Google Serper."""
    raw_query = state.get("refined_question", "").strip()
    cleaned_query = " ".join(raw_query.split())[:250]

    try:
        if not cleaned_query:
            web_text = "No query provided for web search."
        else:
            web_text = _serper.run(cleaned_query)
    except Exception as exc:
        web_text = f"[Web search unavailable: {exc}]"

    return {
        "web_results": web_text,
        "steps_taken": state.get("steps_taken", []) + ["web_enricher"],
    }


def synthesizer(state: AgentState) -> dict:
    """Combines document knowledge and web results into a coherent analysis."""
    doc_context = "\n\n---\n\n".join(state["doc_sections"])
    synthesis = _groq.invoke([
        SystemMessage(content=(
            "You are a research synthesizer. Given knowledge from a document and "
            "from the web, combine both into a clear, structured analysis. "
            "Cite sources where possible. Use markdown formatting."
        )),
        HumanMessage(content=(
            f"Question: {state['refined_question']}\n\n"
            f"=== DOCUMENT KNOWLEDGE ===\n{doc_context}\n\n"
            f"=== WEB SEARCH RESULTS ===\n{state['web_results']}"
        ))
    ])
    return {
        "synthesis": extract_text(synthesis),
        "steps_taken": state.get("steps_taken", []) + ["synthesizer"],
    }


def report_writer(state: AgentState) -> dict:
    """Formats the synthesis into a polished final report using Gemini."""
    try:
        report = _gemini.invoke([
            SystemMessage(content=(
                """
                    You are a technical report writer. Format the given analysis into
                    a clean, well-structured report with: a one-sentence TL;DR at the top,
                    key findings as bullet points, and a brief conclusion. Keep it under 400 words.
                """
            )),
            HumanMessage(content=state.get('synthesis', ''))
        ])
        final = extract_text(report)
    except Exception:
        final = state.get('synthesis', '')

    return {
        "final_report": final,
        "steps_taken": state.get("steps_taken", []) + ["report_writer"],
    }