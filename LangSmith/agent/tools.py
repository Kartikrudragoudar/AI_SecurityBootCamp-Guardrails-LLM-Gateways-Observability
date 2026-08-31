"""Utility tools and helpers used by agent nodes"""
import os
import re
from typing import List, Any


__DOC__PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "Data", "llm_production_guide.txt"))


def extract_text(resp: Any) -> str:
    """Return clean text/content from LLM response, stripping thinking tokens and reasoning tags."""
    text = getattr(resp, "text", None) or getattr(resp, "content", None) or str(resp)
    if isinstance(text, list):
        text = "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in text)
    # Remove <think>...</think> reasoning blocks from Qwen/DeepSeek
    return re.sub(r"<think>.*?</think>", "", str(text), flags=re.DOTALL).strip()


def load_document() -> str:
    """Load the full knowledge-base text file"""
    with open(__DOC__PATH, encoding="utf-8") as f:
        return f.read()


def search_document(query: str, top_k: int = 3) -> List[str]:
    """
    Simple keyword search over the text document.
    Splits into paragraphs, scores by keyword overlap, returns top_k.
    No Vector DB needed - keeps the demo focused on Langsmith tracing.
    """
    doc = load_document()
    paragraphs = [p.strip() for p in doc.split("\n\n") if len(p.strip())]
    keywords = set(re.findall(r"\b\w{4,}\b", query.lower()))

    def score(para: str) -> int:
        text_lower = para.lower()
        return sum(1 for kw in keywords if kw in text_lower)

    ranked = sorted(paragraphs, key=score, reverse=True)
    return ranked[:top_k]