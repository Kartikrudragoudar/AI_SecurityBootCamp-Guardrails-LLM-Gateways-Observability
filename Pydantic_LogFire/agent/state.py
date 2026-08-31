from typing import TypedDict, List

class AgentState(TypedDict):
    question: str
    intent: str
    specialist_output: str
    search_results: str
    final_answer: str
    model_used: str
    session_id: str
    node_path: List[str]