"""
App-wide constants - models, prompts, and output-scan patterns.
"""

GROQ_MODELS = {
    "qwen/qwen3.8-27b":" Qwen * 27b -- Fast",
    "openai/gpt-oss-safeguard-20b":"Open AI OSS Safeguard * Recommended for guardrails",
    "openai/gpt-oss-20b":"OpenAI OSS * 20B -- Fast & Cheap",
    "openai/gpt-oss-120b":"OpenAI OSS * 120B -- Strong Reasoning"
}

GURAD_MODEL_DEFAULT = "openai/gpt-oss-20b"
CHAT_MODEL_DEFAULT = "openai/gpt-oss-120b"


HR_SYSTEM_PROMPT = (
    "You are the HR Policy Assistant for Acme Corp. "
    "Answer the employe's question using ONLY the policy excerpts provided below."
    "Be Concise, cite the relevant policy section, and include specific numbers or rules where applicable."
    "If the answer is not covered in the excerpts, say so clearly and direct the employee to hr@acmecorp.com.\n\n"
    "Policy Excerpts:\n{context}"
)

SENSITIVE_OUTPUT_PATTERNS = {
    "credential_leak":r"(?i)(password|passwd|secret|api[_\-]?key|token)\s*[:=]\s*['\"]?\w{6,}",
    "ssn_in_output":r"\b\d{3}-\d{2}-\d{4}\b",
    "hardcoded_salary": r"(?i)\b(earns?|is paid|salary of)\s+\$[\d,]+\b",
}