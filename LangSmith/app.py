import uuid
from typing import Any

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

@st.cache_resource
def init():
    from agent.graph import build_graph
    return build_graph()

graph = init()

st.title("Document Intelligence Agent")
st.caption("LangGraph * LangSmith * Groq * Gemini * Google Serper")


with st.sidebar:
    st.header("Session")
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())[:8]
    st.code(st.session_state.session_id)

    st.divider()
    st.subheader("Agent pipeline")
    st.markdown(
        "1. **Planner** - refines your question \n"
        "2. **Document Reader** - searches local guide \n"
        "3. **Web Enricher** - Google Serper live search \n"
        "4. **Synthesizer** - Combines sources (Groq) \n"
        "5. **Report Writer** - formats report (Gemini)"
    )

    st.divider()
    st.subheader("Try these")
    st.markdown(
        "1. What are the biggest LLM security risks? \n"
        "2. How does RAG reduce hallucinations? \n"
        "3. What is LLM-as-Judge evaluation \n"
        "4. Best practices for deploying LLm agents?"
    )
    st.divider()
    st.link_button("View LangSmith traces ->", "https://smith.langchain.com")


if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("meta"):
            with st.expander("Pipeline details"):
                st.json(msg["meta"])


if prompt := st.chat_input("Ask anything about LLM production..."):

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Running pipeline (traced in LangSmith)..."):
            pipeline_input: Any = {
                "question":         prompt,
                "session_id":       st.session_state.session_id,
                "refined_question": "",
                "doc_sections":     [],
                "web_results":      "",
                "synthesis":        "",
                "final_report":     "",
                "steps_taken":      [],
            }
            result = graph.invoke(pipeline_input)

        st.markdown(result["final_report"])

        meta = {
            "refined_question": result["refined_question"],
            "pipeline":         " → ".join(result["steps_taken"]),
            "doc_sections_found": len(result["doc_sections"]),
            "session_id":       result["session_id"],
        }
        with st.expander("Pipeline details"):
            st.json(meta)

    st.session_state.messages.append({
        "role":    "assistant",
        "content": result["final_report"],
        "meta":    meta,
    }) 