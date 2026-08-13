import streamlit as st
import json
import os
from rag_engine import build_or_update_index, answer_query, GAP_LOG_FILE

st.set_page_config(page_title="Institutional RAG Assistant", layout="wide")

# Sidebar - Navigation & Index Button
st.sidebar.title("📌 Navigation")
mode = st.sidebar.radio("Select View:", ["Student / Staff Chat", "Admin Analytics Dashboard"])

st.sidebar.divider()
if st.sidebar.button("🔄 Re-Index Documents Folder"):
    with st.spinner("Indexing PDFs..."):
        count = build_or_update_index()
        if count > 0:
            st.sidebar.success(f"Indexed {count} chunks!")
        else:
            st.sidebar.warning("No PDFs found in ./documents folder.")

# ----------------- VIEW 1: STUDENT CHAT -----------------
if mode == "Student / Staff Chat":
    st.title("🎓 Institutional Knowledge Assistant")
    st.caption("Ask questions about prospectus, exam rules, circulars, and campus guidelines.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    query = st.chat_input("Ask a question...")

    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        # Retain short conversation context for follow-up questions
        history_str = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages[-4:]])

        with st.chat_message("assistant"):
            with st.spinner("Searching institutional documents..."):
                answer, sources = answer_query(query, history_str)
                st.markdown(answer)

                if sources:
                    with st.expander("📚 Verified Citations & Document Sources"):
                        for s in sources:
                            st.write(f"📄 **{s['file']}** — Page {s['page']}")
                            st.caption(f'"{s["snippet"]}..."')

        st.session_state.messages.append({"role": "assistant", "content": answer})

# ----------------- VIEW 2: ADMIN DASHBOARD -----------------
else:
    st.title("📊 Institutional Gap Analytics (Admin Mode)")
    st.markdown("Track unanswerable student questions to identify documentation gaps in college policies.")

    if os.path.exists(GAP_LOG_FILE):
        with open(GAP_LOG_FILE, "r") as f:
            gaps = json.load(f)

        st.metric("Total Documented Gaps", len(gaps))
        st.subheader("Logged Unanswered Queries")

        for g in reversed(gaps):
            st.warning(f"❓ **{g['question']}**\n\n*Logged on: {g['timestamp']}*")
    else:
        st.info("No documentation gaps logged yet! The assistant has successfully answered all questions.")