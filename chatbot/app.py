"""Streamlit chat UI for the FAQ chatbot.

Shows the answer, a confidence score (cosine/composite similarity), and,
when the best match is weak, a "Did you mean?" list of the closest FAQs.
"""

import streamlit as st

from chatbot import FAQBot

st.set_page_config(page_title="FAQ Chatbot", page_icon="💬")
st.title("💬 FAQ Chatbot")
st.caption("Type a question in plain English. Matching uses TF-IDF + "
           "cosine similarity, with a confidence score.")

bot = FAQBot()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if "confidence" in msg:
            st.caption(f"Confidence: {msg['confidence'] * 100:.1f}%  "
                       f"(matched: {msg['matched_question'] or '—'})")
        for alt in msg.get("alternatives", []) or []:
            st.caption(f"↳ Did you mean: {alt['question']} "
                       f"({alt['confidence'] * 100:.1f}%)")

if prompt := st.chat_input("Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    result = bot.respond(prompt)
    with st.chat_message("assistant"):
        st.write(result["answer"])
        st.caption(f"Confidence: {result['confidence'] * 100:.1f}%  "
                   f"(matched: {result['matched_question'] or '—'})")
        alts = result.get("alternatives") or []
        if alts and result["matched_question"] is None:
            st.markdown("**Did you mean?**")
            for alt in alts:
                st.caption(f"• {alt['question']} "
                           f"({alt['confidence'] * 100:.1f}%)")

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "confidence": result["confidence"],
        "matched_question": result["matched_question"],
        "alternatives": result["alternatives"],
    })
