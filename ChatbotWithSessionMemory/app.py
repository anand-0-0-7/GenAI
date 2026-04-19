import warnings
warnings.filterwarnings("ignore")

import streamlit as st
from llm_providers import run_llm

st.set_page_config(page_title="AI Chat Assistant", page_icon="🤖")
st.title("🤖 AI Chat Assistant")

# Initialize chat history in session state
if "history" not in st.session_state:
    st.session_state["history"] = [{"role": "assistant", "content": "Hello! How can I assist you today?"}]

# Display chat history
for msg in st.session_state["history"]:
    st.chat_message(msg["role"]).markdown(msg["content"])

# user input
if prompt := st.chat_input("Type your message here..."):
    # Append user message to history
    st.session_state["history"].append({"role": "user", "content": prompt})
    st.chat_message("user").markdown(prompt)

    # Get assistant response
    reply = run_llm(st.session_state["history"])
    
    # Append assistant response to history
    st.session_state["history"].append({"role": "assistant", "content": reply})
    st.chat_message("assistant").markdown(reply)

if st.button("Reset History 🔄"):
    st.session_state["history"] = [{"role": "assistant", "content": "Hello! How can I assist you today?"}]
    st.rerun