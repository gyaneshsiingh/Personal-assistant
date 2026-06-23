
import streamlit as st
import requests


st.title("Personal Knowledge Assistant")

API_URL = "http://localhost:8000/ask"

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("Sources"):
                for source in message["sources"]:
                    st.markdown(f"- `{source}`")


if prompt := st.chat_input("What would you like to know"):
    st.chat_message("user").markdown(prompt)

    st.session_state.messages.append({"role": "user", "content": prompt})

    try:
        with st.spinner("Thinking..."):
            response = requests.post(API_URL, json = {"query": prompt})

            response.raise_for_status()
            data = response.json()

            answer = data.get("answer", "No answer found.")
            sources = data.get("sources",[])

        
        with st.chat_message("assistant"):
            st.markdown(answer)
            if sources:
                with st.expander("Sources"):
                    for source in sources:
                        st.markdown(f"- `{source}`")

    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to backend: {e}. Is the FastAPI server running?")