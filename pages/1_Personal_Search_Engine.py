import os
import streamlit as st
from PIL import Image
from langchain_community.chat_models import ChatOpenAI
from langchain.chains.qa_with_sources import load_qa_with_sources_chain

# Set Session State
if "openai_api_key" not in st.session_state:
    st.error("Please set your OpenAI API Key.", icon="🚨")
else:
    openai_api_key = st.session_state["openai_api_key"]

if "db" not in st.session_state:
    st.error("Please Upload a TXT, PDF or CSV File.", icon="🚨")
else:
    db = st.session_state["db"]

def similarity_search():
    search = db.similarity_search(query, k=k)
    return search

image = Image.open('pfp.png')
image = image.resize((100, int(100 * image.height / image.width)))

with st.sidebar.container():
    MODEL = st.selectbox("Model", ["gpt-3.5-turbo", "gpt-4"])
    temperature = st.slider("Temperature", 0.0, 1.0, 0.3)
    max_tokens = st.slider("Max Tokens", 50, 1000, 500)
    k = st.slider("Context Blocks", 1, 30, 5)
    st.divider()
    st.image(image)
    st.title("Hi, I'm Ramon :wave:")
    st.subheader("I develop simple AI tools for small business applications.")
    st.subheader(
        "You can interact with your personal documents using OpenAI's chatGPT. Try it out..."
    )

st.header("Personal Search Engine")

if openai_api_key and db:
    os.environ["openai_api_key"] = openai_api_key
    llm = ChatOpenAI(
            model_name=MODEL,
            temperature=temperature,
            openai_api_key=openai_api_key,
            max_tokens=max_tokens
        )
    query = st.text_input(label="After writing your Question, Press Enter.:", placeholder="Enter your message here...")
    if query:
        with st.spinner("Generating your Answer..."):
            chain = load_qa_with_sources_chain(llm, chain_type="stuff")
            docs = similarity_search()
            response = chain(
                {"input_documents": docs, "question": query}, return_only_outputs=True
            )
            st.subheader(response["output_text"])
            with st.expander("Sources that contributed to your Answer"):
                st.write(docs)
        st.success("Done!")
else:
    st.error("Please Upload a TXT, PDF or CSV File.", icon="🚨")