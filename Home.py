import os
import openai
import tempfile
import streamlit as st
import pandas as pd
from PIL import Image
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import DataFrameLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

st.set_page_config(
    page_title="Ramon Tilanus Portfolio",
    page_icon="🧊",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None
    }
)

if "db" not in st.session_state:
    st.session_state["db"] = []
if "openai_api_key" not in st.session_state:
    st.session_state["openai_api_key"] = []


data = None
image = Image.open('pfp.png')
image = image.resize((100, int(100 * image.height / image.width)))


def file_save(file_load):
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(file_load.getvalue())
        return f.name


def gen_db():
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=20,
        length_function=len,
    )
    texts = text_splitter.create_documents([doc.page_content for doc in data])
    for i, text in enumerate(texts):
        text.metadata["source"] = f"Document {i + 1}"
    embeddings = OpenAIEmbeddings(openai_api_key=openai.api_key)
    db = Chroma.from_documents(texts, embeddings)
    return db


with st.container():
    st.header("Enter your OpenAI API Key")
    openai_api_key = st.text_input(label="OpenAI API Key", placeholder="Paste your OpenAI API Key, sk-", type="password")
    st.session_state["openai_api_key"] = openai_api_key
    

    if openai_api_key:
        openai.api_key = openai_api_key
        os.environ["openai_api_key"] = openai.api_key
        st.subheader("Upload a document.")
        file_load = st.file_uploader("Upload File (TXT or PDF)")
        if file_load:
            file_path = file_save(file_load)
            filename = file_load.name
            filetype = filename.split(".")[-1]

            if file_load and filetype == "txt":
                loader = TextLoader(file_path, encoding='utf-8')
                data = loader.load()
                if data:
                    with st.spinner("Chunking and Vectorizing..."):
                        db = gen_db()
                        st.session_state["db"] = db
                    st.success("Done!")

            elif file_load and filetype == "csv":
                df = pd.read_csv(file_path)
                loader = DataFrameLoader(df)
                data = loader.load()
                if data:
                    with st.spinner("Chunking and Vectorizing..."):
                        db = gen_db()
                        st.session_state["db"] = db
                    st.success("Done!")

            elif file_load and filetype == "pdf":
                loader = PyPDFLoader(file_path)
                data = loader.load()
                if data:
                    with st.spinner("Chunking and Vectorizing..."):
                        db = gen_db()
                        st.session_state["db"] = db
                    st.success("Done!")
    else:
        st.info("Please provide your API Key.")

with st.sidebar.container():
    st.image(image)
    st.title("Hi, I'm Ramon :wave:")
    st.subheader("I develop simple AI tools for small business applications.")
    st.subheader(
        "You can interact with your personal documents using OpenAI's chatGPT. Try it out..."
    )