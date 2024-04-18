__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import os
import streamlit as st
from components_chatbot import configure_vectordb, configure_vectordb_from_index, PrintRetrievalHandler, StreamHandler, setup_langchain_retriever
from pinecone import Pinecone as PineconeClient
from components_pinecone import describe_index

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"]="https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"]="ls__38a2804f07f44b929b61733aa2936f93"
os.environ["LANGCHAIN_PROJECT"]="portfolio"

st.set_page_config(page_title="Ramon's Portfolio: Chat with PDF", page_icon="💧")
st.title("💧Chat with PDF")

with st.sidebar.container():
    openai_api_key = st.text_input("OpenAI API Key", type="password")
    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()

    data_source = st.selectbox("Choose your data source:", ["Upload File", "Pinecone Index"])
    vectordb = None
    namespace_names = []

    pc = PineconeClient(api_key=st.secrets["PINECONE_API_KEY"])
    indexes = pc.list_indexes()
    index_names = [index['name'] for index in indexes]
    index_name = st.selectbox("Choose an Index Name:", index_names)

    if not index_name:
        st.info("Please enter Pinecone Index Name to continue.")
        st.stop()

    if index_name:
        index = pc.Index(index_name)
        index_describe = index.describe_index_stats()
        namespaces = index_describe['namespaces']
        namespace_names = list(namespaces.keys())
        namespace_names.append("None")
    namespace = st.selectbox("Choose a Namespace: (Optional)", namespace_names)
    if namespace == "None":
        namespace = None
                
    if data_source == "Upload File":
        mode = "Upload File"
        uploaded_files = st.file_uploader(
            label="Upload PDF files", type=["pdf"], accept_multiple_files=True
        )
        if not uploaded_files:
            st.info("Please upload PDF documents to continue.")
            st.stop()

        vectordb = configure_vectordb(uploaded_files=uploaded_files, openai_api_key=openai_api_key, 
                              index_name=index_name, namespace=namespace if namespace else None,
                              )
    elif data_source == "Pinecone Index":
        vectordb = configure_vectordb_from_index(openai_api_key, index_name, namespace=namespace if namespace else None)
    
    if vectordb is None:
        st.stop()

    model = st.selectbox("Model", ["gpt-3.5-turbo", "gpt-4"])
    temperature = st.slider("Temperature", 0.0, 1.0, 0.3)
    max_tokens = st.slider("Max Tokens", 50, 2500, 2000)
    k = st.slider("Context Blocks", 1, 20, 5)

def configure_retriever(k):
    # Define retriever
    retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": k})
    return retriever

retriever = configure_retriever(k)

qa_chain, msgs = setup_langchain_retriever(model=model, openai_api_key=openai_api_key, 
                                           temperature=temperature, max_tokens=max_tokens, 
                                           retriever=retriever)

if len(msgs.messages) == 0 or st.sidebar.button("Clear message history"):
    msgs.clear()
    msgs.add_ai_message("How can I help you?")

avatars = {"human": "user", "ai": "assistant"}
for msg in msgs.messages:
    st.chat_message(avatars[msg.type]).write(msg.content)

if user_query := st.chat_input(placeholder="Ask me anything!"):
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        retrieval_handler = PrintRetrievalHandler(st.container())
        stream_handler = StreamHandler(st.empty())
        response = qa_chain.run(user_query, callbacks=[retrieval_handler, stream_handler])
