__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import os
import tempfile
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain_openai import OpenAIEmbeddings
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"]="https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"]="ls__38a2804f07f44b929b61733aa2936f93"
os.environ["LANGCHAIN_PROJECT"]="portfolio"

st.set_page_config(page_title="Ramon's Portfolio: Chat with CSV", page_icon="💧")
st.title("💧Chat with CSV")

with st.sidebar.container():
    openai_api_key = st.text_input("OpenAI API Key", type="password")
    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()

    if openai_api_key:
        uploaded_files = st.file_uploader(
            label="Upload CSV files", type=["csv"], accept_multiple_files=True
        )
    if not uploaded_files:
        st.info("Please upload CSV documents to continue.")
        st.stop()

    # if st.button("Clear Cache"): 
    #     st.cache_resource.clear()
    #     st.stop()

    if uploaded_files:
        model = st.selectbox("Model", ["gpt-3.5-turbo", "gpt-4"])
        temperature = st.slider("Temperature", 0.0, 1.0, 0.3)
        max_tokens = st.slider("Max Tokens", 50, 2500, 2000)
        k = st.slider("Context Blocks", 1, 20, 5)

@st.cache_resource(ttl="1h", show_spinner="Chunking and vectorizing documents...")
def configure_vectordb(uploaded_files):
    # Read documents
    docs = []
    temp_dir = tempfile.TemporaryDirectory()
    for file in uploaded_files:
        temp_filepath = os.path.join(temp_dir.name, file.name)
        with open(temp_filepath, "wb") as f:
            f.write(file.getvalue())
        loader = CSVLoader(temp_filepath)
        docs = loader.load()

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # Create embeddings and store in vectordb
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    # vectordb = DocArrayInMemorySearch.from_documents(splits, embeddings)
    vectordb = Chroma.from_documents(splits, embeddings)

    return vectordb

def configure_retriever(k):
    # Define retriever
    retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": k})

    return retriever

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container: st.delta_generator.DeltaGenerator, initial_text: str = ""):
        self.container = container
        self.text = initial_text
        self.run_id_ignore_token = None

    def on_llm_start(self, serialized: dict, prompts: list, **kwargs):
        # Workaround to prevent showing the rephrased question as output
        if prompts[0].startswith("Human"):
            self.run_id_ignore_token = kwargs.get("run_id")

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        if self.run_id_ignore_token == kwargs.get("run_id", False):
            return
        self.text += token
        self.container.markdown(self.text)


class PrintRetrievalHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.status = container.status("**Context Retrieval**")

    def on_retriever_start(self, serialized: dict, query: str, **kwargs):
        self.status.write(f"**Question:** {query}")
        self.status.update(label=f"**Context Retrieval:** {query}")

    def on_retriever_end(self, documents, **kwargs):
        for idx, doc in enumerate(documents):
            source = os.path.basename(doc.metadata["source"])
            self.status.write(f"**Document {idx} from {source}**")
            self.status.markdown(doc.page_content)
        self.status.update(state="complete")

vectordb = configure_vectordb(uploaded_files)

retriever = configure_retriever(k)

# Setup memory for contextual conversation
msgs = StreamlitChatMessageHistory()
memory = ConversationBufferMemory(memory_key="chat_history", chat_memory=msgs, return_messages=True)

# Setup LLM and QA chain
llm = ChatOpenAI(
    model_name=model, openai_api_key=openai_api_key, temperature=temperature, max_tokens=max_tokens, streaming=True
)
qa_chain = ConversationalRetrievalChain.from_llm(
    llm, retriever=retriever, memory=memory, verbose=True
)

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