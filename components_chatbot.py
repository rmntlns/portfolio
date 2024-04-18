import streamlit as st
import openai
from langchain_community.chat_models import ChatOpenAI
from llama_index.llms.openai import OpenAI
try:
  from llama_index import VectorStoreIndex, ServiceContext, Document, SimpleDirectoryReader
except ImportError:
  from llama_index.core import VectorStoreIndex, ServiceContext, Document, SimpleDirectoryReader
import tempfile
import os
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain.callbacks.base import BaseCallbackHandler
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma, Pinecone
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.chat_message_histories import StreamlitChatMessageHistory

PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]

@st.cache_resource(show_spinner=False)
def load_data_directory():
    with st.spinner(text="Loading and indexing the docs from directory – hang tight! This should take 1-2 minutes."):
        reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
        docs = reader.load_data()
        # llm = OpenAI(model="gpt-3.5-turbo", temperature=0.5, system_prompt="You are an expert o$
        # index = VectorStoreIndex.from_documents(docs)
        service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-3.5-turbo", 
                                                                  temperature=0.5, 
                                                                  system_prompt="You are an expert on the Streamlit Python library and your job is to answer technical questions. Assume that all questions are related to the Streamlit Python library. Keep your answers technical and based on facts – do not hallucinate features."))
        index = VectorStoreIndex.from_documents(docs, service_context=service_context)
        return index
    

@st.cache_resource(show_spinner=False)
def load_data_file(uploaded_files):
    with st.spinner(text="Loading and indexing the file – hang tight! This should take 1-2 minutes."):
        with tempfile.TemporaryDirectory() as temp_dir:  # Use the context manager to ensure cleanup
            for file in uploaded_files:
                temp_filepath = os.path.join(temp_dir, file.name)  # Use temp_dir directly
                with open(temp_filepath, "wb") as f:
                    f.write(file.getvalue())

            documents = SimpleDirectoryReader(temp_dir).load_data()
        service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-3.5-turbo", 
                                                                  temperature=0.5, 
                                                                  system_prompt="You are an expert on the Streamlit Python library and your job is to answer technical questions. Assume that all questions are related to the Streamlit Python library. Keep your answers technical and based on facts – do not hallucinate features."))
        index = VectorStoreIndex.from_documents(documents, service_context=service_context)
        return index

def load_pinecone_index(index_name, namespace=None):
    pc = Pinecone(api_key=PINECONE_API_KEY)
    pinecone_index = pc.Index(index_name)
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index, namespace=namespace)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)
    return index

def setup_langchain(index_name, namespace=None, model="gpt-3.5-turbo", temperature=0.3, max_tokens=2000, k=5):
    from langchain_openai import OpenAIEmbeddings, ChatOpenAI
    from langchain_pinecone import PineconeVectorStore as PineconeVectorStoreLangchain
    from langchain.chains import RetrievalQA

    # Initialize OpenAI Embeddings with specific settings
    embeddings = OpenAIEmbeddings(
        model='text-embedding-3-large',  # Adjust if needed
        openai_api_key=openai.api_key,
        dimensions=1536
    )

    # Initialize Pinecone Vector Store
    vectorstore = PineconeVectorStoreLangchain(
        pinecone_api_key=PINECONE_API_KEY,
        index_name=index_name,
        embedding=embeddings,
        text_key="text",
        namespace=namespace
    )

    # Initialize ChatOpenAI with dynamic settings
    llm = ChatOpenAI(
        model=model,
        openai_api_key=openai.api_key,
        temperature=temperature,
        max_tokens=max_tokens
    )

    # Setup the Retrieval QA Chain
    qa = RetrievalQA.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={'k': k}
        )
    )

    return qa, llm


@st.cache_resource(ttl="1h", experimental_allow_widgets=True, show_spinner="Chunking and vectorizing documents...")
def configure_vectordb(uploaded_files, openai_api_key, index_name, namespace=None):
    from langchain_community.vectorstores import Pinecone
    from langchain.document_loaders import PyPDFLoader
    from langchain_openai import OpenAIEmbeddings

    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    docs = []
    temp_dir = tempfile.TemporaryDirectory()
    for file in uploaded_files:
        temp_filepath = os.path.join(temp_dir.name, file.name)
        with open(temp_filepath, "wb") as f:
            f.write(file.getvalue())
        loader = PyPDFLoader(temp_filepath)
        docs = loader.load()

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    vectordb = Pinecone.from_documents(splits, embedding=embeddings, index_name=index_name, namespace=namespace)

    return vectordb

def configure_vectordb_from_index(openai_api_key, index_name, namespace=None):
    from langchain_community.vectorstores import Pinecone
    from langchain_openai import OpenAIEmbeddings

    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectordb = Pinecone.from_existing_index(embedding=embeddings, index_name=index_name, namespace=namespace)

    return vectordb

# Setup LLM and QA chain
def setup_langchain_retriever(model, openai_api_key, temperature, max_tokens, retriever):
    from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
    from langchain.memory import ConversationBufferMemory
    from langchain.chat_models import ChatOpenAI
    from langchain.chains import ConversationalRetrievalChain

    # Setup memory for contextual conversation
    msgs = StreamlitChatMessageHistory()
    memory = ConversationBufferMemory(memory_key="chat_history", chat_memory=msgs, return_messages=True)

    llm = ChatOpenAI(
        model_name=model, openai_api_key=openai_api_key, temperature=temperature, max_tokens=max_tokens, streaming=True
    )
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm, retriever=retriever, memory=memory, verbose=True
    )
    return qa_chain, msgs

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