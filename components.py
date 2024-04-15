import streamlit as st
import streamlit_shadcn_ui as ui
import json
from langchain_community.document_loaders import TextLoader
import tempfile
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os 
from llama_index.core import SimpleDirectoryReader
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import StorageContext
from llama_index.core.schema import TextNode
from llama_index.embeddings.openai import OpenAIEmbedding

@st.cache_resource(ttl="1h", experimental_allow_widgets=True, show_spinner="Chunking and vectorizing documents...")
def configure_vectordb(uploaded_files, openai_api_key, _pinecone_index):
    docs = []
    with tempfile.TemporaryDirectory() as temp_dir:  # Use the context manager to ensure cleanup
        for file in uploaded_files:
            temp_filepath = os.path.join(temp_dir, file.name)  # Use temp_dir directly
            with open(temp_filepath, "wb") as f:
                f.write(file.getvalue())

        documents = SimpleDirectoryReader(temp_dir).load_data()

    num_metadata = st.number_input('How many metadata fields would you like to add?', min_value=0, max_value=10, step=1)

    metadata_inputs = {}
    for i in range(num_metadata):
        cols1, cols2 = st.columns(2)
        with cols1: 
            metadata_key = ui.input(placeholder=f'Enter metadata key {i+1}:', key=f'metadata_key_{i}')
        with cols2:
            metadata_value = ui.input(placeholder=f'Enter metadata value for {metadata_key}:', key=f'metadata_value_{i}')
        metadata_inputs[metadata_key] = metadata_value
        st.stop()

    if metadata_inputs is not None:
        nodes = []
        for idx, text_chunk in enumerate(documents):
            node = TextNode(
                text=text_chunk.text,
                metadata=metadata_inputs
            )
            nodes.append(node)
    return nodes

def upsert_vectors(pinecone_client, index_name, namespace=None):
    st.header("Upsert Vectors to Pinecone Index")
    uploaded_files = st.file_uploader(
            label="Upload PDF files", type=["pdf"], accept_multiple_files=True
        )
    if not uploaded_files:
        st.info("Please upload PDF documents to continue.")
    elif uploaded_files and index_name is not None:
        st.text("OpenAI API Key:")
        openai_api_key = ui.input(type='text', placeholder=f"Enter API Key: ", key="openai_api_key")
        if not openai_api_key:
            st.stop()
        pinecone_index = pinecone_client.Index(index_name)
        nodes = configure_vectordb(uploaded_files, openai_api_key, pinecone_index)
        cols1, cols2 = st.columns(2)
        with cols1:
            if st.button("Show Nodes", key="show_nodes"):
                st.write("Nodes configured. Length: ", len(nodes), "\nFirst node: ", nodes[:1])
        with cols2:
            if st.button("Clear Cache", key="clear_cache"):
                st.cache_data.clear()
        # Button to upsert vectors to Pinecone
        upsert_button = st.button("Upsert Vectors", key="upsert_vectors_button")
        if upsert_button:
            vector_store = PineconeVectorStore(pinecone_index=pinecone_index, namespace=namespace if namespace else None)
            # Change embedding model
            embed_model =  OpenAIEmbedding(api_key=openai_api_key, model="text-embedding-3-small")

            for node in nodes:
                node_embedding = embed_model.get_text_embedding(
                    node.get_content(metadata_mode="all")
                )
                node.embedding = node_embedding
            
            st.write("Metadata embedded. Length: ", len(nodes), "\nFirst node: ", nodes[:1])
            response = vector_store.add(nodes)
            if response:
                st.write("Upsert Response:", response)
            else:
                st.write("Something went wrong. Please check the input variables and try again.")

def query_vectors(pinecone_client, index_name, namespace):
    st.header("Query Vectors from Pinecone Index")
    col1, col2 = st.columns(2)
    with col1:
        st.text("Top K:")
        top_k = ui.input(type='num', placeholder=f"Enter tok_k", key="top_k")
    with col2:
        st.text("Include Values:")
        include_values = ui.select(options=["True", "False"], key="include_values")
    st.text("Vector: ")
    vector = ui.input(type='text', placeholder=f"Enter Vector as a list: '[0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]'", key="vector_values")
    st.text("Filter: ")
    filter_example = """{
        "genre": {"$eq": "documentary"}
    }"""
    filter = st.text_area(label="Enter filter as a JSON: ", placeholder=f"'{filter_example}'", key="filter_values")  
    button = st.button("Query Vector", key="query_vector_button")
    if button:
        index = pinecone_client.Index(index_name)
        response = index.query(
            namespace=namespace if namespace else None,
            vector=vector,
            filter=filter,
            top_k=top_k,
            include_values=include_values
        )
        st.write("Query Response:", response)
        return response

def fetch_vector(pinecone_client, index_name, namespace):
    st.header("Fetch Vector from Pinecone Index")
    st.text("IDs: ")
    ids = ui.input(type='text', placeholder="Enter IDs (comma-separated): 'id-1, id-2'", key="ids")

    button = st.button("Fetch Vector", key="fetch_vector_button")
    if button:
        index = pinecone_client.Index(index_name)
        response = index.fetch(ids=ids.split(','), namespace=namespace if namespace else None)
        st.write("Fetch Response:", response)


def update_vector(pinecone_client, index_name, namespace):
    st.header("Update Vector in Pinecone Index")
    st.text("ID: ")
    id = ui.input(type='text', placeholder="Enter ID: 'id-3'", key="id")
    st.text("Values: ")
    values = ui.input(type='text', placeholder="Enter Values as a list: '4.0, 2.0'", key="values")

    button = st.button("Update Vector", key="update_vector_button")
    if button:
        index = pinecone_client.Index(index_name)
        response = index.update(id=id, values=values, namespace=namespace if namespace else None)
        st.write("Update Response:", response)

def pinecone_delete(pinecone_client, index_name, namespace):
    st.header("Delete Vectors from Pinecone Index")
    st.text("IDs: ")
    ids = ui.input(type='text', placeholder="Enter IDs (comma-separated)", key="ids")

    button = st.button("Delete Vectors", key="delete_vector_button")
    if button:
        index = pinecone_client.Index(index_name)
        index.delete(ids=ids.split(','), namespace=namespace if namespace else None)
        st.write("Vectors Deleted Successfully")

def list_vector_id(pinecone_client, index_name, namespace):
    st.header("List Vector IDs in Pinecone Index")
    st.text("Prefix: ")
    prefix = ui.input(type='text', placeholder="Enter Prefix", key="prefix")
    st.text("Limit (optional): ")
    limit = ui.input(type='num', placeholder="Enter Limit", key="limit")
    st.text("Pagination Token (optional): ")
    pagination_token = ui.input(type='text', placeholder="Enter Pagination Token", key="pagination_token")

    button = st.button("List Vector IDs", key="list_vector_id_button")
    if button:
        index = pinecone_client.Index(index_name)
        if limit:
            results = index.list_paginated(prefix=prefix, limit=int(limit), namespace=namespace if namespace else None, pagination_token=pagination_token)
        else:
            results = index.list(prefix=prefix, namespace=namespace if namespace else None)
        st.write("Vector IDs:", results)


def describe_index(pinecone_client, index_name):
    st.text("Pinecone Index Stats")
    button = st.button("Describe Index", key="describe_index_button")
    if button:
        index = pinecone_client.Index(index_name)
        stats = index.describe_index_stats()
        print(stats)
        
        with ui.card(key="Stats1"):
            ui.element("h1", children=[f"Index Stats: "], className="text-md font-bold", key="pricing_starter_1")
            ui.element("h1", children=[f"Dimension: {stats['dimension']}"], className="text-sm font-bold", key="pricing_starter_1")
            ui.element("h1", children=[f"index_fullness: {stats['index_fullness']}"], className="text-sm font-bold", key="pricing_starter_1")
            ui.element("h1", children=[f"namespaces: {stats['namespaces']}"], className="text-sm font-bold", key="pricing_starter_1")
            ui.element("h1", children=[f"total_vector_count: {stats['total_vector_count']}"], className="text-sm font-bold", key="pricing_starter_1")

def list_indexes(pinecone_client):
    st.text("List Indexes")
    button = st.button("List Index", key="list_index_button")
    if button:
        indexes = pinecone_client.list_indexes()
        st.write(indexes)