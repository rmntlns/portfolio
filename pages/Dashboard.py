import streamlit as st
import streamlit_shadcn_ui as ui
import os
from components import upsert_vectors, query_vectors, fetch_vector, update_vector, pinecone_delete, list_vector_id, describe_index, list_indexes
# Pinecone 
from pinecone import Pinecone

pinecone_client = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])


options = [
    "Upsert vectors",
    "Query vectors",
    "Fetch vectors",
    "Update a vector",
    "Delete vectors",
    "List vector IDs",
    "Get index stats"
]

# In your main function
def main():
    st.title("Dashboard")
    col1, col2 = st.columns(2)
    with col1:
        st.text("Choose Operation: ")
        choice = ui.select(options=options)
        st.text("Index Name: ")
        index_name = ui.input(type='text', placeholder="Enter Index Name", key="index_name")
        st.text("Namespace: ")
        namespace = ui.input(type='text', placeholder="Enter Namespace", key="namespace")
    with col2:
        describe_index(pinecone_client, index_name=index_name)
        list_indexes(pinecone_client)


    if choice == "Upsert vectors":
        upsert_vectors(pinecone_client, index_name=index_name, namespace=namespace)
    if choice == "Query vectors":
        query_vectors(pinecone_client, index_name=index_name, namespace=namespace)
    if choice == "Fetch vectors":
        fetch_vector(pinecone_client, index_name=index_name, namespace=namespace)
    if choice == "Update a vector":
        update_vector(pinecone_client, index_name=index_name, namespace=namespace)
    if choice == "Delete vectors":
        pinecone_delete(pinecone_client, index_name=index_name, namespace=namespace)
    if choice == "List vector IDs":
        list_vector_id(pinecone_client, index_name=index_name, namespace=namespace)

if __name__ == "__main__":
    main()










