import os
import streamlit as st
from PIL import Image
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.memory import ConversationBufferMemory, CombinedMemory, ConversationKGMemory

# Set Session State
if "openai_api_key" not in st.session_state:
    st.error("Please set your OpenAI API Key.", icon="🚨")
else:
    openai_api_key = st.session_state["openai_api_key"]

if "db" not in st.session_state:
    st.error("Please Upload a TXT, PDF or CSV File.", icon="🚨")
else:
    db = st.session_state["db"]

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []

if "input" not in st.session_state:
    st.session_state["input"] = ""

if "stored_session" not in st.session_state:
    st.session_state["stored_session"] = []

os.environ["OPENAI_API_KEY"] = openai_api_key
image = Image.open('pfp.png')
image = image.resize((100, int(100 * image.height / image.width)))

# Define function to get user input
def get_text():
    query = st.text_input("You:", st.session_state["input"], key="input", placeholder="Enter your message here...", label_visibility='hidden')
    return query

# Define function to start a new chat
def new_chat():
    save = []
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        save.append("User:" + st.session_state["past"][i])
        save.append("Bot:" + st.session_state["generated"][i]['output_text'])      
    st.session_state["stored_session"].append(save)
    st.session_state["generated"] = []
    st.session_state["past"] = []
    st.session_state["input"] = ""
    if "entity_memory" in st.session_state:
        del st.session_state["entity_memory"]

with st.sidebar.container():
    model = st.selectbox("Model", ["gpt-3.5-turbo", "gpt-4"])
    temperature = st.slider("Temperature", 0.0, 1.0, 0.5)
    max_tokens = st.slider("Max Tokens", 50, 1000, 500)
    k = st.slider("Context Blocks", 1, 5, 3)
    st.button(label="Wipe Memory", on_click=new_chat, type="primary")
    st.divider()
    st.image(image)
    st.title("Hi, I'm Ramon :wave:")
    st.subheader("I develop simple AI tools for small business applications.")
    st.subheader(
        "You can interact with your personal documents using OpenAI's chatGPT. Try it out..."
    )

with st.container():
    st.header("Chat with your own Data")

query = get_text()

if openai_api_key and db:
    # Create OpenAI Instance
    llm = ChatOpenAI(
        temperature=temperature, 
        openai_api_key=openai_api_key,
        model_name=model,
        max_tokens=max_tokens,
        streaming=True
        )
    
    # Create a ConversationEntityMemory object if not already created
    if 'entity_memory' not in st.session_state:
            KG = ConversationKGMemory(llm=llm, input_key="human_input")
            CBM = ConversationBufferMemory(memory_key="chat_history", input_key="human_input")
            st.session_state["entity_memory"] = CombinedMemory(memories=[KG, CBM])


    # Set Template
    template = """You are a chatbot having a conversation with a human.

    Given the following extracted parts of a long document and a question, create a final answer.

    {context}

    Relevant Information:

    {history}

    Chat History:

    {chat_history}
    Human: {human_input}
    Chatbot:"""

    # Set the Prompt
    prompt = PromptTemplate(
        input_variables=["chat_history", "human_input", "context", "history"], 
        template=template
    )

    # Get Context
    if query:
        docsearch = db.similarity_search(query, k=k)

else:
    st.error('Please Upload a TXT, PDF or CSV File.', icon="🚨")

if query:
    # Fetch docs using user input for cosine similarity
    docs = db.similarity_search(query, k=k)

    # Get Response
    chain = load_qa_chain(llm, chain_type="stuff", memory=st.session_state["entity_memory"], prompt=prompt, verbose=True)

    # Generate the output using user input and store it in the session state
    output = chain({"input_documents": docs, "human_input": query}, return_only_outputs=True)
    st.session_state.past.append(query)
    st.session_state.generated.append(output)

    with st.container():
        for i in range(len(st.session_state["generated"])):
            st.info("User: " + st.session_state["past"][i])
            st.success("Chatbot: " + st.session_state["generated"][i]['output_text'])
            with st.expander("Sources that contributed to your Answer"):
                st.write(docs)