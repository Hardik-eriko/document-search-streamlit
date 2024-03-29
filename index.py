import streamlit as st

import json, os

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import CharacterTextSplitter

from huggingface_hub import InferenceClient

import random

from dotenv import load_dotenv

load_dotenv()


json_file_path = "info.json"

with open(json_file_path, "r") as file:
    documents = json.load(file)

label_list = [key for key, value in documents.items()]

def embedding(filepaths, vector_db):
    for file_path in filepaths:
        loader = PyPDFLoader(file_path)
        documents = loader.load()

        text_splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=20, separator="\n\n")
        docs = text_splitter.split_documents(documents)
        print(docs)
        # create the open-source embedding function
        embedding_function = SentenceTransformerEmbeddings(model_name="multi-qa-MiniLM-L6-cos-v1")

        # load it into Chroma
        Chroma.from_documents(docs, embedding_function, persist_directory=vector_db) 

def format_prompt(message):
    prompt = f"[INST] {message} [/INST]"
    return prompt

def detail_generate(context, query):
    client = InferenceClient(
        "mistralai/Mistral-7B-Instruct-v0.2"
    )
    
    system_prompt = f"""
    You are an advanced document analyzer. You have to describe the following document briefly and how it relates to follwing query.
    Query: {query},
    Document: {context},
    Answer: 
    """
    formatted_prompt = format_prompt(f"{system_prompt}")
    
    output = client.text_generation(formatted_prompt, stream=False, details=True, return_full_text=False, stop_sequences=["\n"], max_new_tokens=256)

    return output.generated_text

def evidence_generate(context, query):
    client = InferenceClient(
        "mistralai/Mistral-7B-Instruct-v0.2"
    )
    
    system_prompt = f"""
    You are an advanced document analyzer. You have to describe why the following document is related to follwing query. This could be how the document content aligns with the query's intent, the use of synonyms or related terms, or the overall context of the document.
    Query: {query},
    Context: {context},
    Answer: 
    """
    formatted_prompt = format_prompt(f"{system_prompt}")
    
    output = client.text_generation(formatted_prompt, stream=False, details=True, return_full_text=False, stop_sequences=["\n"], max_new_tokens=256)

    return output.generated_text

def result_generate(query, source_list, detail, evidence):
    result = ""
    result += f"User Input: {query}\n\nInformation Found:\n- Document: {source_list[0]}\n- Details: {detail}\n- Evidence: {evidence}\n\nRelated Documents:\n\n"

    length = len(source_list)
    if length != 1:
        for i in range(1, length):
            result += f"- {source_list[i]}\n"
    
    return result
def document_search(vector_db, query):
    embedding_function = SentenceTransformerEmbeddings(model_name="multi-qa-MiniLM-L6-cos-v1")
    db = Chroma(persist_directory=vector_db, embedding_function=embedding_function)
    docs = db.similarity_search(query)
    print(docs)
    source_list = []

    for index in range(len(docs)):
        if not docs[index].metadata['source'] in source_list:
            source_list.append(docs[index].metadata['source'])
    
    context = ""
    for doc in docs:
        context += f"{doc.page_content}\n\n"
    
    detail = detail_generate(context, query)
    evidence = evidence_generate(context, query)
    
    return result_generate(query, source_list, detail, evidence)

st.set_page_config(layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

col1, col2 = st.columns([1, 3])

col1.subheader("Select a group")
with col1:
     
    selected_group = st.radio(
            "Group list",
            key="visibility",
            label_visibility="hidden",
            options=label_list,
        )

    with st.expander("Document Upload"):
        group_name = st.text_input('Group Label', placeholder="Type group label")
        uploaded_files = st.file_uploader("Choose PDF files", accept_multiple_files=True, type="pdf")
        
        upload_but = st.button("Upload", use_container_width=True)
        if upload_but:
            if not group_name:
                st.toast('Please type group label.')
            elif not uploaded_files:
                st.toast('Please select files.')
            elif group_name in label_list:
                st.toast("Such document label already exist.")
            else:
                dir_path = f"./database/{group_name}_data"
                os.makedirs(dir_path)
                
                filepaths = []
                for uploaded_file in uploaded_files:
                    with open(os.path.join(dir_path, uploaded_file.name),"wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    filepaths.append(os.path.join(dir_path, uploaded_file.name))
                    
                embedding(filepaths, f"./database/{group_name}_db")
                
                with open(json_file_path, "r") as file:
                    data = json.load(file)

                data[group_name] = f"./database/{group_name}_db"

                with open(json_file_path, "w") as file:
                    json.dump(data, file, indent=4)

                st.rerun()
with col2:
    query = st.chat_input('Query')
    
    if query:
        with st.chat_message("user"):
            st.markdown(query)
        with open(json_file_path, "r") as file:
            documents = json.load(file)
        
        with st.spinner('Please wait...'):
            result = document_search(documents[selected_group], query)
        
        with st.chat_message("assistant"):
            response = st.write(result)
