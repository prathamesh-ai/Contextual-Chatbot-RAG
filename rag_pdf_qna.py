import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader

from dotenv import load_dotenv
load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(groq_api_key = groq_api_key, model_name = "gemma2-9b-it")
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on provided context only.
    Please provide context
    <context>
    {context}
    <context>
    Question:{input}
    """
)

file = st.file_uploader("Upload a file", type="pdf")
if file is not None:
    file_path = os.path.join(os.getcwd(),file.name)
    with open(file_path,"wb") as f:
        f.write(file.getbuffer())

def create_vectors_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = OllamaEmbeddings(model="gemma:2b")
        st.session_state.loader = PyPDFLoader(file_path)
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
        #No need to return anything since everything is stored in session state
st.title("RAG Document Q&A")

user_prompt = st.text_input("Enter your query from the pdf.")

if st.button("Document Embedding"):
    create_vectors_embedding()

if user_prompt:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    response = retrieval_chain.invoke({"input":user_prompt})

    st.write(response['answer'])

    with st.expander("Document similarity Search"):
        for i,doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write('------------------------')

