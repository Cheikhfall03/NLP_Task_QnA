import streamlit as st
#import torch
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from transformers import pipeline
import os

# Set up Streamlit page configuration
st.set_page_config(page_title="AIMS 2023/2024 Report Chatbot", layout="centered")

# Custom CSS for AIMS branding
st.markdown("""
    <style>
    .main {
        background-color: #f4f4f4;
        font-family: "Segoe UI", sans-serif;
    }
    .stTextInput > div > div > input {
        padding: 12px;
        border-radius: 6px;
        border: 1px solid #ccc;
        font-size: 16px;
        transition: border-color 0.3s;
    }
    .stTextInput > div > div > input:focus {
        border-color: #9d2235;
        outline: none;
    }
    .stButton > button {
        background-color: #9d2235;
        color: white;
        padding: 12px 20px;
        font-size: 16px;
        border-radius: 6px;
        border: none;
        width: 100%;
        transition: background-color 0.3s;
    }
    .stButton > button:hover {
        background-color: #7c1a2a;
    }
    .response-box {
        background-color: #fdf6f6;
        border-left: 4px solid #9d2235;
        padding: 20px;
        border-radius: 8px;
        margin-top: 20px;
    }
    .header {
        text-align: center;
        padding: 20px 0;
    }
    .header img {
        height: 60px;
    }
    .meta {
        text-align: center;
        font-size: 14px;
        color: #666;
        margin-top: 40px;
    }
    h1 {
        text-align: center;
        font-size: 26px;
        color: #2e2e2e;
        margin-bottom: 30px;
    }
    </style>
""", unsafe_allow_html=True)

# Display AIMS logo and header
st.markdown('<div class="header"><img src="./static/aims-logo.jpg" alt="AIMS Logo"></div>', unsafe_allow_html=True)
st.markdown('<h1>Chatbot – AIMS 2023/2024 Report</h1>', unsafe_allow_html=True)

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Load and process the PDF
@st.cache_resource
def load_documents():
    pdf_paths = ["Review-Year-2023-2024.pdf"]
    pdf_docs = []
    for path in pdf_paths:
        if os.path.exists(path):
            loader = PyPDFLoader(path)
            pdf_docs.extend(loader.load())
        else:
            st.error(f"PDF file {path} not found.")
            return []
    return pdf_docs

# Set up the LangChain pipeline
@st.cache_resource
def setup_langchain():
    # Load documents
    all_docs = load_documents()
    if not all_docs:
        return None, None, None

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
    docs = text_splitter.split_documents(all_docs)

    # Set up embeddings
    embeddings = HuggingFaceEmbeddings()

    # Create vector store
    vectorstore = Chroma.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 2})

    # Set up LLM
    model_id = "tiiuae/falcon-7b-instruct"
    text_generation_pipeline = pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        max_new_tokens=400
    )
    llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

    # Define prompt template
    prompt_template = """
    <|system|>
    Answer the question based on your knowledge. Use the following context to help:

    {context}
    </s>
    <|user|>
    {question}

    <|assistant|>
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    # Create RAG chain
    llm_chain = prompt | llm | StrOutputParser()
    rag_chain = {"context": retriever, "question": RunnablePassthrough()} | llm_chain

    return retriever, llm, rag_chain

# Initialize LangChain components
retriever, llm, rag_chain = setup_langchain()

# User input form
with st.form(key="question_form", clear_on_submit=True):
    question = st.text_input("Posez votre question sur le rapport AIMS...", placeholder="Ex: Who is the academic director?")
    submit_button = st.form_submit_button("Envoyer")

# Process question and display response
if submit_button and question and rag_chain:
    with st.spinner("Génération de la réponse..."):
        response = rag_chain.invoke(question)
        response = response.replace("</s>", "").strip()
        
        # Store in chat history
        st.session_state.chat_history.append({"question": question, "answer": response})

# Display chat history
for chat in st.session_state.chat_history[::-1]:  # Reverse to show latest first
    st.markdown(f"""
    <div class="response-box">
        <h3>Question :</h3>
        <p>{chat['question']}</p>
        <h3>Réponse :</h3>
        <p>{chat['answer']}</p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown('<div class="meta">© 2024 – Institut Africain des Sciences Mathématiques (AIMS)</div>', unsafe_allow_html=True)
