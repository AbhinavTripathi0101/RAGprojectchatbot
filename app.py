import os
import streamlit as st
import textwrap
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFaceHub
from langchain.chains import RetrievalQA

# 🔹 Set Hugging Face API Key
HUGGINGFACEHUB_API_TOKEN = "your_huggingface_api_key"  # Replace with your key
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN

# 🔹 Streamlit UI
st.set_page_config(page_title="RAG PDF Chatbot", layout="wide")
st.title("📄 RAG-based PDF Chatbot")
st.sidebar.header("Upload a PDF")

# 🔹 Upload PDF File
uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file:
    # 🔹 Step 3: Load PDF and Extract Text
    with open("uploaded.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    loader = PyPDFLoader("uploaded.pdf")
    documents = loader.load()

    # 🔹 Step 4: Split Text into Chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)

    # 🔹 Step 5: Embed Text Using Sentence Transformers
    embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

    # 🔹 Step 6: Store in ChromaDB (Persistent Storage)
    vector_store = Chroma.from_documents(
        chunks, embeddings, persist_directory="./chroma_db"
    )
    vector_store.persist()
    retriever = vector_store.as_retriever()

    # 🔹 Step 7: Load Mistral-7B from Hugging Face API
    llm = HuggingFaceHub(
        repo_id="mistralai/Mistral-7B-Instruct-v0.1",
        model_kwargs={"temperature": 0.5, "max_length": 512},
    )

    # 🔹 Step 8: Setup RAG Pipeline
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    # 🔹 Step 9: Chat Interface
    st.sidebar.success("✅ PDF Processed! You can now ask questions.")

    user_input = st.text_input("💬 Ask a question about the PDF:")
    if user_input:
        try:
            response = qa_chain.run(user_input)
            st.markdown(f"**🤖 Bot:** {textwrap.fill(response, width=80)}")
        except Exception as e:
            st.error(f"Error: {str(e)}")

# 🔹 Footer
st.sidebar.markdown("---")
st.sidebar.markdown("🔹 **Developed by Abhinav Tripathi** 🚀")
