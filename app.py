import streamlit as st
import os
import asyncio
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import WikipediaLoader
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline

# ✅ Fix "No Running Event Loop" Issue
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.run(asyncio.sleep(0))

# ✅ Fix Missing Torch Installation
try:
    import torch
except ImportError:
    os.system("pip install torch torchvision torchaudio")
    import torch

FAISS_INDEX_PATH = "faiss_index"

# ✅ Use WikipediaLoader Instead of `wikipedia-api`
loader = WikipediaLoader(query="Diabetes", lang="en")
documents = loader.load()

# ✅ Load Model (Cache to Reduce Loading Time)
@st.cache_resource
def load_model():
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", 
                    device=0 if torch.cuda.is_available() else -1)

hf_pipeline = load_model()
llm = HuggingFacePipeline(pipeline=hf_pipeline)

# ✅ Optimize FAISS Retrieval
@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    if os.path.exists(FAISS_INDEX_PATH):
        return FAISS.load_local(FAISS_INDEX_PATH, embeddings)
    else:
        vectorstore = FAISS.from_documents(documents, embeddings)
        vectorstore.save_local(FAISS_INDEX_PATH)
        return vectorstore

vectorstore = load_vectorstore()

# ✅ Avoid Re-Initializing Model on Every Query
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = RetrievalQA.from_llm(llm=llm, retriever=vectorstore.as_retriever())

qa_chain = st.session_state.qa_chain

# ✅ Streamlit UI
st.title("🚀 AI Health Assistant (Optimized)")

user_input = st.text_input("Ask your medical question:")
if st.button("Submit"):
    retrieved_docs = vectorstore.similarity_search(user_input, k=3)  # ✅ Fetch fewer docs for speed

    if not retrieved_docs:
        st.write("⚠️ No relevant information found. Try a different query.")
    else:
        response = qa_chain.run(user_input)
        st.write("Healthcare Assistant:", response)
