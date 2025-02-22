import asyncio

# Fixes "no running event loop" error in Streamlit
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.run(asyncio.sleep(0))
import streamlit as st
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import WikipediaLoader
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline
import os
import torch

FAISS_INDEX_PATH = "faiss_index"

# ✅ Optimize Model Loading: Cache to prevent reloading on each query
@st.cache_resource
def load_model():
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", 
                    device=0 if torch.cuda.is_available() else -1)  # ✅ Uses smaller, faster model

hf_pipeline = load_model()
llm = HuggingFacePipeline(pipeline=hf_pipeline)

# ✅ Optimize FAISS Retrieval: Save & reload prebuilt index instead of recomputing every time
@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    if os.path.exists(FAISS_INDEX_PATH):
        return FAISS.load_local(FAISS_INDEX_PATH, embeddings)
    else:
        loader = WikipediaLoader(query="Diabetes", lang="en")  # ✅ Fetch only once
        documents = loader.load()
        vectorstore = FAISS.from_documents(documents, embeddings)
        vectorstore.save_local(FAISS_INDEX_PATH)
        return vectorstore

vectorstore = load_vectorstore()

# ✅ Avoid Re-Initializing Model & Retriever on Every Query
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
