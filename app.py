import streamlit as st
import os
import asyncio

# ✅ Fix "No Running Event Loop" Error
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.run(asyncio.sleep(0))

# ✅ Dynamically Install Missing Dependencies
try:
    import torch
except ImportError:
    os.system("pip install torch")
    import torch

try:
    import wikipediaapi
except ImportError:
    os.system("pip install wikipedia-api")
    import wikipediaapi

from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import WikipediaLoader
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline

FAISS_INDEX_PATH = "faiss_index"

# ✅ Optimize Model Loading: Cache the Model
@st.cache_resource
def load_model():
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", 
                    device=0 if torch.cuda.is_available() else -1)  # ✅ Uses a smaller, faster model

hf_pipeline = load_model()
llm = HuggingFacePipeline(pipeline=hf_pipeline)

# ✅ Optimize FAISS Retrieval: Save & reload prebuilt index
@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    if os.path.exists(FAISS_INDEX_PATH):
        return FAISS.load_local(FAISS_INDEX_PATH, embeddings)
    else:
        loader = WikipediaLoader(query="Diabetes", lang="en")  # ✅ Fetch only once
        documents = loader.load()
        vectorstore = FAISS.from_documents(doc
