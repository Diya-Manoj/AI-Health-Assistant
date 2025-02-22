import streamlit as st
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings  # Fixed Import
from langchain.vectorstores import FAISS
from langchain.document_loaders import WikipediaLoader
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline

# Load pre-trained Hugging Face model correctly
hf_pipeline = pipeline("summarization", model="facebook/bart-large-cnn")

# Wrap it in LangChain's HuggingFacePipeline
llm = HuggingFacePipeline(pipeline=hf_pipeline)

# Load medical data from Wikipedia
loader = WikipediaLoader(query="Diabetes", lang="en")
documents = loader.load()

# Convert text into vector embeddings for retrieval
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(documents, embeddings)

# Setup RAG-based retrieval system
qa_chain = RetrievalQA(llm=llm, retriever=vectorstore.as_retriever())

# Streamlit UI
st.title("AI Health Assistant (RAG-based)")

user_input = st.text_input("Ask your medical question:")
if st.button("Submit"):
    response = qa_chain.run(user_input)
    st.write("Healthcare Assistant:", response)
