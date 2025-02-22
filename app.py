import streamlit as st
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import WikipediaLoader
from langchain.llms import HuggingFacePipeline

# Load pre-trained Hugging Face model
llm = HuggingFacePipeline.from_model_id("facebook/bart-large-cnn")

# Load medical data from Wikipedia
loader = WikipediaLoader(query="Diabetes", lang="en")
documents = loader.load()

# Convert text into vector embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(documents, embeddings)

# Setup retrieval-based Q&A system
qa_chain = RetrievalQA(llm=llm, retriever=vectorstore.as_retriever())

# Streamlit UI
st.title("AI Health Assistant (RAG-based)")

user_input = st.text_input("Ask your medical question:")
if st.button("Submit"):
    response = qa_chain.run(user_input)
    st.write("Healthcare Assistant:", response)
