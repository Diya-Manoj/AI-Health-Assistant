import streamlit as st
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import WikipediaLoader
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline

# Load model using Hugging Face transformers pipeline
hf_pipeline = pipeline("summarization", model="facebook/bart-large-cnn")

# Wrap in LangChain's HuggingFacePipeline
llm = HuggingFacePipeline(pipeline=hf_pipeline)

# Load medical data from Wikipedia
loader = WikipediaLoader(query="Heart Attack", lang="en")
documents = loader.load()

# Convert text into vector embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(documents, embeddings)

# Setup retrieval-based Q&A system
qa_chain = RetrievalQA.from_llm(llm=llm, retriever=vectorstore.as_retriever())

# Streamlit UI
st.title("AI Health Assistant (RAG-based)")

user_input = st.text_input("Ask your medical question:")
if st.button("Submit"):
    # Retrieve documents before calling the model
    retrieved_docs = vectorstore.similarity_search(user_input, k=3)
    
    if not retrieved_docs:
        st.write("⚠️ No relevant information found. Please try a different query.")
    else:
        response = qa_chain.run(user_input)
        st.write("Healthcare Assistant:", response)
