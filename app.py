import streamlit as st
from transformers import pipeline
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Explicitly download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.data.path.append("/tmp")  # Ensure download path is correct

# Load the medical Q&A model on demand (prevents caching issues)
def load_model():
    return pipeline("question-answering", model="deepset/roberta-base-squad2")

# Function to fetch context dynamically (This can be expanded later)
def get_medical_context(user_input):
    context_dict = {
        "flu": "Flu symptoms include fever, cough, sore throat, muscle aches, fatigue, and chills.",
        "diabetes": "Diabetes symptoms include increased thirst, frequent urination, hunger, fatigue, and blurred vision.",
        "hypertension": "Hypertension can cause headaches, dizziness, and shortness of breath.",
        "medication": "Always consult a doctor before changing your medication."
    }
    for keyword, context in context_dict.items():
        if keyword in user_input.lower():
            return context
    return "Please provide more details for an accurate response."

# Function for text preprocessing using NLTK
def preprocess_text(text):
    tokens = word_tokenize(text.lower())  # Convert to lowercase and tokenize
    filtered_tokens = [word for word in tokens if word.isalnum() and word not in stopwords.words("english")]
    return " ".join(filtered_tokens)  # Return cleaned text

# Function to process user input dynamically
def healthcare_chatbot(user_input):
    cleaned_input = preprocess_text(user_input)  # Preprocess user input
    if cleaned_input:
        chatbot = load_model()  # Reload model every time to prevent caching issues
        context = get_medical_context(cleaned_input)  # Get dynamic context
        response = chatbot(question=cleaned_input, context=context)
        return response["answer"]
    return "I'm sorry, I couldn't understand that. Please try again."

# Streamlit web app
def main():
    import numpy as np
    import pandas as pd

    st.title("Healthcare Assistant Chatbot 🤖")
    st.write("Ask your health-related question below and get AI-powered insights!")

    # Initialize Prediction History in session state if not exists
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    # User input field
    user_input = st.text_input("💬 Your Question:", "")

    # Predict Button
    if st.button("🔍 Get Response"):
        if user_input:
            response = healthcare_chatbot(user_input)  # Assuming function exists
            st.success(f"🤖 Healthcare Assistant: **{response}**")

            # Save Prediction History
            st.session_state['history'].append((user_input, response))
        else:
            st.warning("⚠️ Please enter a question before searching.")

    # Reset Button
    if st.button("🔄 Reset"):
        # Clear only inputs, retain history
        st.session_state['last_inputs'] = []
        st.rerun()

    # Ensure history is retained without resetting it
    if 'last_inputs' not in st.session_state:
        st.session_state['last_inputs'] = []

    # Download Option
    if user_input and 'response' in locals():
        st.download_button("📥 Download Response", f"Your Question: {user_input}\nAI Response: {response}", file_name="health_response.txt")

    # Sidebar for Prediction History
    with st.sidebar:
        st.header("📜 Previous Responses")
        if st.session_state['history']:
            for idx, (question, answer) in enumerate(st.session_state['history'][-5:]):
                st.write(f"**{idx + 1}.** ❓ {question}")
                st.write(f"👉 **{answer}**")
        else:
            st.write("No previous queries yet.")

if __name__ == "__main__":
    main()
