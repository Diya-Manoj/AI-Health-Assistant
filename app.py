import streamlit as st
from transformers import pipeline
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load a pre-trained medical Q&A model
med_chatbot = pipeline("question-answering", model="timpal0l/mdeberta-v3-base-squad2")

# Define a generic medical knowledge context
medical_context = """
Common symptoms of flu include fever, cough, sore throat, muscle aches, fatigue, and chills. 
Diabetes symptoms include increased thirst, frequent urination, hunger, fatigue, and blurred vision.
It's important to take prescribed medications regularly and consult a doctor before making any changes.
"""

# Function for text preprocessing using NLTK
def preprocess_text(text):
    tokens = word_tokenize(text.lower())  # Convert to lowercase and tokenize
    filtered_tokens = [word for word in tokens if word.isalnum() and word not in stopwords.words("english")]
    return " ".join(filtered_tokens)  # Return cleaned text

# Function to process user input
def healthcare_chatbot(user_input):
    cleaned_input = preprocess_text(user_input)  # Preprocess user input
    if cleaned_input:
        response = med_chatbot(question=cleaned_input, context=medical_context)
        return response["answer"]
    return "I'm sorry, I couldn't understand that. Please try again."

# Streamlit web app
def main():
    st.title("Healthcare Assistant Chatbot")

    # User input field
    user_input = st.text_input("Ask your health-related question:", "")

    if st.button("Submit"):
        if user_input:
            response = healthcare_chatbot(user_input)
            st.write("Healthcare Assistant: ", response)
        else:
            st.write("Please enter a query.")

if __name__ == "__main__":
    main()
