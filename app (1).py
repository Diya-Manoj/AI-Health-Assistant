import streamlit as st
from transformers import pipeline
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')


#load a pre-trained hugging face model
chatbot = pipeline("question-answering", model="timpal0l/mdeberta-v3-base-squad2")


#Define healthcare-specific response logic (or use a model to generate responses)
def healthcare_chatbot(user_input):
    #Simple rule based keywords to respond
    if "symptom" in user_input:
        return "It seems like you are experiencing symptoms. Please consult a doctor for accurate advice."
    elif "appointment" in user_input:
        return "Would you like to schedule an appointment with the doctor?"
    elif "medication" in user_input:
        return "It's important to take prescribed medicines regularly. If you have any concerns, consult your doctor."
    else:
        # Foe other inputs, use the hugging face model to generate a response
        response = chatbot(user_input,max_length=300,num_return_sequences=1)
        # Specifies the maximum length of te generated text response, including the input and the generated tokens.
        # If set to 3, te model generates three different possible responses based on the input.
        return response[0]['generated_text']
  
# Streamlit Web app interface
def main():
    st.title("AI Healthcare Assistant")
    user_input = st.text_input("How can I assist you today?")
    if st.button("Submit"):
        if user_input:
            st.write("User: ",user_input)
            with st.spinner("Processing your Query, Please Wait...."):
                response = healthcare_chatbot(user_input)
            st.write("Healthcare Assistant: ",response)
        else:
            st.write("Please enter a query.")
    
main()
