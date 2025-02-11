import streamlit as st
import requests
import google.genai as genai
from transformers import pipeline
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Initialize the Hugging Face pipeline for text generation using GPT-2
chatbot = pipeline("text-generation", model="distilgpt2", framework="tf")

# Your Gemini API Key
GEMINI_API_KEY = 'AIzaSyAgArq044CgjNLZvMg3tIAIjHPyoD-lQHs'

# Initialize the Gemini client
client = genai.Client(api_key=GEMINI_API_KEY)

# Function to send request to Gemini API
def get_gemini_response(user_input):
    try:
        # Make request to Gemini API using the generate_content method
        response = client.models.generate_content(
            model="gemini-2.0-flash-exp",  # You can choose the model you need, this is just an example
            contents=user_input
        )
        return response.text, "Gemini"
    except Exception as e:
        # If Gemini fails, fallback to DistilGPT-2
        print(f"Gemini API failed: {e}")
        return None, "GPT-2"

# Function to handle user input and generate responses
def healthcare_chatbot(user_input):
    """Function to handle user input and integrate with either Gemini API or fallback to GPT-2"""
    # Predefined responses for certain keywords
    if "headache" in user_input.lower():
        return "I'm sorry to hear you're having a headache. It might be a good idea to rest and stay hydrated. If it persists, please consult a healthcare professional.", "Predefined"
    elif "symptom" in user_input.lower():
        return "Please consult a doctor for an accurate diagnosis.", "Predefined"
    elif "appointment" in user_input.lower():
        return "Would you like to schedule an appointment with a doctor?", "Predefined"
    elif "medication" in user_input.lower():
        return "It's important to take prescribed medicines regularly. Consult your doctor if you have concerns.", "Predefined"
    else:
        # Attempt to get a response from Gemini API
        gemini_response, source = get_gemini_response(user_input)
        if gemini_response:
            return gemini_response, source
        else:
            # Fallback to GPT-2 model text generation if Gemini fails
            response = chatbot(user_input, max_length=200, num_return_sequences=1)
            return response[0]['generated_text'], "GPT-2"

import streamlit as st

def main():
    # Page configuration
    st.set_page_config(page_title="Healthcare Assistant", layout="wide")

    # Sidebar content
    with st.sidebar:
        st.title("🤖 Healthcare Chatbot")
        st.info("An AI-powered assistant for health-related inquiries.")
        st.markdown("---")
        st.write("💡 **Disclaimer:** This chatbot does not provide medical advice. Always consult a professional.")
        st.markdown("---")
        st.write("📚 **How to Use:**\n- Enter your query.\n- Click 'Submit'.\n- Receive an AI-generated response.")

    # Main content
    st.title("🩺 Healthcare Assistant Chatbot")
    st.write("Welcome! Ask me anything related to healthcare. I'll do my best to assist you. 😊")

    # User input
    st.markdown("### 📝 Your Query:")
    user_input = st.text_area("Type your question here...", height=150, placeholder="e.g., What are the symptoms of the flu?")

    # Submit button
    col1, col2, _ = st.columns([1, 1, 3])  # Adjust column widths for better button alignment
    with col1:
        if st.button("Submit 🚀"):
            if user_input.strip():
                with st.spinner("Processing your query... ⏳"):
                    response, source = healthcare_chatbot(user_input)
                st.markdown("### 💬 Response:")
                st.write(f"🧑‍⚕️ **You:** {user_input}")
                st.write(
                    f'<div style="text-align: left; font-size: 16px; white-space: pre-line;">🤖 <b>Chatbot:</b> {response}</div>',
                    unsafe_allow_html=True
                )
                st.write(f"📖 **Source:** {source}")

            else:
                st.warning("⚠️ Please enter a message to get a response.")

    with col2:
        if st.button("Clear 🧹"):
            st.query_params()

# Run the Streamlit app
if __name__ == "__main__":
    main()

