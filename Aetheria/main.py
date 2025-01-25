from openai import OpenAI
import tiktoken
import streamlit as st
import fitz  # PyMuPDF
import os
from dotenv import load_dotenv
import re
import requests

# Load environment variables from .env file
load_dotenv()

# Ambil API Key dan variabel lainnya dari file .env
DEFAULT_API_KEY = os.getenv("API_KEY")
DEFAULT_BASE_URL = os.getenv("BASE_URL")
DEFAULT_MODEL = os.getenv("MODEL")
DEFAULT_TEMPERATURE = float(os.getenv("TEMPERATURE", 0.7))
DEFAULT_MAX_TOKENS = int(os.getenv("MAX_TOKENS", 512))
DEFAULT_TOKEN_BUDGET = int(os.getenv("TOKEN_BUDGET", 4096))
DEFAULT_TOP_P = float(os.getenv("TOP_P", 1.0))

# Setel konfigurasi Streamlit
st.set_page_config(page_title="Aetheria", layout="wide", initial_sidebar_state='collapsed')

class ConversationManager:
    def __init__(self, api_key=None, base_url=None, model=None, temperature=None, max_tokens=None, token_budget=None):
        self.api_key = api_key or DEFAULT_API_KEY
        self.base_url = base_url or DEFAULT_BASE_URL
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

        self.model = model or DEFAULT_MODEL
        self.temperature = temperature or DEFAULT_TEMPERATURE
        self.max_tokens = max_tokens or DEFAULT_MAX_TOKENS
        self.token_budget = token_budget or DEFAULT_TOKEN_BUDGET
        self.top_p = DEFAULT_TOP_P

        self.system_message = "You are a supportive and kind career guidance assistant. Your role is to review and provide feedback on curriculum vitae, cover letters, job applications, and any career-related inquiries. You respond with encouragement, constructive advice, and helpful insights to boost the user's confidence and preparedness. However, if a user's question is not related to career or job topics, you should respond with, 'I'm sorry.' Always aim to be friendly, patient, and uplifting in your guidance."
        self.conversation_history = [{"role": "system", "content": self.system_message}]

    def count_tokens(self, text):
        try:
            encoding = tiktoken.encoding_for_model(self.model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
        tokens = encoding.encode(text)
        return len(tokens)

    def total_tokens_used(self):
        return sum(self.count_tokens(message['content']) for message in self.conversation_history)

    def enforce_token_budget(self):
        while self.total_tokens_used() > self.token_budget:
            if len(self.conversation_history) <= 1:
                break
            self.conversation_history.pop(1)

    def is_career_related(self, prompt):
        en_career_keywords = [
            "job", "resume", "cv", "interview", "career", "cover letter","passionate",
            "application", "promotion", "hiring", "employment", "internship",
            "networking", "skills", "work experience", "workplace", "salary",
            "offer", "manager", "colleague", "performance", "professional",
            "career growth", "linkedin", "portfolio", "ATS"
        ]

        in_career_keywords = [
            "pekerjaan", "resume", "cv", "wawancara", "karier", "surat lamaran",
            "lamaran", "promosi", "rekrutmen", "pekerjaan", "magang",
            "koneksi", "keterampilan", "pengalaman kerja", "tempat kerja", "gaji",
            "penawaran", "manajer", "rekan kerja", "kinerja", "profesional",
            "pertumbuhan karier", "linkedin", "portofolio"
        ]

        return any(re.search(rf"\b{keyword}\b", prompt, re.IGNORECASE) for keyword in en_career_keywords+in_career_keywords)

    def chat_completion(self, prompt):
        self.conversation_history.append({"role": "user", "content": prompt})
        self.enforce_token_budget()

        if not self.is_career_related(prompt):
            ai_response = "I apologize, but I can only assist with career-related questions and topics. Please feel free to ask me about resumes, job applications, interviews, or any other career guidance you need."
            self.conversation_history.append({"role": "assistant", "content": ai_response})
            return ai_response
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.conversation_history,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p
            )
        except Exception as e:
            print(f"Error generating response: {e}")
            return None

        ai_response = response.choices[0].message.content
        self.conversation_history.append({"role": "assistant", "content": ai_response})
        print(ai_response)
        return ai_response

    def reset_conversation_history(self):
        self.conversation_history = [{"role": "system", "content": self.system_message}]

    def update_system_message(self, new_message):
        self.system_message = new_message
        self.reset_conversation_history()

# Streamlit code
st.title("AI Chatbot")

# Initialize the ConversationManager object
if 'chat_manager' not in st.session_state:
    st.session_state['chat_manager'] = ConversationManager()

chat_manager = st.session_state['chat_manager']

# Sidebar widgets for settings
st.sidebar.header("Settings")
model_options = ["meta-llama/Llama-Vision-Free", "M2-BERT-Retrieval-32k", "BAAI-Bge-Base-1p5"]
chat_manager.model = st.sidebar.selectbox("Model Name", model_options, index=model_options.index(chat_manager.model) if chat_manager.model in model_options else 0)
chat_manager.temperature = st.sidebar.slider("Temperature", 0.0, 1.0, chat_manager.temperature, 0.01)
chat_manager.max_tokens = st.sidebar.number_input("Max Tokens", value=chat_manager.max_tokens, min_value=1, step=1)
chat_manager.top_p = st.sidebar.slider("Top-p", 0.0, 1.0, chat_manager.top_p, 0.01)

# File input for PDF
uploaded_file = st.file_uploader("Upload a file", type=["pdf"])

# Extract and display content from PDF
if uploaded_file is not None:
    with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
        file_content = ""
        for page_num in range(doc.page_count):
            page = doc[page_num]
            file_content += page.get_text()

    st.write("**File Content:**")
    st.write(file_content)

    # Optionally, use the file content as input for the chatbot
    question_about_file = st.text_input("What questions do you have about this file?")
    if st.button("Send File Content to Chatbot"):
        if question_about_file:
            prompt = f"Here is the file content:\n{file_content}\n\nQuestion: {question_about_file}"
            response = chat_manager.chat_completion(prompt)
            print("response file: ", response)
            st.write("**Chatbot Response:**")
            st.write(response)
        else:
            st.warning("Please enter a question about the file first.")

# Chat input from the user
user_input = st.chat_input("Write a message")
print("User input: ", user_input)
# Call the chat manager to get a response from the AI
if user_input:
    response = chat_manager.chat_completion(user_input)

# Display the conversation history
for message in chat_manager.conversation_history:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.write(message["content"])
