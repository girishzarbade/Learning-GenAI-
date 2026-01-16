from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

st.title("QnA Chatbot 🤖")
st.markdown("My Q&A chatbot with langchain using Google Gemini model.")

if "messages" not in st.session_state:
    st.session_state.messages = []
    
for message in st.session_state.messages:
    st.chat_message(message["role"]).markdown(message["content"])
    

query = st.chat_input("Ask me anything !")
if query:
    st.session_state.messages.append({"role": "user", "content": query})
    st.chat_message("user").markdown(query)
    response = llm.invoke(query)
    st.chat_message("assistant").markdown(response.content)
    st.session_state.messages.append({"role": "assistant", "content": response.content})