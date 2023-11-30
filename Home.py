import streamlit as st

st.set_page_config(
    page_title="AI Journalist Assistant",
    page_icon='💬',
    layout='wide'
)

st.header("Chatbot Implementations with Langchain")
st.write("""
[![view source code ](https://img.shields.io/badge/GitHub%20Repository-gray?logo=github)](https://github.com/Joaoluislins/gpt-news-pages)
![Visitors](https://api.visitorbadge.io/api/visitors?path=https%3A%2F%2Fpages-news.streamlit.app//&label=Visitors&labelColor=%235d5d5d&countColor=%231e7ebf&style=flat)
""")
st.write("""
Langchain is a powerful framework designed to streamline the development of applications using Language Models (LLMs). It provides a comprehensive integration of various components, simplifying the process of assembling them to create robust applications.

Leveraging the power of Langchain, the creation of chatbots becomes effortless. Here are a few examples of chatbot implementations catering to different use cases:

- **Short story development**: Provide a short story and get a full news article.
- **Journalist Assistant (Internet Search)**: Your personal assistant that remembers previous interactions and generates a news article in a conversational manner, using web content to develop it's generations.
- **Journalist Assistant (Your Documents)** Empower the Assistant with the ability to access custom documents, enabling it to provide news articles based on the referenced information.

To explore sample usage of each chatbot, please navigate to the corresponding chatbot section.
""")