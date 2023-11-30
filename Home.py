import streamlit as st

st.set_page_config(
    page_title="AI Journalist Assistant",
    page_icon='💬',
    layout='wide'
)

st.header("AI Journalist Assistant")
st.write("""
[![view source code ](https://img.shields.io/badge/GitHub%20Repository-gray?logo=github)](https://github.com/Joaoluislins/gpt-news-pages)
![Visitors](https://api.visitorbadge.io/api/visitors?path=https%3A%2F%2Fpages-news.streamlit.app//&label=Visitors&labelColor=%235d5d5d&countColor=%231e7ebf&style=flat)
""")
st.write("""
Tackling trustworthy GenAI in the news domain through the usage of LangChain + Chain-of-Thought + Internet Search + RAG.

- **Short Story Journalist**: Provide a short story and get a full news article.
- **Journalist Assistant (Internet Search)**: Your personal assistant that remembers previous interactions and generates a news article in a conversational manner, using web content to develop it's generations.
- **Journalist Assistant (Your Documents)** Empower the Assistant with the ability to access custom documents, enabling it to provide news articles based on the referenced information.

To explore sample usage of each Assistant, please navigate to the corresponding chatbot section.
""")