import os
import streamlit as st

st.set_page_config(
    page_title="AI Retailing Assistant",
    page_icon='💬',
    layout='wide'
)

with open(os.path.join(st.secrets["DEMO_PATH"], "scripts/style.css"), "r") as f:
    css = f.read()

st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


st.header("AI Chatbot Research")
st.write("""
[![view source code ](https://img.shields.io/badge/GitHub%20Repository-gray?logo=github)](https://github.com/Joaoluislins/llm-agents)
![Visitors](https://api.visitorbadge.io/api/visitors?path=https%3A%2F%2Fpages-news.streamlit.app//&label=Visitors&labelColor=%235d5d5d&countColor=%231e7ebf&style=flat)
""")

st.markdown("""
            <div class="big-font">
                <p>
                    Some experimental Implementations of LLama3 8B designed to leverage personal Research in RAG, Reinforcement Learning and collection of Conversational Data.
                    <br><br>
                </p>
                        <ul class = 'big-font'>
                            <li>💬&nbsp&nbsp&nbsp&nbsp<b>Generalist</b>: Quantized (4bit) Checkpoint of Llama3 8B model built with Llama.cpp. This checkpoint is not finetuned in any specific dataset, and doesn't perform RAG. It excels in interactive general domain conversations.</li>
                            <li>🗃️&nbsp&nbsp&nbsp&nbsp<b>(in-dev) Book Retailing Assistant (LLMchat + LLMrouter + RAG)</b>: Multimodal Chatbot tuned to excel in the online retailing assitance domain. This project aims to build an Assistant that interacts with the user, providing personalized product recommendations and operating the retail website cart in order to fully serve the User demands. Uses a custom database to augment (RAG) factualness of it's responses.</li>
                            <li>🌐&nbsp&nbsp&nbsp&nbsp<b>Web-Assistant (LLMchat + LLMrouter + LLMsearchquery + WebAPI +RAG)</b>: An internet-enabled chatbot capable of searching the web for information, reference it's answers and score the credibility of it's responses. (In Dev.)</li>
                            <li>⭐&nbsp&nbsp&nbsp&nbsp<b>(in-dev) Reinforcing Connection</b>: An implementation of <a href="https://arxiv.org/abs/2203.02155" target="_blank">RLHF</a> tailored with personalized rewards (In Dev.)</li>
                        </ul>
                <p>
                    To explore sample usage of each chatbot, please navigate to the corresponding chatbot section on the left.
                </p>
            </div>""", unsafe_allow_html=True)
