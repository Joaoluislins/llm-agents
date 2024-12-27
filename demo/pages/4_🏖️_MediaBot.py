#TODO: add logging
#TODO: add error handling
#TODO: add streaming feature


import os
import streamlit as st
import uuid
import re
import json

import scripts.utils as utils
from groq import Groq
from scripts.mediaBot.mediaLLM import MediaLLM

st.set_page_config(page_title="MediaBot",
                page_icon="🏖️",
                layout='wide')

with open(os.path.join(st.secrets["DEMO_PATH"], "scripts/style.css"), "r") as f:
    css = f.read()

st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

st.header('MediaBot')
st.header('\n')

# st.write("""
#         <div class="big-font">
#             <p>
#                 This is a conversational chatbot built to interact in a general domain setting.
#             </p>
#         </div>""", unsafe_allow_html=True)


@utils.enable_chat_history_pure(bot_name = 'mediabot')
def main():
    user = st.chat_input(placeholder="Ask me anything!")
    if user:
        utils.display_msg(user, 'user')
        with st.chat_message("assistant"):
            response = st.session_state.MediaLLM.prepare_and_generate(user, generation_type = 'chat')
            st.write(response)
            st.session_state.messages.append({"role": "assistant",
                                              "content": response,
                                            })


generation_model = 'groq'

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    
if "MediaLLM" not in st.session_state:
    MediaLLM = MediaLLM(generation_model)
    st.session_state.MediaLLM = MediaLLM

llm = st.session_state.MediaLLM.setup_model(generation_model)
st.session_state.MediaLLM.main()
