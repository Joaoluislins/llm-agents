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
from scripts.model_setup import LLM_SETUP

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
            # Start global handler to get response and media
            print(st.session_state.MediaLLM.generation_model)
            response, media = st.session_state.MediaLLM.GlobalHandler(user)
            
            if media['videos']:
                st.video(media['videos'][0])
            elif media['images']:
                st.image(media['images'][0])
                
            st.write(response)
            st.session_state.messages.append({"role": "assistant",
                                              "content": response,
                                              "media": media,
                                            })
            
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    
if "LLM_SETUP" not in st.session_state:
    LLM_SETUP = LLM_SETUP()
    st.session_state.LLM_SETUP = LLM_SETUP

generation_model = 'groq'
llm = st.session_state.LLM_SETUP.setup_llm(generation_model)
    
if "MediaLLM" not in st.session_state:
    MediaLLM = MediaLLM(llm = llm, generation_model = generation_model)
    st.session_state.MediaLLM = MediaLLM

main()
