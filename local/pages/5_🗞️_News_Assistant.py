import os
import streamlit as st
import uuid
from scripts.news.AI_journalist import LLM_SETUP, AIEditorAssistant, AIEditor
from scripts import utils 
# from src.db import DB
from datetime import datetime
import pytz
from dotenv import load_dotenv
load_dotenv()
# TODO: update the code so that the two agents speaks to one another properly and the UI is updated with the conversation


# Page configs
st.set_page_config(page_title="🗞️ News Assistant",
                page_icon="🗞️",
                layout='wide')

DEMOBOT_HOME = os.getenv("DEMOBOT_HOME")
# Loading CSS Style
with open(os.path.join(DEMOBOT_HOME, "scripts/style.css"), "r") as f:
    css = f.read()

st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


st.header('🗞️ News Assistant')
st.header('\n')

col1, col2 = st.columns(2)

  
with col1:
    user = st.chat_input(placeholder="How can I help you in writing your news article?")
@utils.enable_chat_history_news
def main():
    
    ai_editor_assistant = st.session_state.AIEditorAssistant
    ai_editor = st.session_state.AIEditor
    
    
    with col1:
        # if not st.session_state.messages:
            # st.session_state.messages = [{"role": "assistant", "content": "How can I help you?"}]
            # st.chat_message("assistant").write("How can I help you?")
        # st.header("Chat Interface")
        # user = st.chat_input(placeholder="How can I help you in writing your news article?")
        if user:
            utils.display_msg(user, 'user')
            # Update the article using the user input as a prompt
            article_response = ai_editor.prepare_and_generate(user, previous_article = st.session_state.article_content)
            st.session_state.article_content = article_response

    with col2:
        st.header("Your News Article")
        previous_article = st.session_state.article_content
        if previous_article:
            st.write(st.session_state.article_content)
        else:
            st.write("No article content yet.")
            
    with col1:
        if user:
            chat_response = ai_editor_assistant.prepare_and_generate(user, previous_article = previous_article)
            # utils.display_msg(chat_response, 'assistant')
            st.chat_message("assistant").write(chat_response)
            st.session_state.messages.append({"role": "assistant", "content": chat_response})
        



# now = datetime.now(tz=pytz.timezone('US/Eastern')).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
# data to append db
# data_to_append = {
#     'datetime': f"{now}",
#     'user': f"{st.experimental_user.email}",
#     'input': f"{article_input}",
#     'output': f"{article_response}",}
# db = DB()
# df = db.read()
# df_updated = db.write(df, data_to_append)
# db.update()


if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    
if "LLM_SETUP" not in st.session_state:
    LLM_SETUP = LLM_SETUP()
    st.session_state.LLM_SETUP = LLM_SETUP
    
llm = st.session_state.LLM_SETUP.setup_llm("llama-3")
    
if "AIEditorAssistant" not in st.session_state:
    AIEditorAssistant = AIEditorAssistant(llm)
    st.session_state.AIEditorAssistant = AIEditorAssistant
    
if "AIEditor" not in st.session_state:
    AIEditor = AIEditor(llm)
    st.session_state.AIEditor = AIEditor
    
if "article_content" not in st.session_state:
    st.session_state.article_content = ""
    
main()