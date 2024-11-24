import os
import random
import streamlit as st

#decorator
def enable_chat_history(func):
    # to clear chat history after swtching chatbot
    current_page = func.__qualname__
    if "current_page" not in st.session_state:
        st.session_state["current_page"] = current_page
    
    if st.session_state["current_page"] != current_page:
        try:
            st.cache_resource.clear()
            del st.session_state["current_page"]
            del st.session_state["messages"]
            del st.session_state['model']
            del st.session_state['obj']
        except:
            pass
    # to show chat history on ui
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?", 'fig': None}]
        
    for msg in st.session_state["messages"]:
        if msg['role'] == 'assistant':
            if msg['fig']:
                with st.chat_message("assistant"):
                    st.image(msg['fig'], use_column_width = 'wide')
        st.chat_message(msg["role"]).write(msg["content"])

    def execute(*args, **kwargs):
        func(*args, **kwargs)
    return execute


def enable_chat_history_pure(func):
    # to clear chat history after swtching chatbot
    current_page = func.__qualname__
    if "current_page" not in st.session_state:
        st.session_state["current_page"] = current_page
    
    if st.session_state["current_page"] != current_page:
        try:
            st.cache_resource.clear()
            del st.session_state["current_page"]
            del st.session_state["messages"]
            del st.session_state['model']
            del st.session_state['obj']
        except:
            pass
    
    # this part shows messages in the ui at every turn
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?", 'fig': None}]
        
    for msg in st.session_state["messages"]:
        role = msg['role']
        with st.chat_message(role):
            st.markdown(f"<ol>{msg['content']}</ol>", unsafe_allow_html=True)
        # st.chat_message(msg["role"]).write(msg["content"])
        
    def execute(*args, **kwargs):
        func(*args, **kwargs)
    return execute

def enable_chat_history_news(func):
    st.markdown("""
    <style>
        [data-testid="column"]:nth-child(2){
            background-color: #2e2e2e; /* Dark grey background */
            color: #f5f5f5; /* Light text color */
            border-radius: 10px; /* Rounded corners */
            padding: 15px;
            
        }
        
        
        [data-testid="column"]:nth-child(1) {
            overflow: auto;
            height: 70vh;
        }
    </style>
    """, unsafe_allow_html=True
    )
    col1, col2 = st.columns(2)
    # to clear chat history after swtching chatbot
    current_page = func.__qualname__
    if "current_page" not in st.session_state:
        st.session_state["current_page"] = current_page
    
    if st.session_state["current_page"] != current_page:
        try:
            st.cache_resource.clear()
            del st.session_state["current_page"]
            del st.session_state["messages"]
            del st.session_state['model']
            del st.session_state['obj']
        except:
            pass
        
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    with col1:
        for msg in st.session_state["messages"]:
            st.chat_message(msg["role"]).write(msg["content"])


    def execute(*args, **kwargs):
        func(*args, **kwargs)
    return execute

def display_msg(msg, author):
    """Method to display message on the UI

    Args:
        msg (str): message to display
        author (str): author of the message -user/assistant
    """
    st.session_state.messages.append({"role": author, "content": msg})
    with st.chat_message(author):
            st.markdown(f"<ol>{msg}</ol>", unsafe_allow_html=True)
    # st.chat_message(author).write(msg)

def configure_openai_api_key():
    openai_api_key = st.sidebar.text_input(
        label="OpenAI API Key",
        type="password",
        value=st.session_state['OPENAI_API_KEY'] if 'OPENAI_API_KEY' in st.session_state else '',
        placeholder="sk-..."
        )
    if openai_api_key:
        st.session_state['OPENAI_API_KEY'] = openai_api_key
        os.environ['OPENAI_API_KEY'] = openai_api_key
    else:
        st.error("Please add your OpenAI API key to continue.")
        st.info("Obtain your key from this link: https://platform.openai.com/account/api-keys")
        st.stop()
    return openai_api_key