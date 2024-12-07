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

st.set_page_config(page_title="Generalist",
                page_icon="💬",
                layout='wide')

with open(os.path.join(st.secrets["DEMO_PATH"], "scripts/style.css"), "r") as f:
    css = f.read()

st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

st.markdown(
    """
    <style>
    [data-testid="stChatMessageContent"] p{
        font-size: 1.3rem;
    }
    </style>
    """, unsafe_allow_html=True
)

st.header('Generalist')
st.header('\n')

# st.write("""
#         <div class="big-font">
#             <p>
#                 This is a conversational chatbot built to interact in a general domain setting.
#             </p>
#         </div>""", unsafe_allow_html=True)


class PureLLM:
    def __init__(self, generation_model):
        # self.model_id = "/data1/demobot/hf/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf"
        self.cache_dir = os.path.join(st.secrets["DEMO_PATH"], 'hf')
        self.generation_model = generation_model
        # TODO start log session
                
    def setup_model(self, model_type):
        # if model_type == 'llama':
        #     @st.cache_resource
            # def load_llama():
            #     llm = Llama(
            #         model_path=self.model_id,
            #         chat_format = 'llama-3',
            #         n_gpu_layers=-1,
            #         device = f"cuda:{llama_cuda}",
            #         verbose = False,
            #         main_gpu = 0,
            #         n_ctx = 0
            #         # seed=1337,
            #         )
                # return llm
            # return load_llama()
        
        if model_type == 'groq':
            @st.cache_resource
            def load_groq_client():
                client = Groq(api_key=st.secrets["GROQ_API_KEY"])
                return client
            return load_groq_client()
        
        
    def load_sys_prompts(self, prompt_name):
        if prompt_name == 'chat':
            with open(os.path.join(st.secrets["DEMO_PATH"], "prompts/simple_chat/sys_prompt_chat_pure.json"), 'r') as file:
                prompt = json.load(file)
        return prompt
    
    def format_user_sys_prompts(self, user, generation_type):
        if generation_type == 'chat':
            user_prompt = {"role": "user", "content": f"{user}"}
            sys_prompt = self.load_sys_prompts('chat')
            return user_prompt, sys_prompt
        
    def messages_builder(self, sys_prompt, user_prompt, generation_type):
        ### types
        # chat -> have all conversation context
        messages = [sys_prompt]

        if generation_type == 'chat':

            for msg in st.session_state["messages"]:
                messages.append({
                                 'role': msg['role'],
                                 'content': msg['content']
                                 })
            
            # case of not first message
            if messages[-1]['role'] == 'user':
                messages[-1] = user_prompt            
            # first message case
            else:
                messages.append(user_prompt)

        # with open(f"{DEMOBOT_HOME}/logs/flow_conversation/{st.session_state.session_id}.txt", 'a') as file:
            # file.write(f"\nMessages sent to '{type}' generation:\n\n")
            # for msg in messages:
            #     file.write(f"+++{msg['role']}: {msg['content']}\n")
            # file.write(f"\n\n")
            # dashes = '-' * 20
            # file.write(f"{dashes}\n") 

        return messages
    
    def generate(self, messages):
        if self.generation_model == 'llama-local':
            response = llm.create_chat_completion(
                max_tokens = 256,
                temperature = 0.2,
                top_p=0.9,
                messages = messages
            )
            return response['choices'][0]['message']['content']
        
        elif self.generation_model == 'groq':
            chat_completion = llm.chat.completions.create(
                messages=messages,
                model="llama-3.3-70b-versatile",
                temperature=0.3,
                max_tokens=1024,
                top_p=1,
                stop=None,
                stream=False,)
            
            return chat_completion.choices[0].message.content
    
    def format_response(self, response):
        response =re.sub(r'\$', '\$', response)
        if response.startswith('Assistant') or response.startswith('assistant'):
            return response[len('Assistant: '): ]
        else:
            return response
        
    def prepare_and_generate(self, user, generation_type):
        user_prompt, sys_prompt = self.format_user_sys_prompts(user, generation_type = generation_type)
        messages = self.messages_builder(sys_prompt, user_prompt, generation_type = generation_type)        
        response = self.generate(messages)        
        response = self.format_response(response)
        return response


    @utils.enable_chat_history_pure
    def main(self):
        user = st.chat_input(placeholder="Ask me anything!")
        if user:
            utils.display_msg(user, 'user')
            with st.chat_message("assistant"):
                
                response = self.prepare_and_generate(user, generation_type = 'chat')
                st.write(response)
                st.session_state.messages.append({"role": "assistant",
                                                  "content": response,
                                                })
                
                
generation_model = 'groq'

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    
if "LLM" not in st.session_state:
    LLM = PureLLM(generation_model)
    st.session_state.LLM = LLM

llm = st.session_state.LLM.setup_model(generation_model)
st.session_state.LLM.main()