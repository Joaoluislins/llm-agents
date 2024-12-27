import os
import json
import re
import streamlit as st
from groq import Groq

class MediaLLM:
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
        """
        Builds the message history for the chat conversation.
        
        Args:
            sys_prompt (dict): System prompt that sets the context/behavior
            user_prompt (dict): Current user message
            generation_type (str): Type of generation (e.g. 'chat')
            
        Returns:
            list: List of message dictionaries containing the conversation history
        """
        # Initialize with system prompt
        messages = [sys_prompt]
        
        if generation_type == 'chat':
            # Add conversation history
            messages.extend(st.session_state["messages"])
            # Handle the user prompt
            if messages and messages[-1]['role'] == 'user':
                # Replace last message if it was from user
                messages[-1] = user_prompt
            else:
                # Add new user message
                messages.append(user_prompt)
                
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
