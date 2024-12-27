import os
import json
import re
import streamlit as st
from groq import Groq
from scripts.mediaBot.pixabayAPI import PixabayImageSearch

PIXABAY_API_KEY = st.secrets["PIXABAY_API_KEY"]
DEMO_PATH = st.secrets["DEMO_PATH"]

class MediaLLM:
    def __init__(self, llm, generation_model):
        self.llm = llm
        self.generation_model = generation_model
        # TODO start log session
        
    def load_sys_prompts(self, prompt_name):
        if prompt_name == 'chat':
            with open(os.path.join(DEMO_PATH, "prompts/simple_chat/sys_prompt_chat_pure.json"), 'r') as file:
                prompt = json.load(file)
        elif prompt_name == 'create_query':
            with open(os.path.join(DEMO_PATH, "prompts/mediaBot/sys_prompt_create_query_MediaLLM.json"), 'r') as file:
                prompt = json.load(file)
        
        return prompt
    
    
    
    def format_user_sys_prompts(self, user, generation_type):
        if generation_type == 'chat':
            user_prompt = {"role": "user", "content": f"{user}"}
            sys_prompt = self.load_sys_prompts('chat')
            return user_prompt, sys_prompt
        
        elif generation_type == 'create_query':
            user_prompt = {"role": "user", "content": f"{user}"}
            sys_prompt = self.load_sys_prompts('create_query')
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
        
        if generation_type == 'chat' or generation_type == 'create_query':
            # Add conversation history
            # Add last 50 messages from history, keeping only role and content
            for msg in st.session_state["messages"][-50:]:
                messages.append({
                    'role': msg['role'],
                    'content': msg['content']
                })
            
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
            response = self.llm.create_chat_completion(
                max_tokens = 256,
                temperature = 0.2,
                top_p=0.9,
                messages = messages
            )
            return response['choices'][0]['message']['content']
        
        elif self.generation_model == 'groq':
            chat_completion = self.llm.chat.completions.create(
                messages=messages,
                model="llama-3.3-70b-versatile",
                temperature=0.3,
                max_tokens=1024,
                top_p=1,
                stop=None,
                stream=False,)
            
            return chat_completion.choices[0].message.content
    
    #TODO: this is a temporary solution, I need to implement a better regex to format the response
    def format_response(self, response, generation_type = 'chat'):
        response =re.sub(r'\$', '\$', response)
        if response.startswith('Assistant') or response.startswith('assistant'):
            return response[len('Assistant: '): ]
        elif response.startswith('search query: ') or response.startswith('search query') or response.startswith('Search Query:') :
            return response[len('search query: '): ]
        else:
            return response
        
    def prepare_and_generate(self, user, generation_type):
        user_prompt, sys_prompt = self.format_user_sys_prompts(user, generation_type = generation_type)
        messages = self.messages_builder(sys_prompt, user_prompt, generation_type = generation_type)        
        response = self.generate(messages)        
        response = self.format_response(response)
        return response
    
    def search_media(self, query):
        pixabay = PixabayImageSearch(PIXABAY_API_KEY)
        images = pixabay.search_images(query)
        videos = pixabay.search_videos(query)
        return images, videos
    
    def GlobalHandler(self, user):
        search_query = self.prepare_and_generate(user, generation_type = 'create_query')
        print(search_query)
        images, videos = self.search_media(search_query)
        media = {'images': images, 'videos': videos, 'search_query': search_query}
        
        response = self.prepare_and_generate(user, generation_type = 'chat')
        return response, media