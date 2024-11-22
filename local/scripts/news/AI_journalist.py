
#TODO: Consolidate the log writer for the editor and the assistant

import os
import re
from operator import itemgetter
from functools import partial
import concurrent.futures
import uuid
import json
import numpy as np

os.environ['HF_HOME'] = '/data1/demobot/hf'
# Type nvidia-smi in the terminal and choose two GPUs that is not being used, remember that we can only use 0,1,2,3,4
cuda_llama = 0
cuda_llava = 4
os.environ["CUDA_VISIBLE_DEVICES"] = f"{cuda_llama},{cuda_llava}"
os.environ["TRANSFORMERS_CACHE"] = '/data1/demobot/hf'

import scripts.utils as utils
import streamlit as st

# import torch
from huggingface_hub import login
from sentence_transformers import SentenceTransformer
# from sentence_transformers.quantization import quantize_embeddings,semantic_search_usearch
from llama_cpp import Llama
# from llama_cpp.llama_chat_format import Llama3VisionAlpha

import ast
import requests
from PIL import Image
from io import BytesIO
import base64

from dotenv import load_dotenv
load_dotenv()

#login hf key to use llama models
HF_TOKEN = os.getenv("HF_TOKEN")
YOU_API_KEY = os.getenv("YOU_API_KEY")
login(HF_TOKEN)

class LLM_SETUP:
    def __init__(self):
        self.llama_model_id = "/data1/demobot/hf/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf"
        
    def setup_llm(self, llm_name):
        if llm_name == 'llama-3':
            @st.cache_resource
            def load_llm():
                llm = Llama(
                    model_path=self.llama_model_id,
                    chat_format = 'llama-3',
                    n_gpu_layers=-1,
                    device = f"cuda:{cuda_llama}",
                    verbose = False,
                    main_gpu = 0,
                    n_ctx = 0
                    # seed=1337, # Uncomment to set a specific seed
                    )
                return llm
            return load_llm()
        else:
            raise ValueError(f"LLM {llm_name} not found")



class AIEditor:
    def __init__(self, llm):
        self.cache_dir = '/data1/demobot/hf'
        self.YOU_API_KEY = os.getenv("YOU_API_KEY")
        self.DEMOBOT_HOME = os.getenv("DEMOBOT_HOME")
        self.llm = llm
        
        # TODO: define the log file for the editor
        if not os.path.exists(f"{self.DEMOBOT_HOME}/logs/news/{st.session_state.session_id}/"):
            os.makedirs(f"{self.DEMOBOT_HOME}/logs/news/{st.session_state.session_id}/")
            
        with open(f"{self.DEMOBOT_HOME}/logs/news/{st.session_state.session_id}/AI_editor.txt", 'w') as file:
                file.write(f"START\n")
                dashes = '-' * 20
                file.write(f"{dashes}\n")
                
    # Load system prompts
    def load_sys_prompts(self, prompt_name):
        if prompt_name == 'AI_editor':
            with open(f"{self.DEMOBOT_HOME}/prompts/news/sys_prompt_editor.json", 'r') as file:
                prompt = json.load(file)
        return prompt
    
    def list_out_of_num_list(self, input):
        numbered_list = re.findall(r'\d+\.\s+(.+)', input)
        return numbered_list

    def log_writer_AI_editor(self, log_message, log_type):
        # TODO: change to news logs, conditioning on the type of generation task
        if log_type == 'messages_sent':
            with open(f"{self.DEMOBOT_HOME}/logs/news/{st.session_state.session_id}/AI_editor.txt", 'a') as file:
                for msg in log_message:
                    file.write(f"+++{msg['role']}: {msg['content']}\n")
                file.write(f"\n\n")
                dashes = '-' * 20
                file.write(f"{dashes}\n\n") 
                
        if log_type == 'raw_response':
            with open(f"{self.DEMOBOT_HOME}/logs/news/{st.session_state.session_id}/AI_editor.txt", 'a') as file:
                file.write(f"{log_message}\n\n")
                dashes = '-' * 20
                file.write(f"{dashes}\n\n") 
                
        if log_type == 'final_response':
            with open(f"{self.DEMOBOT_HOME}/logs/news/{st.session_state.session_id}/AI_editor.txt", 'a') as file:
                file.write(f"{log_message}\n\n")
                dashes = '-' * 20
                file.write(f"{dashes}\n\n")  
                
    def prepare_messages(self, conversation=None, previous_article=None):
        sys_prompt = self.load_sys_prompts('AI_editor')
        sys_prompt = {"role": "system", "content": f"{sys_prompt}"}
        user_prompt = {"role": "user", "content": f"Conversation history: {conversation}\n\nPrevious article: {previous_article}"}
        messages = [sys_prompt, user_prompt]
        return messages
    
    def generate(self,prompts):
        response = self.llm.create_chat_completion(
            max_tokens = 256,
            temperature = 0.2,
            top_p=0.9,
            messages = prompts
            )
        return response['choices'][0]['message']['content']
    
    def format_response(self, response):
        response = re.sub(r'\$', '\$', response)
        if response.startswith('Assistant') or response.startswith('assistant'):
            return response[len('Assistant: '): ]
        else:
            return response
    
    def prepare_and_generate(self, conversation, previous_article):
        messages = self.prepare_messages(conversation=conversation, previous_article=previous_article)
        self.log_writer_AI_editor(messages, log_type='messages_sent')
        
        response = self.generate(messages)
        self.log_writer_AI_editor(response, log_type='raw_response')
        
        response = self.format_response(response)
        self.log_writer_AI_editor(response, log_type='final_response')
        return response
    
# TODO: define generation with internet call for the editor or define a new class (agent) for the investigator

class AIEditorAssistant:
    def __init__(self, llm):
        self.llm = llm
        self.DEMOBOT_HOME = os.getenv("DEMOBOT_HOME")
        
        if not os.path.exists(f"{self.DEMOBOT_HOME}/logs/news/{st.session_state.session_id}/AI_editor_assistant/"):
            os.makedirs(f"{self.DEMOBOT_HOME}/logs/news/{st.session_state.session_id}/AI_editor_assistant/")
            
        with open(f"{self.DEMOBOT_HOME}/logs/news/{st.session_state.session_id}/AI_editor_assistant.txt", 'w') as file:
                file.write(f"START\n")
                dashes = '-' * 20
                file.write(f"{dashes}\n")
                
    def load_sys_prompts(self, prompt_name):
        if prompt_name == 'AI_editor_assistant':
            with open(f"{self.DEMOBOT_HOME}/prompts/news/sys_prompt_editor_assistant.json", 'r') as file:
                prompt = json.load(file)
        return prompt
    
    def log_writer_AI_assistant(self, log_message, log_type):
        with open(f"{self.DEMOBOT_HOME}/logs/news/{st.session_state.session_id}/AI_editor_assistant.txt", 'a') as file:
    
            if log_type == 'messages_sent':
                file.write(f"\nMessages sent to generation:\n\n")
                for msg in log_message:
                    file.write(f"+++{msg['role']}: {msg['content']}\n")
                file.write(f"\n\n")
                dashes = '-' * 20
                file.write(f"{dashes}\n\n")
                
            if log_type == 'raw_response':
                file.write(f"\nRaw response from News Assistant:\n\n")
                file.write(f"{log_message}\n\n")
                dashes = '-' * 20   
                file.write(f"{dashes}\n\n")
                
            if log_type == 'final_response':
                file.write(f"Final response from News Assistant:\n\n")
                file.write(f"{log_message}\n\n")
                dashes = '-' * 20
                file.write(f"{dashes}\n\n")
    
    # def format_user_sys_prompts(self, conversation, previous_article):
    #     # TODO: The prompt here includes instruction on how to deal with question such as "who created you", artificially pointing to stevens
        
    #     return user_prompt, sys_prompt
    
    def messages_builder(self, user, previous_article):
        sys_prompt = self.load_sys_prompts('AI_editor_assistant')
        messages = [sys_prompt]

        for msg in st.session_state["messages"][-6:-1]:
            messages.append({
                            'role': msg['role'],
                            'content': msg['content']
                            })
        
        user_prompt = {"role": "user", "content": f"Last user request: {user}\n\Current article: {previous_article}"}
        messages.append(user_prompt)
        
        return messages
    
    def generate(self, messages):
        response = self.llm.create_chat_completion(
            max_tokens = 256,
            temperature = 0.2,
            top_p=0.9,
            messages = messages
            )
        return response['choices'][0]['message']['content']
    
    def format_response(self, response):
        response = re.sub(r'\$', '\$', response)
        if response.startswith('Assistant') or response.startswith('assistant'):
            return response[len('Assistant: '): ]
        else:
            return response
        
    def prepare_and_generate(self, user, previous_article):
        # user_prompt, sys_prompt = self.format_user_sys_prompts(conversation, previous_article) 
        messages = self.messages_builder(user, previous_article)
        self.log_writer_AI_assistant(messages, log_type='messages_sent')
        
        response = self.generate(messages)
        self.log_writer_AI_assistant(response, log_type='raw_response')
        
        response = self.format_response(response)
        self.log_writer_AI_assistant(response, log_type='final_response')
        return response
    
    

    
        
        
        
        
        
        

    # # Receives a theme and returns a short factual story.
    # # TODO implement a version that already creates this first story using search.
    # def generate_short_story(self, theme):
    #     short_story_prompt = PromptTemplate.from_template(
    #         """You are a Journalist of a digital newspaper. Given the theme, it is your job to write a factual short story (5 lines).

    #         Theme: {theme}
    #         Short Story:"""
    #     )
        


    #     chain = short_story_prompt | self.llm_api('gpt-3.5-turbo-1106') 

    #     return self.chain_invoke(chain, {"theme": theme})


    # # Given a short story through the user UI,
    # # generates the baseline article, identify its factual statements
    # # return a dict with 'article', 'factual_statements'
    # def generate_article_and_statements(self, story):

    #     # Prompt to generate a baseline article about the short story provided
    #     article_prompt = PromptTemplate.from_template(
    #         """You are a Journalist of a digital newspaper. Given the Short Story, it is your job to write a long, engaging and factual news article about this story. First search the internet for content related to the Short Story, then write the long article based on the search results.

    #         Short Story: {story}
    #         News article:"""
    #     )

    #     # Prompt to generate a baseline article about the short story provided
    #     article_prompt_2 = PromptTemplate.from_template(
    #         """You are a Journalist of a digital newspaper. Given the Initial Article, it is your job to extend it. Write an extremely long, engaging and factual news article about this story. First search the internet for related content, then greatly extend the long article based on the search results.

    #         Initial Article: {article}
    #         Long News article:"""
    #     )

    #     # Prompt to identify the factual statements contained within the generated baseline article
    #     factual_id_prompt = PromptTemplate.from_template(
    #         """You are a factual checker critic for a digital newspaper. Given the News Article, it is your job to identify the factual statements written in the article. Only if necessary, you can slightly rewrite a factual statement to include relevant information contained in the rest of the article in order to achieve a comprehensive and independent phrase.

    #         News Article: {article_extended}
    #         Factual Statements:"""
    #     )

        
    #     # Subchain for article
    #     article_provider = article_prompt | self.agent_api('gpt-4')| {"article": RunnablePassthrough()}
    #     # Extending the initial article
    #     article_provider_2 = article_prompt_2 | self.agent_api('gpt-4')
    #     # Subchain for factual statements
    #     factual_statements_provider = factual_id_prompt | self.llm_api('gpt-3.5-turbo-1106')
        

    #     # Joining in a big chain
    #     chain = (
    #             article_provider
    #             | {'article': itemgetter('article'), 'article_extended': article_provider_2} 
    #             | {'article_extended': itemgetter('article_extended'), 'factual_statements': factual_statements_provider})

    #     return self.chain_invoke(chain, {"story": story})
    

    # # Given a baseline article and its factual statements in the form of a list,
    # # search in the web to check for their factualness
    # # and return the reviews in the form of a list
    # def statement_checker_chain(self, article, statement_list):
    #     factual_checker_prompt = PromptTemplate.from_template(
    #         """You are a factual checker for a digital newspaper. Given News Article and the Statement below, search on the internet and provide a very short review about the factualness of the statement. Use elements of the article to improve search results.
            
    #         News Article: {article}
            
    #         Statement: {statement}
            
    #         Short Review:""")

    #     # processing the str numbered list into a list obj
    #     statement_list = self.list_out_of_num_list(statement_list)

    #     statement_checker = factual_checker_prompt | self.agent_api('gpt-3.5-turbo-1106')

    #     # Use concurrent.futures to make parallel calls
    #     # with concurrent.futures.ThreadPoolExecutor() as executor:

    #     #     # Create a list of futures for each statement
    #     #     futures = [executor.submit(self.chain_invoke, statement_checker, {'article': article, 'statement': statement}) for statement in statement_list]

    #     #     # Wait for all futures to complete
    #     #     concurrent.futures.wait(futures)

    #     #     # Get the results from completed futures
    #     #     reviews_list = [future.result() for future in futures]

    #     reviews_list = []
    #     for statement in statement_list:
    #         reviews_list.append(statement_checker.invoke({'article': article, 'statement': statement}))  


    #     return reviews_list



    # # First part of refining the article is to double check its grammar
    # # receives the baseline article
    # # returns the grammar checked one 
    # def grammatical_chain_invoke(self, context, config):
    #     grammar_prompt = PromptTemplate.from_template(
    #         "Refine this article for any grammatical errors:\n\n{context}")
        
    #     # Prompt constructor to first insert the baseline article and then incrementally insert a new statement+review and refine the baseline article
    #     document_prompt = PromptTemplate.from_template("{page_content}")
    #     partial_format_doc = partial(format_document, prompt=document_prompt)

    #     # Grammar chain
    #     grammar_chain = {"context": partial_format_doc} | grammar_prompt | self.llm_api('gpt-3.5-turbo-1106') | StrOutputParser()


    #     return self.chain_invoke(grammar_chain,
    #                              context,
    #                              config)
    

    # # Given a version of the baseline article, a factual statement and its review,
    # # refines the article
    # # returns the refined version
    # # TODO abstract more
    # def refine_article_chain(self, baseline_article:str, statement_list:str, reviews_list:list[str]):

    #     # Prompt constructor to first insert the baseline article and then incrementally insert a new statement+review and refine the baseline article
    #     document_prompt = PromptTemplate.from_template("{page_content}")
    #     partial_format_doc = partial(format_document, prompt=document_prompt)

    #     # Refining prompt (considering each statement+review)
    #     refine_prompt = PromptTemplate.from_template(
    #     """Here's your baseline article: {prev_response}.

    #     Now refine the baseline article considering the factualness review of the below statement. If its factual, do not add anything, but if its not factual or inconclusive, take away just the statement from the News Article, adapting the remaining text.
    #     {context}

    #     Refined Article:""")

    #     # Refining chain
    #     refine_chain = (
    #     {
    #         "prev_response": itemgetter("prev_response"),
    #         "context": lambda x: partial_format_doc(x["doc"]),
    #     }
    #     | refine_prompt
    #     | self.llm_api('gpt-3.5-turbo-1106')
    #     | StrOutputParser()
    #     )

    #     # Refining function to loop over all statements
    #     # TODO abstract the function to refine any document, given other documents
    #     def refine_loop(docs):
    #         with trace_as_chain_group("refine loop", inputs={"input": docs}) as manager:

    #             refine = self.grammatical_chain_invoke(
    #                 docs[0], config={"callbacks": manager, "run_name": "initial grammar check"})
                
    #             for i, doc in enumerate(docs[1:]):
    #                 refine = refine_chain.invoke(
    #                 {"prev_response": refine, "doc": doc},
    #                 config={"callbacks": manager, "run_name": f"refine {i}"},
    #             )
    #             manager.on_chain_end({"output": refine})
    #         return refine


    #     # Building the list that is going to be iterated by the refine_loop function,
    #     # the first element is the baseline article itself to be used by the grammar chain,
    #     # then each instance of statement+review
    #     statement_and_check = ['Statement: ' + statement + '\n' + 'Review: ' + review['output']
    #                             for statement, review in zip(statement_list, reviews_list)]

    #     # Inserting baseline article as first doc
    #     statement_and_check.insert(0, baseline_article)

    #     # Building a Langchain document out of the list
    #     docs = [Document(page_content = item)
    #             for item in statement_and_check]

    #     # Applying the refining loop
    #     refined_article = refine_loop(docs)
    #     # Basic clean of \ns
    #     refined_article = re.sub('\n\n', ' ', refined_article)

    #     return refined_article


    # # Given a refined article
    # # search in the web for real testimonies about the discussed subject
    # # returns two testimonies or nothing
    # def generate_testimonies(self, refined_article):
    #     real_testimonies_prompt = PromptTemplate.from_template(
    #         """You are a Journalist of a digital newspaper. Given the News Article below, it is your job to search the internet for two relevant real testimonies about the content of the article, also output the name of the person who provided it. Review your observations to check if the testimonies are good enough.

    #         News Article: {article}

    #         If only 1 testimony is found, output it. If none, just output: no testimonies

    #         Person 1:
    #         Testimony 1:

    #         Person 2:
    #         Testimony 2:"""
    #     )

    #     testimony_chain = real_testimonies_prompt | self.agent_api('gpt-3.5-turbo-1106')

    #     return self.chain_invoke(testimony_chain, {"article": refined_article})



    # # Given two testimonies
    # # Take a look into the observations (web results)
    # # Refine the testiminies
    # # Return the reviewed ones.
    # def refine_testimonies(self, refined_article, observations, testimonies):
    #     refine_testimonies_prompt = PromptTemplate.from_template(
    #         """You are a Journalist of a digital newspaper. Given the News Article below, the Observations found in an internet research and the Extracted Testimonies, it is your job to review the observations to check if the testimonies are good enough. You can include or exclude details as necessary to make the testimonies fully comprehensive.

    #         News Article: {article}

    #         Observations: {observations}

    #         Extracted testimonies: {testimonies}

    #         If no testimonies could be found, please just output 'no testimonies found'. But if only one testimony is found, output it.
    #         - Reviewed testimonies -

    #         Person 1:
    #         Testimony 1:

    #         Person 2:
    #         Testimony 2:"""
    #     )
    #     refine_testimonies_chain = refine_testimonies_prompt | {"refined_testimonies": self.llm_api('gpt-3.5-turbo')} 
    #     return self.chain_invoke(refine_testimonies_chain,
    #                              {"article": refined_article,
    #                               "observations": observations,
    #                               "testimonies": testimonies})


    # # Given a refined article and two testimonies
    # # integrate them into the article
    # # return the resulting article
    # def integrate_testimonies(self, refined_article, refined_testimonies):
    #     integrate_testimonies_prompt = PromptTemplate.from_template(
    #         """You are a Journalist of a digital newspaper. Given the News Article and Real Testimonies below, it is your job to integrate and combine them into the article, finding the best position to fit them in while maintaining and adapting the context.

    #         News Article: {article}

    #         Real Testimonies: {refined_testimonies}

    #         If there are no found testimonies to integrate, just repeat the article.
    #         Combined News Article:"""
    #     )

    #     integrate_testimonies_chain = integrate_testimonies_prompt | {"article_testimonies": self.llm_api('gpt-4')}

    #     return self.chain_invoke(integrate_testimonies_chain,
    #                              {"article": refined_article,
    #                               "refined_testimonies": refined_testimonies})


    # # Given the prefinal article
    # # Format the design, including title, sections..
    # # return the formatted article
    # def format_article(self, combined_article_testimonies):
    #     format_prompt = PromptTemplate.from_template(
    #         """You are a Journalist designer of a digital newspaper. Given the News Article below, it is your job to format the text into a News Article design, giving it a title, splitting paragraphs/sections and making final revisions for repetitive information.

    #         News Article: {combined_article}

    #         Formatted News Article:"""
    #     )

    #     format_design_chain = format_prompt | {'formatted_article': self.llm_api('gpt-3.5-turbo-1106')}
    #     return self.chain_invoke(format_design_chain,
    #                              {"combined_article": combined_article_testimonies})



    # # Global function that receives an input from the user
    # # applies all chains to transform the short story into a full article
    # # return the final article to the user UI
    # def generate_response(self, text):
    #     self.set_openai_key()
    #     self.set_serper_api_key()

    #     # deactivated for now
    #     # input -> theme to build a short story - str
    #     # output -> short_story_response['story'] - dict
    #     # short_story_response = self.generate_short_story("Your theme here")
      

    #     # input -> user's short story - str
    #     # processes -> build a baseline article out of the short story and identify its factual statements
    #     # outputs ->  article_and_statements_response = {'article', 'factual_statements'} - dict
    #     article_and_statements_response = self.generate_article_and_statements(text)
    #     baseline_article, factual_statements = article_and_statements_response['article_extended']['output'], article_and_statements_response['factual_statements'].content


    #     # input -> baseline article, its factual statements - str
    #     # processes -> str numbered list into list obj with the statements
    #     # outputs -> reviews_list with each statement review - lst
    #     reviews_list = self.statement_checker_chain(baseline_article,
    #                                                 factual_statements)


    #     # input -> baseline article, factual statements, its reviews - str
    #     # processes -> apply grammar check in the baseline article and refine it considering each statement's review
    #     # output -> refined article - dict
    #     refined_article = self.refine_article_chain(baseline_article, factual_statements, reviews_list)
    #     # refined_article = refined_article['article']
        

    #     # input -> refined article - str
    #     # processes -> search in the web for real testimonies about the discussed subject in the article
    #     # output -> testimonies_response = {"intermediate_steps", "testimonies"} - dict
    #     testimonies_response = self.generate_testimonies(refined_article)
    #     testimonies = testimonies_response['output']
    #     full_observations = testimonies_response['intermediate_steps']

    #     # retrieving onlt observation from intermediate steps
    #     observations = ''
    #     for i in full_observations:
    #         observations += i[1] + '\n'

        
    #     if testimonies == 'no testimonies':
    #         article_testimonies = refined_article

    #     else:

    #         # input -> refined article - str
    #         # processes -> search in the web for real testimonies about the discussed subject in the article
    #         # output -> refined article = {"refined_testimonies"} - dict
    #         refined_testimonies_response = self.refine_testimonies(refined_article,
    #                                                       observations,
    #                                                       testimonies)
    #         refined_testimonies = refined_testimonies_response['refined_testimonies'].content


    #         # input -> refined_article, refined_testimonies - str
    #         # processes -> integrate the found testimonies into the refined article
    #         # output -> article wih testimonies = {"article_testimonies":} - dict
    #         article_testimonies_response = self.integrate_testimonies(refined_article,
    #                                                          refined_testimonies)

    #         article_testimonies = article_testimonies_response["article_testimonies"].content


    #     # input -> refined_article integrated with testimonies - str
    #     # processes -> edit the design to adapt for News, including sections, title..
    #     # output -> final formated article = {"article_testimonies":} - dict
    #     formatted_article_response = self.format_article(article_testimonies)
    #     formatted_article = formatted_article_response['formatted_article'].content


    #     # Return the final response
    #     return formatted_article
