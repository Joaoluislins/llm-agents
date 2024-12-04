# import os
# import uuid
# import re
# import json

# os.environ['HF_HOME'] = '/data1/demobot/hf'
# # Type nvidia-smi in the terminal and choose two GPUs that is not being used, remember that we can only use 0,1,2,3,4
# cuda_llama = 0
# cuda_llava = 4
# os.environ["CUDA_VISIBLE_DEVICES"] = f"{cuda_llama},{cuda_llava}"
# os.environ["TRANSFORMERS_CACHE"] = '/data1/demobot/hf'
# from scripts.data_index import BookDataIndex
# import scripts.utils as utils
# import streamlit as st

# import torch
# from huggingface_hub import login
# from sentence_transformers import SentenceTransformer
# from sentence_transformers.quantization import quantize_embeddings,semantic_search_usearch
# from llama_cpp import Llama
# from llama_cpp.llama_chat_format import Llama3VisionAlpha

# import ast
# import requests
# from PIL import Image
# from io import BytesIO
# import base64

# from dotenv import load_dotenv
# load_dotenv()

# #login hf key to use llama models
# HF_TOKEN = os.getenv("HF_TOKEN")
# login(HF_TOKEN)

# # Page configs
# st.set_page_config(page_title="🗃️ Book Retailing Assistant",
#                 page_icon="🗃️",
#                 layout='wide')

# DEMOBOT_HOME = os.getenv("DEMOBOT_HOME")
# # Loading CSS Style
# with open(os.path.join(DEMOBOT_HOME, "scripts/style.css"), "r") as f:
#     css = f.read()

# st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

# # Defining CSS Style for chat
# st.markdown(
#     """
#     <style>
#     [data-testid="stChatMessageContent"] p{
#         font-size: 1.3rem;
#     }
#     </style>
#     """, unsafe_allow_html=True
# )

# st.header('🗃️ Book Retailing Assistant')
# st.header('\n')

# # st.write("""
# #         <div class="big-font">
# #             <p>
# #                 This is a conversational chatbot built to have access to a retailer books database in order to respond in an engaging and factual manner.
# #             </p>
# #         </div>""", unsafe_allow_html=True)


# class LLMRag:
#     def __init__(self):
#         self.model_id = "/data1/demobot/hf/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf"
#         self.model_id_llava = "/data1/demobot/hf/llava-llama-3-8b-v1_1-int4.gguf"
#         # self.embed_model_id = "mixedbread-ai/mxbai-embed-large-v1"
#         self.embed_model_id = "mixedbread-ai/mxbai-embed-large-v1"
#         self.image_converted_folder_path = "/data1/demobot/images_converted"
#         self.cache_dir = '/data1/demobot/hf'
#         self.chat_history = ''
#         self.sys_prompt_chat = self.load_sys_prompts('chat')
#         self.sys_prompt_create_query = self.load_sys_prompts('create_query')
#         self.sys_prompt_respond_query = self.load_sys_prompts('respond_query')
#         self.sys_prompt_summary = self.load_sys_prompts('summary')
#         self.sys_prompt_llava = self.load_sys_prompts('llava')
#         self.sys_prompt_router = self.load_sys_prompts('router')
#         # Define the standard image dimensions
#         self.target_width = 256
#         self.target_height = 512
        
#         if not os.path.exists(f"{DEMOBOT_HOME}/logs/flow_conversation/"):
#             os.makedirs(f"{DEMOBOT_HOME}/logs/flow_conversation/")
            
#         with open(f"{DEMOBOT_HOME}/logs/flow_conversation/{st.session_state.session_id}.txt", 'w') as file:
#                 file.write(f"START\n")
#                 dashes = '-' * 20
#                 file.write(f"{dashes}\n")
        
        
#         self.messages = []


#     def load_sys_prompts(self, prompt_name):
#         if prompt_name == 'chat':
#             with open(f"{DEMOBOT_HOME}/prompts/books/sys_prompt_chat_llmrag2.json", 'r') as file:
#                 prompt = json.load(file)

#         elif prompt_name == 'create_query':
#             with open(f"{DEMOBOT_HOME}/prompts/books/sys_prompt_create_query_llmrag.json", 'r') as file:
#                 prompt = json.load(file)

#         elif prompt_name == 'respond_query':
#             with open(f"{DEMOBOT_HOME}/prompts/books/sys_prompt_query_respond_llmrag.json", 'r') as file:
#                 prompt = json.load(file)

#         elif prompt_name == 'summary':
#             with open(f"{DEMOBOT_HOME}/prompts/books/sys_prompt_summary_llmrag.json", 'r') as file:
#                 prompt = json.load(file)

#         elif prompt_name == 'llava':
#             with open(f"{DEMOBOT_HOME}/prompts/books/sys_prompt_llava_llmrag.json", 'r') as file:
#                 prompt = json.load(file)
        
#         elif prompt_name == 'router':
#             with open(f"{DEMOBOT_HOME}/prompts/books/sys_prompt_router_llmrag.json", 'r') as file:
#                 prompt = json.load(file)
        
#         return prompt


#     def setup_data_index_embeddings(self):
#         @st.cache_resource
#         def load_data_index_embeddings():

#             # data_loader = BookDataIndex(
#             #     index_path = f"{DEMOBOT_HOME}/data/books/indexes/index_concat_1_10_usearch_int8_binary.index",
#             #     data_path = f"{DEMOBOT_HOME}/data/books/concatenate1-10_fulltext_images.hf",
#             #     embeddings_path = f"{DEMOBOT_HOME}/data/books/indexes/embeddings_concatenate_1_10.npy"
#             #     )
            
#             data_loader = BookDataIndex(
#                 index_path = f"/data1/demobot/data/books/indexes/index_concat_1_10_usearch_int8_binary.index",
#                 data_path = f"/data1/demobot/data/books/concatenate1-10_fulltext_images.hf",
#                 embeddings_path = f"/data1/demobot/data/books/indexes/embeddings_concatenate_1_10.npy"
#                 )

#             index = data_loader.load_index()
#             data = data_loader.load_data()
#             embeddings = data_loader.load_embeddings()

#             return data, index, embeddings
#         return load_data_index_embeddings()
    
#     def setup_llm(self):
#         @st.cache_resource
#         def load_llm():
#             llm = Llama(
#                 model_path=self.model_id,
#                 chat_format = 'llama-3',
#                 n_gpu_layers=-1,
#                 device = f"cuda:{cuda_llama}",
#                 verbose = False,
#                 main_gpu = 0,
#                 n_ctx = 0
#                 # seed=1337, # Uncomment to set a specific seed
#                 )
#             return llm
       
#         return load_llm()
    
#     def setup_llava(self):
#         @st.cache_resource
#         def load_llava():
#             chat_handler = Llama3VisionAlpha(clip_model_path="/data2/joao/hf/llava-llama-3-8b-v1_1-mmproj-f16.gguf")
#             llava = Llama(
#               model_path=self.model_id_llava,
#               chat_handler=chat_handler,
#               chat_format='llama-3-vision-alpha',
#               n_gpu_layers=-1,
#               n_ctx=0,
#               verbose=False,
#               temperature = 0.2,
#               max_tokens = 256,
#               device = f"cuda:{cuda_llava}"
#               # n_ctx=2048, # n_ctx should be increased to accommodate the image embedding
#             )
#             return llava
#         return load_llava()


#     def setup_embed_model(self):
#         @st.cache_resource
#         def load_embed_model():
#             embed_model = SentenceTransformer(self.embed_model_id)
#             device = torch.device("cuda:0")
#             embed_model.to(device)
#             return embed_model
#         return load_embed_model()


#     def usearch_helper(self, queries, k):
#         query_embeddings = embed_model.encode(queries, normalize_embeddings=True)

#         results, search_time = semantic_search_usearch(
#             query_embeddings,
#             corpus_index=index,
#             # corpus_embeddings=corpus_embeddings if corpus_index is None else None,
#             corpus_precision='binary',
#             top_k=k,
#             calibration_embeddings=embeddings,
#             rescore='binary' != "float32",
#             rescore_multiplier=4,
#             exact=False,
#             # output_index=True,
#             )

#         context = ''
#         images = ''
#         for idx, entry in enumerate(results[0]):
#             context += f"Book {idx+1}:\n"
#             context += f"{data[entry['corpus_id']]['full_text']}\n"
#             images += f"{data[entry['corpus_id']]['images']}\n"

#         return context, images
    

#     def format_search_query(self, bot_response):
#         search_query_formatted = None
#         pattern = r"Search Query: (.*)"

#         match = re.search(pattern, bot_response)
#         if match:
#             search_query_formatted = match.group(1).strip()

#         return search_query_formatted
    
    
#     # TODO analyze if there is a need to include in the system prompt additional details about the user request
#     def summarize_query_result(self, query_result):

#         # TODO if the user ask to create a query in first call, it didnt used the last turns of the conversation to produce the query
#         user_prompt_summarize, sys_prompt_summary = self.format_user_sys_prompts(user = query_result, type = 'summary')
#         messages_summarize = self.messages_builder(sys_prompt_summary, user_prompt_summarize, type = 'summary')
#         summary_response_raw = self.generate(messages_summarize)
#         summary_response = self.format_response(summary_response_raw)
#         return summary_response
    

#     def search_summarize_images(self, user):

#         # Create Query
#         user_prompt_create_query, sys_prompt_create_query = self.format_user_sys_prompts(user, type = 'create_query')
#         messages_create_query = self.messages_builder(sys_prompt_create_query, user_prompt_create_query, type = 'create_query')
#         search_query_raw = self.generate(messages_create_query)
#         # Retrieve the actual Query
#         search_query = self.format_response(search_query_raw)
#         with open(f"{DEMOBOT_HOME}/logs/flow_conversation/{st.session_state.session_id}.txt", 'a') as file:
#                 file.write(f"Query created: {search_query}\n\n")
#                 dashes = '-' * 20
#                 file.write(f"{dashes}\n") 

#         # Search
#         result_books_query, images = self.usearch_helper(queries= [search_query, 'hi'], k = 2)
#         ### Below is an attempt to summarize them query result in order to reduce the usage 
#         ### of the context due to large query results
#         # summary_result_books_query = self.summarize_query_result(result_books_query)
#         # summary_result_books_query =re.sub('\n\n', '\n', summary_result_books_query)
#         # with open('flow_of_conversation.txt', 'a') as file:
#         #         file.write(f"Result of query: {result_books_query}\n")
#         #         file.write(f"Summary of query result: {summary_result_books_query}\n")
        

#         images_result_list = self.image_url_handler(images)

#         return result_books_query, None, images_result_list

# #                 with open('query_result.txt', 'a') as file:
# #                     file.write('\nSearch Query: ' + search_query + '\n')

# #                     file.write('Original Query Result:\n')
# #                     file.write(result_books_query)

# #                     file.write('\nSummarized Query Result:\n')
# #                     file.write(summary_result_books_query + '\n\n')



    
#     def format_user_sys_prompts(self, user, type):

#         if type == 'router':
#             fig_description = None
#             item_summary = None
#             user_prompt = {"role": "user", "content": f"{user}"}

#             # TODO format chat sys prompt to include Inventory Context when the last 5 turns contained any retrieval, if two, retrieve both
#             sys_prompt = self.load_sys_prompts('router')
#             content_from_last_retrievals = st.session_state["messages"]
#             # print(content_from_last_retrievals)
#             if content_from_last_retrievals:
#                 content_from_last_retrievals = [msg for msg in content_from_last_retrievals if msg['role'] == 'assistant']
#                 for msg in content_from_last_retrievals:
#                     fig_description = msg.get('fig_description', None)
#                     item_summary = msg.get('items_summary', None)
#                     if fig_description and item_summary:
#                         sys_prompt = self.load_sys_prompts('router')
#                         sys_prompt['content'] = sys_prompt['content'].format(f"{item_summary}\nBook Cover/Image currently being shown:{fig_description}.")
            
#             return user_prompt, sys_prompt
        
#         elif type == 'chat':
#             fig_description = None
#             item_summary = None
#             user_prompt = {"role": "user", "content": f"{user}"}

#             # TODO format chat sys prompt to include Inventory Context when the last 5 turns contained any retrieval, if two, retrieve both
#             sys_prompt = self.load_sys_prompts('chat')
#             content_from_last_retrievals = st.session_state["messages"]
#             if content_from_last_retrievals:
#                 content_from_last_retrievals = [msg for msg in content_from_last_retrievals if msg['role'] == 'assistant']
#                 for msg in content_from_last_retrievals:
#                     fig_description = msg.get('fig_description', None)
#                     item_summary = msg.get('items_summary', None)
#                     if fig_description and item_summary:
#                         sys_prompt = self.load_sys_prompts('chat')
#                         sys_prompt['content'] = sys_prompt['content'].format(f"{item_summary}\nBook Cover/Image currently being shown:{fig_description}.")
            
#             return user_prompt, sys_prompt
        
#         elif type == 'create_query':
#             user_prompt = {"role": "user", "content": f"{user}"}
#             # user_prompt = {"role": "user", "content": f"Please create a search query."}
#             sys_prompt = self.sys_prompt_create_query
#             return user_prompt, sys_prompt
        
#         elif type == 'respond_query':
#             user_prompt = {"role": "user", "content": "Nice!"}
#             # have to include the retrieved documents in the system message
#             sys_prompt = self.load_sys_prompts('respond_query')
#             sys_prompt['content'] = sys_prompt['content'].format(user)
#             return user_prompt, sys_prompt

#         elif type == 'summary':
#             user =re.sub('\n\n', '\n', user)
#             user_prompt = {"role": "user", "content": f"Books Information: {user}"}
#             sys_prompt = self.sys_prompt_summary
#             return user_prompt, sys_prompt

    
    
#     def messages_builder(self, sys_prompt, user_prompt, type):
#         ### types
#         # router -> have all conversation context 
#         # chat -> have all conversation context
#         # create query -> have all conversation context
#         # summarize -> only has sys prompt and user prompt
#         # respond query -> have all conversation context
#         messages = [sys_prompt]

#         if type in ['router', 'chat', 'create_query', 'respond_query']:

#             for msg in st.session_state["messages"][-6:]:
#                 messages.append({
#                                  'role': msg['role'],
#                                  'content': msg['content']
#                                  })
            
#             # case of not first message
#             if messages[-1]['role'] == 'user':
#                 if type != 'respond_query':
#                     messages[-1] = user_prompt
#                 # if type is respond query, we are injecting in the context that the bot searched
#                 else:
#                     messages.append({'role': 'assistant', 'content':'Let me search in our inventory!'})
#                     messages.append(user_prompt)

            
#             # first message case
#             else:
#                 messages.append(user_prompt)

#         # case for summary type
#         else:
#             messages.append(user_prompt)


#         with open(f"{DEMOBOT_HOME}/logs/flow_conversation/{st.session_state.session_id}.txt", 'a') as file:
#             file.write(f"\nMessages sent to '{type}' generation:\n\n")
#             for msg in messages:
#                 file.write(f"+++{msg['role']}: {msg['content']}\n")
#             file.write(f"\n\n")
#             dashes = '-' * 20
#             file.write(f"{dashes}\n") 
            

#         return messages
            
    
#     def generate(self, messages):

#         response = llm.create_chat_completion(
#             max_tokens = 256,
#             temperature = 0.2,
#             top_p=0.9,
#             messages = messages
#             )
#         return response['choices'][0]['message']['content']
    

#     def format_response(self, response):
#         response =re.sub(r'\$', '\$', response)
#         if response.startswith('Assistant') or response.startswith('assistant'):
#             return response[len('Assistant: '): ]
#         else:
#             return response


#     def chatLlama3(self, user):

#         ### Generate router response
#         ### build prompts to route between response or query for lookup operation
#         user_prompt_router, sys_prompt_router = self.format_user_sys_prompts(user, type='router') 
#         # format all messages to include in the generation
#         messages_router = self.messages_builder(sys_prompt_router, user_prompt_router, type = 'router')
#         router_response = self.generate(messages_router)
#         router_response = self.format_response(router_response)
#         # TODO create writer function
#         with open(f"{DEMOBOT_HOME}/logs/flow_conversation/{st.session_state.session_id}.txt", 'a') as file:
#             file.write(f"Router decision: {router_response}\n\n")
#             dashes = '-' * 20
#             file.write(f"{dashes}\n") 

#         ### Retrieve decision and propagate next step

#         # in the case of SEARCH action
#         if router_response.startswith('SEARCH_SUGGEST') or router_response.startswith('SEARCH_ SUGGEST') or router_response.startswith('SUGGEST') or router_response.startswith('SEARCH'):
#             query_result, query_summary_result, images_result_list = self.search_summarize_images(user)
#             # TODO include in the chat history or query result document
#             # Produce the response based on the query result and conversation
#             # In this case, we give the summary as context to the bot
#             user_prompt_respond_query, sys_prompt_respond_query = self.format_user_sys_prompts(user = query_result, type = 'respond_query')
#             messages_respond_query = self.messages_builder(sys_prompt_respond_query, user_prompt_respond_query, type = 'respond_query')
#             query_response_raw = self.generate(messages_respond_query)
#             query_response = self.format_response(query_response_raw)
#             with open(f"{DEMOBOT_HOME}/logs/flow_conversation/{st.session_state.session_id}.txt", 'a') as file:
#                 file.write(f"Query Response: {query_response}\n\n")
#                 dashes = '-' * 20
#                 file.write(f"{dashes}\n") 
#             # TODO include the response in the chat history document
#             return query_response, images_result_list, query_result, query_summary_result
    

#         # in the case of simple chat
#         else:
#             # format chat prompts
#             user_prompt_chat, sys_prompt_chat = self.format_user_sys_prompts(user, type = 'chat')
#             # messages to be included in current context
#             messages_chat = self.messages_builder(sys_prompt_chat, user_prompt_chat, type = 'chat')
#             chat_response = self.generate(messages_chat)
#             chat_response = self.format_response(chat_response)
#             with open(f"{DEMOBOT_HOME}/logs/flow_conversation/{st.session_state.session_id}.txt", 'a') as file:
#                 file.write(f"chat response: {chat_response}\n\n")
#                 dashes = '-' * 20
#                 file.write(f"{dashes}\n") 
#             # TODO include the response in the chat history document
#             # self.messages.append({
#             #         "role": "assistant",
#             #         "content": f"{bot_response}"
#             #         })
#             # self.chat_history += f"Human: {user}" + '\n' + f"{bot_response}\n"
#             # return the response with image = None
#             return chat_response, None, None, None
                  
    
#     def is_file_in_folder(self, file_name, folder_path):
#         # Construct the full path of the file
#         file_path = os.path.join(folder_path, file_name)
#         # Check if the file exists and is a file
#         return os.path.isfile(file_path)
    

#     def image_to_base64_data_uri(self, file_path):
#         with open(file_path, "rb") as img_file:
#             base64_data = base64.b64encode(img_file.read()).decode('utf-8')
#             return f"data:image/png;base64,{base64_data}"

    
#     # TODO better structure variables of paths
#     def image_webp_to_jpg_converter(self, image_url):
#         # Download the WebP image from the URL
#         webp_url = image_url

#         # Check if already made the conversion earlier
#         image_converted_name = re.sub(r'https://', '', image_url)
#         image_converted_name = re.sub(r'webp', '', image_converted_name)
#         image_converted_name = re.sub(r'/', '-', image_converted_name)
        
#         folder_path = self.image_converted_folder_path
#         file_path = folder_path + image_converted_name

#         if self.is_file_in_folder(folder_path=folder_path,
#                                   file_name=image_converted_name):
#             return self.image_to_base64_data_uri(file_path=file_path)
        
#         else:
#             response = requests.get(webp_url)
#             response.raise_for_status()  # Ensure the request was successful

#             # Open the WebP image using Pillow
#             webp_image = Image.open(BytesIO(response.content))

#             # Convert and save the image in JPG format
#             webp_image.convert('RGB').resize((self.target_width, self.target_height)).save(file_path, 'JPEG')
#             data_uri = self.image_to_base64_data_uri(file_path = file_path)
#             return data_uri

    
#     # TODO Choose the image to show, currently, Im just displaying the first
#     def image_url_handler(self, images):
#         images_result_list = []
#         with open(f"{DEMOBOT_HOME}/logs/flow_conversation/{st.session_state.session_id}.txt", 'a') as file:
#                 file.write(f"IMAGES: {images}\n\n")
#                 dashes = '-' * 20
#                 file.write(f"{dashes}\n") 
#         for image in images.split('\n'):
#             if image:
#                 image_dict = ast.literal_eval(image)
#                 if image_dict:
#                     image_url = image_dict[0]['large']
#                 else:
#                     continue
#                 if image_url.endswith('webp_.jpg'):
#                     image_converted = self.image_webp_to_jpg_converter(image_url=image_url)
#                     images_result_list.append(image_converted) 

#                 else:
#                     image_converted_name = re.sub(r'https://', '', image_url)
#                     image_converted_name = re.sub(r'webp', '', image_converted_name)
#                     image_converted_name = re.sub(r'/', '-', image_converted_name)
#                     folder_path = self.image_converted_folder_path
#                     file_path = folder_path + image_converted_name

#                     if self.is_file_in_folder(folder_path=folder_path, file_name=image_converted_name):
#                         images_result_list.append(self.image_to_base64_data_uri(file_path=file_path))

#                     else:
#                         image_download = requests.get(image_url)
#                         if image_download.status_code == 200:
#                             image = Image.open(BytesIO(image_download.content))
#                             image.resize((self.target_width, self.target_height)).save(file_path, 'JPEG')
#                             images_result_list.append(self.image_to_base64_data_uri(file_path=file_path))
        

#         return images_result_list
        
    
#     def generate_llava_description(self, image):
        
#         response = llava.create_chat_completion(
#                     messages = [self.sys_prompt_llava, 
#                         {"role": "user",
#                          "content": [
#                             {"type" : "text", "text": "What is in this image?"},
#                             {"type": "image_url", "image_url": {"url": image}}
#                                 ]
#                             }
#                         ]
#                     )
#         return response['choices'][0]["message"]['content']
    
    
#     @utils.enable_chat_history
#     def main(self):
#         user = st.chat_input(placeholder="Ask me anything!")
#         if user:
#             utils.display_msg(user, 'user')
#             with st.chat_message("assistant"):

#                 with open(f"{DEMOBOT_HOME}/logs/flow_conversation/{st.session_state.session_id}.txt", 'a') as file:
#                     file.write(f"User New Input: {user}\n\n")
#                     dashes = '-' * 20
#                     file.write(f"{dashes}\n") 

#                 response, images_result_list, query_result, query_result_summary = self.chatLlama3(user)
#                 if images_result_list:
#                     st.image(images_result_list)
   
#                 st.write(response)

#                 fig_descriptions = None
#                 if images_result_list:
#                     fig_descriptions = ''
#                     for image in images_result_list:
#                         llava_response_raw = self.generate_llava_description(image)
#                         llava_response = self.format_response(llava_response_raw)
#                         llava_response =re.sub('\n\n', '\n', llava_response)
#                         fig_descriptions += llava_response + '\n'

#                 st.session_state.messages.append({"role": "assistant",
#                                                   "content": response,
#                                                   'fig': images_result_list, 
#                                                   'fig_description':fig_descriptions,
#                                                   'items_retrieved': query_result,
#                                                   'items_summary': query_result_summary})

# if "session_id" not in st.session_state:
#     st.session_state.session_id = str(uuid.uuid4())
    
# if "LLMRAG" not in st.session_state:
#     LLMRAG = LLMRag()
#     st.session_state.LLMRAG = LLMRAG


# llm = st.session_state.LLMRAG.setup_llm()
# llava = st.session_state.LLMRAG.setup_llava()
# data, index, embeddings = st.session_state.LLMRAG.setup_data_index_embeddings()
# embed_model = st.session_state.LLMRAG.setup_embed_model()
# st.session_state.LLMRAG.main()