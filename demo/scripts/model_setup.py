import streamlit as st
from groq import Groq

class LLM_SETUP:
    def __init__(self):
        self.llama_model_id = "/data1/demobot/hf/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf"
        self.groq_api_key = st.secrets["GROQ_API_KEY"]
        
    def setup_llm(self, llm_name):
    
    #     if llm_name == 'llama-3':
    #         @st.cache_resource
    #         def load_llm():
    #             llm = Llama(
    #                 model_path=self.llama_model_id,
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
    #     else:
    #         raise ValueError(f"LLM {llm_name} not found")
    
        if llm_name == 'groq':
                @st.cache_resource
                def load_groq_client():
                    client = Groq(api_key=self.groq_api_key)
                    return client
                return load_groq_client()
    
