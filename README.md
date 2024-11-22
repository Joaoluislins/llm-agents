# Chatbot Implementations

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](link/)

Some experimental Implementations of LLama3 8B designed to leverage personal Research in RAG, Reinforcement Learning and collection of Conversational Data.

## 💬 Architectures and Use Cases
Please find below a brief description of the current implementations:
-  **Basic Chatbot** \
    [Quantized (4bit) Checkpoint of Llama3 8B model](https://huggingface.co/bartowski/Meta-Llama-3-8B-Instruct-GGUF) built with Llama.cpp. This checkpoint is not finetuned in any specific dataset, and doesn't perform RAG. It excels in interactive general domain conversations. 

- **Book Shop Assistant -> Chatbot + LLMrouter + RAG** \
    Here I built a RAG on top of the above example, using the same model as a Router to decide on either performing a RAG (Search in database, Retrieve Documents, Generating with the help of additional information) or a normal reply. For the Search, I'm generating a Search Query with the same LLama3 model using the conversational context and performing the search with the library [usearch](https://github.com/unum-cloud/usearch). The database that this chatbot has access is a subset (books) of the [Amazon Reviews Dataset (2023)](https://amazon-reviews-2023.github.io/), so it is designed to provide books suggestions and help an user in the process of buying a book by delivering grounded information about reviews, plots, pricing, images of covers and etc.   

-  **Chatbot with Internet Access (In Dev)** \
  An internet-enabled chatbot capable of answering user queries about recent events.

-  **Chatbot that improves with time RL (In Dev)** \
  An implementation of [RLHF](https://arxiv.org/abs/2203.02155) tailored with personalized rewards.

## <img src="https://streamlit.io/images/brand/streamlit-mark-color.png" width="40" height="22"> Streamlit App
To help with the experiments, follows a basic User Interface, a multi-page streamlit app containing all sample chatbot use cases. \
You can access this app through this link: [link](https://langchain-chatbot.streamlit.app)

## 🖥️ Running locally
```shell
# Clone this repository using your terminal
$ git clone https://github.com/Joaoluislins/demo.git

# Branch your development
$ git checkout -b <branch_name>
# (Optional, push the new branch to the remote repository)
$ git push -u origin <branch_name>

# Navigate to the repository
$ cd demo
# Create a .env file with your environment paths and variables
# Include your Huggingface api token to access some models
$ echo "HF_TOKEN='your_huggingface_api_token'" > .env
# Define a default home path for the repository, follows an example, please replace the your_user_folder with your server username
$ echo "DEMOBOT_HOME='/home/your_user_folder/demo'" >> .env
# Add more variables as needed

# LLama.cpp is a tricky package to compile, so I strongly advise using this procedure in which you create a conda environment by cloning an already compiled lamma.cpp package.
$ conda create -n your_new_env_name python=3.10 -y
$ conda activate your_new_env_name
$ tar -xzvf /data1/demobot/llama_cpp.tar.gz -C ~/.conda/envs/your_new_env_name/lib/python3.10/site-packages/
# Now install the other packages through the requirements.txt
$ pip install -r requirements.txt

# Almost ready, Streamlit uses a local port to host the application, but the local in this case is the Stevens Server. Your Web Browser is running on your local (notebook) ip address. So you You need to link the server port that is hosting the application to a port on your local (notebook).

### ON YOUR LOCAL TERMINAL - YOUR NOTEBOOK TERMINAL (NOT IN SERVER)
# bind a port of the server to your local
# please don't use the following ports: 8506 or 8504
$ ssh -N -f -L available_port_number:localhost:available_port_number your_username@ai.cs.stevens.edu
# enter your server password

### ON THE STEVENS SERVER TERMINAL (NOT IN YOUR NOTEBOOK)
# start the application
$ cd demo # if your are not here already
$ streamlit run Home.py --server.port same_available_port_number

# You've done it, go to your browser and test the application in http://localhost:available_port_number/

# Please Update this guide to include your port in the used ones above.
```

## 💁 Contributing
If you want to setup a new page in the web application, just navigate to the ../demo/pages/ folder, create a new py file with the name of the page and develop on it.

If you want to contribute to the existing pages, please dev in your branch, test the application and ping me so that we can merge in main. Soon I'm going to put some effort into documenting the whole code and point out some next best contributions in each section of the code.
