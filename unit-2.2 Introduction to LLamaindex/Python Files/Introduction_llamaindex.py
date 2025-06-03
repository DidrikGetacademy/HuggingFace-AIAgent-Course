#####Introduction to the LlamaHub####




###Installation Guide####


#LlamaIndex installation instructions are available as a well-structured overview on LlamaHub. 
#This might be a bit overwhelming at first, but most of the installation commands generally follow an easy-to-remember format:
#---------------------------------------------------------------------------------------------------------------------------------#
#🔑pip install llama-index-{component-type}-{framework-name}







#Let’s try to install the dependencies for an LLM and embedding component using the Hugging Face inference API integration.
#---------------------------------------------------------------------------------------------------------------------------------#
#🔑pip install llama-index-llms-huggingface-api llama-index-embeddings-huggingface#





#USAGE#
#---------------------------------------------------------------------------------------------------------------------------------#
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
import os
from dotenv import load_dotenv


#load env file
load_dotenv()


#Retrieve the hf token
hf_token = os.getenv("HF_TOKEN")


llm = HuggingFaceInferenceAPI(
    model_name="Qwen/Qwen2.5-Coder-32B-Instruct",
    temperature=0.7,
    max_tokens=100,
    token=hf_token
)

response= llm.complete("Hello, how are you?")
print(response)
#Hello! I'm just a computer program, so I don't have feelings, but thanks for asking. How can I assist you today?