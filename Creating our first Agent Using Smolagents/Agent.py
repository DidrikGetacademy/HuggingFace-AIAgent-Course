#Serverless API
#-----------------------------------------
#In the Hugging Face ecosystem, there is a convenient future called Serverless API that allows you to easily run inference on many models. There's no installation or deployment required. 



import os 
from huggingface_hub import InferenceClient

#Getting the HF_TOKEN from .env 
from dotenv import load_dotenv
load_dotenv()
hf_token = os.getenv("HF_TOKEN")

##You need a token from https://hf.co/settings/tokens, ensure that you select 'read' as the token type. If you run this on Google Colab, you can set it up in the "settings" tab under "secrets". Make sure to call it "HF_TOKEN"

os.environ["HF_TOKEN"]=hf_token