from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os 
#Serverless API
#-----------------------------------------
#In the Hugging Face ecosystem, there is a convenient future called Serverless API that allows you to easily run inference on many models. There's no installation or deployment required. 
##You need a token from https://hf.co/settings/tokens, ensure that you select 'read' as the token type. If you run this on Google Colab, you can set it up in the "settings" tab under "secrets". Make sure to call it "HF_TOKEN"


#Getting the HF_TOKEN from .env 
load_dotenv()
hf_token = os.getenv("HF_TOKEN")
os.environ["HF_TOKEN"]=hf_token





def HuggingFace_Serverless_API():
    client = InferenceClient("meta-llama/Llama-3.2-3B-Instruct")
    output = client.text_generation(
        "The capital of France is",
        max_new_tokens=100,             
    )
    print(output)
    #✅[Expected output according too huggingface course]: "Paris. The capital of France is Paris. The capital of France is Paris. The capital of France is Paris. The capital of France is Paris. The capital of France is Paris. The capital of France is Paris. The capital of France is Paris. The capital of France is Paris. The capital of France is Paris. The capital of France is Paris. The capital of France is Paris. The capital of France is Paris. The capital of France is Paris. The capital of France is Paris."
    #✅[output that we got from model using this client: 
    #The capital of Italy is Rome. The capital of Spain is Madrid. The capital of Germany is Berlin. The capital of the United Kingdom is London. The capital of Australia is Canberra. The capital of China is Beijing. The capital of Japan is Tokyo. The capital of India is New Delhi. The capital of Brazil is Brasília. The capital of Russia is Moscow. The capital of South Africa is Pretoria. The capital of Egypt is Cairo. The capital of Turkey is Ankara. The
    #📌The output was wrong, so the free model is overloaded. let's try the public endpoint (client = InferenceClient("https://jc26mwg228mkj8dw.us-east-1.aws.endpoints.huggingface.cloud")  ---> FUNCTION [HuggingFace_Serverless_API_Public_endpoint])






def HuggingFace_Serverless_API_Public_endpoint():
    client = InferenceClient("https://jc26mwg228mkj8dw.us-east-1.aws.endpoints.huggingface.cloud") #Public endpoint

    output = client.text_generation(
        "The capital of France is",
        max_new_tokens=100,             
    )
    print(output)
    #✅[output that we got from model using this client ("https://jc26mwg228mkj8dw.us-east-1.aws.endpoints.huggingface.cloud")]: 
    #The capital of Italy is Rome. The capital of Spain is Madrid. The capital of Germany is Berlin. The capital of the United Kingdom is London. The capital of Australia is Canberra. The capital of China is Beijing. The capital of Japan is Tokyo. The capital of India is New Delhi. The capital of Brazil is Brasília. The capital of Russia is Moscow. The capital of South Africa is Pretoria. The capital of Egypt is Cairo. The capital of Turkey is Ankara. The
    #📌 The output was wrong here also according too what expected from the course output, that means huggingface free models are overloaded.. 📌but we did get an output from the api, so that's what is most important for now. 
  






#✅As seen in the LLM section, if we just do decoding, the model will only stop when it predicts an EOS token, and this does not happen here because this is a conversational (chat) model and we didn’t apply the chat template it expects.
#✅If we now add the special tokens related to the Llama-3.2-3B-Instruct model that we’re using, the behavior changes and it now produces the expected EOS.
def Huggingface_EOS_token():
    client = InferenceClient("https://jc26mwg228mkj8dw.us-east-1.aws.endpoints.huggingface.cloud") 
    prompt="""<|begin_of_text|><|start_header_id|>user<|end_header_id|>
    The capital of France is<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
    output = client.text_generation(
        prompt,
        max_new_tokens=100,
    )

    print(output)  #output we got: ...Paris!
    #Output expected from the course: [The capital of France is paris.]




def huggingface_with_chat_Template():
    client = InferenceClient("https://jc26mwg228mkj8dw.us-east-1.aws.endpoints.huggingface.cloud")
    output = client.chat.completions.create(
        messages=[
            {"role": "user", "content": "The capital of France is"},
        ],
            stream=False,
            max_tokens=1024
    )
    print(output.choices[0].message.content) #output: Paris. 
    #we got the same expected output mentioned in the course. 
    #📌The chat method is the RECOMMENDED method to use in order to ensure a smooth transition between models, but since this notebook is only educational, we will keep using the “text_generation” method to understand the details.




if __name__ == "__main__":
    #HuggingFace_Serverless_API()
    #HuggingFace_Serverless_API_Public_endpoint()
    #Huggingface_EOS_token()
    #huggingface_with_chat_Template()

    

