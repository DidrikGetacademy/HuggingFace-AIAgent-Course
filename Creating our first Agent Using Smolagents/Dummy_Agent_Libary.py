from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os 
from huggingface_hub import login


#Serverless API
#-----------------------------------------
#In the Hugging Face ecosystem, there is a convenient future called Serverless API that allows you to easily run inference on many models. There's no installation or deployment required. 
##You need a token from https://hf.co/settings/tokens, ensure that you select 'read' as the token type. If you run this on Google Colab, you can set it up in the "settings" tab under "secrets". Make sure to call it "HF_TOKEN"


#Getting Values from .env 
Huggingface_Token = os.getenv("Hugging_face_login")
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










def Dummy_Agent_text_generation():
    client = InferenceClient("https://jc26mwg228mkj8dw.us-east-1.aws.endpoints.huggingface.cloud")
    SYSTEM_PROMPT = """ 
    Answer the following questions as best you can. You have access to the following tools:

    get_weather: Get the current weather in a given location

    The way you use the tools is by specifying a json blob.
    Specifically, this json should have an `action` key (with the name of the tool to use) and an `action_input` key (with the input to the tool going here).

    The only values that should be in the "action" field are:
    get_weather: Get the current weather in a given location, args: {"location": {"type": "string"}}
    example use : 

    {{
    "action": "get_weather",
    "action_input": {"location": "New York"}
    }}

    ALWAYS use the following format:

    Question: the input question you must answer
    Thought: you should always think about one action to take. Only one action at a time in this format:
    Action:

    $JSON_BLOB (inside markdown cell)

    Observation: the result of the action. This Observation is unique, complete, and the source of truth.
    ... (this Thought/Action/Observation can repeat N times, you should take several steps when needed. The $JSON_BLOB must be formatted as markdown and only use a SINGLE action at a time.)

    You must always end your output with the following format:

    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    Now begin! Reminder to ALWAYS use the exact characters `Final Answer:` when you provide a definitive answer.
    """

    prompt=f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
        {SYSTEM_PROMPT}
        <|eot_id|><|start_header_id|>user<|end_header_id|>
        What's the weather in London ?
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """
    
    output = client.text_generation(
        prompt,
        max_new_tokens=100
    )
    print(output)








#we can also do it like this, which is what happends inside the chat method: 
#📌you need access to huggingfaces repo, ask for permission, you will most likely get permission within an hour 
def Dummy_Agent_Chat_method():
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
    client = InferenceClient("meta-llama/Llama-3.2-3B-Instruct")


    SYSTEM_PROMPT = """
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>
        Answer the following questions as best you can. You have access to the following tools:

    get_weather: Get the current weather in a given location

    The way you use the tools is by specifying a json blob.
    Specifically, this json should have a `action` key (with the name of the tool to use) and a `action_input` key (with the input to the tool going here).

    The only values that should be in the "action" field are:
    get_weather: Get the current weather in a given location, args: {"location": {"type": "string"}}
    example use : 

    {{
    "action": "get_weather",
    "action_input": {"location": "New York"}
    }}

    ALWAYS use the following format:

    Question: the input question you must answer
    Thought: you should always think about one action to take. Only one action at a time in this format:
    Action:

    $JSON_BLOB (inside markdown cell)

    Observation: the result of the action. This Observation is unique, complete, and the source of truth.
    ... (this Thought/Action/Observation can repeat N times, you should take several steps when needed. The $JSON_BLOB must be formatted as markdown and only use a SINGLE action at a time.)

    You must always end your output with the following format:

    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    Now begin! Reminder to ALWAYS use the exact characters `Final Answer:` when you provide a definitive answer. 
        """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "What's the weather in London ?"},
    ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    output = client.text_generation(
        prompt,
        max_new_tokens=200,
       #stop=["Observation:"] # Let's stop before any actual function is called
    )
    print("output",output)


#📌Do you see the issue in output??
"The answear was hallucinated by the model, we need to stop to actually execute the function! Let's now stop on"
"[Observation” so that we don’t hallucinate the actual function response.]"










#get_weather TOOL (Dummy Function)
def get_weather(location):
    return f"the weather in {location} is sunny with low temperatures. \n"




def Dummy_Agent_with_dummy_tool_getweather():
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
    client = InferenceClient("meta-llama/Llama-3.2-3B-Instruct")
    SYSTEM_PROMPT = """
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>
        Answer the following questions as best you can. You have access to the following tools:

    get_weather: Get the current weather in a given location

    The way you use the tools is by specifying a json blob.
    Specifically, this json should have a `action` key (with the name of the tool to use) and a `action_input` key (with the input to the tool going here).

    The only values that should be in the "action" field are:
    get_weather: Get the current weather in a given location, args: {"location": {"type": "string"}}
    example use : 

    {{
    "action": "get_weather",
    "action_input": {"location": "New York"}
    }}

    ALWAYS use the following format:

    Question: the input question you must answer
    Thought: you should always think about one action to take. Only one action at a time in this format:
    Action:

    $JSON_BLOB (inside markdown cell)

    Observation: the result of the action. This Observation is unique, complete, and the source of truth.
    ... (this Thought/Action/Observation can repeat N times, you should take several steps when needed. The $JSON_BLOB must be formatted as markdown and only use a SINGLE action at a time.)

    You must always end your output with the following format:

    Thought: I now know the final answer
    Final Answer: the final answer to the original input question
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "What's the weather in London ?"},
    ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    output = client.text_generation(
        prompt,
        max_new_tokens=200,
        stop=["Observation:"] # Let's stop before any actual function is called
    )
    print("initial output",output)

    weather_output = get_weather('London')
    print("Get_weather output: ",weather_output)

    new_prompt = prompt + output + weather_output

    final_output = client.text_generation(
        new_prompt,
        max_new_tokens=200
    )
    print(final_output)







if __name__ == "__main__":
    #login(Huggingface_Token)
    #HuggingFace_Serverless_API()
    #HuggingFace_Serverless_API_Public_endpoint()
    #Huggingface_EOS_token()
    #huggingface_with_chat_Template()
    #Dummy_Agent_text_generation()
    #Dummy_Agent_Chat_method()
    #print(get_weather('London'))
    Dummy_Agent_with_dummy_tool_getweather()

#📌We learned how we can create Agents from scratch using Python code, and we saw just how tedious that process can be. Fortunately, many Agent libraries simplify this work by handling much of the heavy lifting for you.