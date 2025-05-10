from transformers import AutoTokenizer

messages = [
    {"role": "system", "content": "You are an AI assistant with access to various tools."},
    {"role": "user", "content": "Hi !"},
    {"role": "assistant", "content": "Hi human, what can help you with ?"},
]



#To convert the previous conversation into a prompt, we load the tokenizer and call apply_chat_template:
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-1.7B-Instruct")
rendered_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

#The rendered_prompt returned by this function is now ready to use as the input for the model you chose!
#Note ---> This apply_chat_template() function will be used in the backend of your API, when you interact with messages in the ChatML format