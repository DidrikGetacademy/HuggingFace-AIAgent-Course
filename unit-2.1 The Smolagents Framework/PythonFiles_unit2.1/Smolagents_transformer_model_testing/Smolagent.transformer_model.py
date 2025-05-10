from huggingface_hub import login, InferenceClient

login()

model_id = "meta-llama/Llama-3.3-70B-Instruct"

client = InferenceClient(model=model_id)

messages = [
    {"role": "system", "content": "Du er en hjelpsom assistent."},
    {"role": "user", "content": "Forklar hvordan stop-sekvenser fungerer."}
]

def custom_model(messages, stop_sequences=["Task"]):
    response = client.chat_completion(messages, stop=stop_sequences, max_tokens=1000)
    answer = response.choices[0].message
    return answer



if __name__ == "__main__":
    result = custom_model(messages)
    print(result)



#Key points:

#huggingface_hub gives access to any Hugging Face model (e.g., meta-llama/Llama-3.3-70B-Instruct).
#Requires API token and internet connection — it queries Hugging Face's cloud inference API.
#More customizable — you can control max_tokens, stop sequences, and more.
#Supports bigger, more powerful models (e.g., LLaMA, GPT-based models).
#More flexibility in handling responses (e.g., parsing .choices[0].message).#
#✅ Good for:
#Large, cutting-edge models
#Cloud-based, scalable inference
#Production-ready apps
