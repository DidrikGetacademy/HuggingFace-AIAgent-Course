from PIL import Image
import requests
from io import BytesIO
from smolagents import CodeAgent, OpenAIServerModel,TransformersModel,AutoModelForImageTextToText
import torch
from dotenv import load_dotenv
import os
os.environ["TRANSFORMERS_NO_FLASH_ATTENTION"] = "1"
import os
print("FlashAttention disabled:", os.getenv("TRANSFORMERS_NO_FLASH_ATTENTION"))

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
def Alfred_catching_joker_image_reading():
    image_urls = [
        "https://upload.wikimedia.org/wikipedia/commons/e/e8/The_Joker_at_Wax_Museum_Plus.jpg",
        "https://upload.wikimedia.org/wikipedia/en/9/98/Joker_%28DC_Comics_character%29.jpg"
    ]

    images = []

    for url in image_urls:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36" 
        }
        response = requests.get(url, headers=headers)
        image = Image.open(BytesIO(response.content)).convert("RGB")
        images.append(image)

    model = AutoModelForImageTextToText(
        model_id=r"c:\Users\didri\Desktop\LLM-models\microsoft\microsoft\Phi-4-multimodal-instruct",
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True
    )



    agent = CodeAgent(
        tools=[],
        model=model,
        max_steps=20,
        verbosity_level=4
    )

    response = agent.run(
        """
        Describe the costume and makeup that the comic character in these photos is wearing.
        Identify if the character is The Joker or Wonder Woman.
        """,
        images=images,
    )



def images_dynamic_Retrieval():
    return


if __name__ == "__main__":
    Alfred_catching_joker_image_reading()