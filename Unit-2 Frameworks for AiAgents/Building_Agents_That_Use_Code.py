#For the model, we’ll rely on HfApiModel, which provides access to Hugging Face’s Serverless Inference API. The default model is "Qwen/Qwen2.5-Coder-32B-Instruct", which is performant and available for fast inference, but you can select any compatible model from the Hub.
#Learn more about how to build code agents in the smolagents documentation "https://huggingface.co/docs/smolagents"
from huggingface_hub import login
from smolagents import CodeAgent, DuckDuckGoSearchTool, tool, HfApiModel
import numpy as np
import time
import datetime

login()

# TOOLS
@tool
def suggest_menu(occasion: str) -> str:
    """
        Suggests a menu based on the occasion.
        Args:
            occasion: The type of occasion for the party.
    """
    if occasion == "casual":
        return "Pizza, snacks and drinks."
    elif occasion == "formal":
        return "3-course dinner with wine and dessert."
    elif occasion == "superhero":
        return "Buffet with high-energy and healthy food."
    else:
        return "Custom menu for the butler."


alfred_agent = CodeAgent(
    tools=[DuckDuckGoSearchTool(), suggest_menu],
    model=HfApiModel(),
    additional_authorized_imports=['datetime']
)


alfred_agent.run("Search for the best music recommendation for a party at Wayne's mansion.")

alfred_agent.run("Prepare a formal menu for the party.")

alfred_agent.run(
    """
    Alfred needs to prepare for the party. Here are the tasks:
    1. Prepare the drinks - 30 minutes
    2. Decorate the mansion - 60 minutes
    3. Set up the menu - 45 minutes
    4. Prepare the music and playlist - 45 minutes

    If we start right now, at what time will the party be ready?
    """
)

alfred_agent.push_to_hub('didrikSkjelbred/Custom_Party_AlfredAgent')
