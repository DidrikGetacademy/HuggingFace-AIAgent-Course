#Building and Integrating Tools for Your Agent


#---------------------------------#
# Give Your Agent Access to the Web
#---------------------------------#
from smolagents import DuckDuckGoSearchTool
from typing import List


#Initalize the DuckDuckgo search tool
search_tool = DuckDuckGoSearchTool()

#Exsample usage:
results = search_tool("Who's the current president of France?")
print(results)
# Expected output: The current President of France in Emmanuel Macron.


#-----------------------------------------------------------------------------------------#
# Creating a Custom Tool that can be used to get the latest news about a specific topic.
#-----------------------------------------------------------------------------------------#
from newsapi import NewsApiClient
class GetLatestNewsTool(Tool):
    name="Latest_news"
    description="""Fetch the latest breaking headline news worldwide. supports filtering by keyword, country, category, or specific sources.
                Supports filtering by keyword, country, category, or specific sources.
                **Note:** you cannot use 'sources' together with 'country' or 'category';
                choose either sources OR country/category filters.
                """
    inputs = {
        "Query": {
            "type": "string",
            "description": "keywords or phrase to search for in headlines."
        },
        "Country": {
            "type": "string",
            "descripton": "2-letter country code (e.g., 'us', 'gb'). optional",
            "required": False
        },
        "category": {
            "type": "string",
            "description": "News category (e.g, 'buisness', 'sports'). optional",
            "required": False
        },
        "sources": {
            "type": "string",
            "description": "Comma-seperated list of news source ID'S to get headliens from. Optional",
            "required": False
        }
    }
    output = {
        "articles": {
            "type": "list",
            "description": "List of matching news articles. Each article contains: "
                            "`source` (ID and name), `author`, `title`, `description`,"
                            "`url`, `urlToImage`, `PublishedAt`, and `content`."
        }
    }
    
    def __init__(self, api_key):
        self.newsapi = NewsApiClient(api_key=api_key)

    def run(self, Query=None, Country=None, Category=None, sources= None):
        """
        Run the tool: Call NewsApi with the provided filters.
        """
        if sources and (Country or Category):   
            return "You cannot use `sources` together with 'country' or 'category'"
        

        response = self.newsapi.get_top_headlines(
            q=Query,
            country=Country,
            category=Category,
            sources=sources
        )

        return response








#------------------------------------------------------------------------------#
# Creating a Custom Tool for Weather Information to Schedule the Fireworks
#------------------------------------------------------------------------------#
from smolagents import Tool
import random 

class weatherinfoTool(Tool):
    name = "weather_info"
    description ="Fetches dummy weather information for a given location."
    inputs = {
        "location": {
            "type": "string",
            "description": "The location to get weather information for."
        }
    }
    output_type = "string"

    def forward(self, location: str):
        # Dummy weather data
        weather_conditions = [
            {"condition": "Rainy", "temp_c": 15},
            {"condition": "Clear", "temp_c": 25},
            {"condition": "Windy", "temp_c": 20}
        ]
        #Randomaly select a weather condition
        data = random.choice(weather_conditions)
        return f"Weather in {location}: {data['condition']}, {data['temp_c']}°C"

#Initalize the tool
weather_info_tool = weatherinfoTool()


#--------------------------------------------------------#
# Creating a Hub Stats Tool for Influential AI Builders
#--------------------------------------------------------#
from smolagents import Tool
from huggingface_hub import list_models

class HubStatsTool(Tool):
    name = "hub_stats"
    description = "Fetches the most downloaded model from a specific author on the Hugging Face Hub."
    inputs = {
        "author": {
            "type": "string",
            "description": "The username of the model author/organization to find models from."
        }
    }
    output_type = "string"

    def forward(self, author: str):
        try:
            models = List(list_models(author=author, sort="downloads", direction=-1, limit=1))
            if models:
                model = models[0]
                return f"The most downloaded model by {author} is {model.id} with {model.downloads:,} downloads"
            else:
                return f"No models found for author: {author}."
        except Exception as e:
            print(f"Error fetching model for {author}: {str(e)}")

#Initalize the tool
hub_stats_tool = HubStatsTool()

#Exsample usage
print(hub_stats_tool("facebook"))
#Expected output: The most downloaded model by facebook is facebook/esmfold_v1 with 12,544,550 downloads.

#--------------------------------------------------------#
# Integrating Tools with Alfred
#--------------------------------------------------------#

from smolagents import CodeAgent, InferenceClientModel

model = InferenceClientModel()

alfred = CodeAgent(
    tools=[search_tool, weather_info_tool, hub_stats_tool,GetLatestNewsTool()],
    model=model
)

#Exsample query Alfred might recieve during the gala 
response = alfred.run("what is facebook and what's their most popular model?")
print("🎩 Alfred's Response:")
print(response)
#Expected output: 🎩 Alfred's Response: Facebook is a social networking website where users can connect, share information, and interact with others. The most downloaded model by Facebook on the Hugging Face Hub is ESMFold_v1.


#--------------------------------------------------------#
# Conclusion
#--------------------------------------------------------#
#By integrating these tools, Alfred is now equipped to handle a variety of tasks, from web searches to weather updates and model statistics. This ensures he remains the most informed and engaging host at the gala.


