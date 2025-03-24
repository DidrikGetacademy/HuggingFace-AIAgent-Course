from langchain.agents import load_tools
from smolagents import CodeAgent, HfApiModel, tool,Tool,load_tool

@tool
def catering_service_tool(query: str)-> str:
        """
        This tool returns the highest-rated catering service in Gotham City.
        
        Args:
            query: A search term for finding catering services.
        """

        services = {
            "Gotham Catering Co.": 4.9,
            "Wayne Manor Catering": 4.8,
            "Gotham City Events": 4.7,
        }

        best_service = max(services, key=services.get)

        return best_service





class superheroPartyThemeTool(Tool):
    name = "superhero_party_theme_generator"
    description = """
    This tool suggests creative superhero-themed party ideas based on a category.
    It returns a unique party theme idea"""

    inputs = {
        "category": {
            "type": "string",
            "description": "The type of superhero party (e.g., 'classic heroes', 'villain masquerade', 'futuristic Gotham')."
        }
    }
    
    output_type = "string"

    def forward(self,category: str):
        themes = {
            "classic heroes": "Justice League Gala: Guests come dressed as their favorite DC heroes with themed cocktails like 'The Kryptonite Punch'.",
            "villain masquerade": "Gotham Rogues' Ball: A mysterious masquerade where guests dress as classic Batman villains.",
            "futuristic Gotham": "Neo-Gotham Night: A cyberpunk-style party inspired by Batman Beyond, with neon decorations and futuristic gadgets."
        }

        return themes.get(category.lower(), "Themed party idea not found. Try 'classic heroes', 'villain masquerade', or 'futuristic Gotham'.")
    



image_generation_tool = load_tool(
     "m-ric/text-to-image",
     trust_remote_code=True
)

image_generation_tool_from_hub = Tool.from_space(
    "black-forest-labs/FLUX.1-schnell",
    name="image_generator",
    description="Generate an image from a prompt"
)







#FUNCTIONS
def testing_langchain_tool():
    model = HfApiModel("Qwen/Qwen2.5-Coder-32B-Instruct")

    search_tool = Tool.from_langchain(load_tools(["serpapi"])[0])

    agent = CodeAgent(tools=[search_tool], model=model)

    agent.run("Search for luxury entertainment ideas for a superhero-themed event, such as live performances and interactive experiences.")


def generate_image_from_hub_tool():
    model = HfApiModel("Qwen/Qwen2.5-Coder-32B-Instruct")
    agent = CodeAgent(tools=[image_generation_tool_from_hub], model=model)

    agent.run(
        "Improve this prompt, then generate an image of it.", 
        additional_args={'user_prompt': 'A grand superhero-themed party at Wayne Manor, with Alfred overseeing a luxurious gala'}
    )
     

def generate_superhero_party_theme_image():
    agent = CodeAgent(
    tools=[image_generation_tool],
    model=HfApiModel()
)
    agent.run("Generate an image of a luxurious superhero-themed party at Wayne Manor with made-up superheros.")


def Get_the_highest_rated_gathering():
    agent = CodeAgent(tools=[catering_service_tool], model=HfApiModel())

    result = agent.run(
    "Can you give me the name of the highest-rated catering service in Gotham City?"
    )
    print(result)


def superhero_themed_event():
    # name: The tool’s name.
    # description: A description used to populate the agent’s system prompt.
    # inputs: A dictionary with keys type and description, providing information to help the Python interpreter process inputs.
    # output_type: Specifies the expected output type.
    # forward: The method containing the inference logic to execute.¨

    party_theme_tool = superheroPartyThemeTool()
    agent = CodeAgent(tools=[party_theme_tool], model=HfApiModel())

    result = agent.run(
        "What would be a good superhero party idea for a 'villain masquerade' theme?"
)
    print(result)




if __name__ == "__main__":
    #Get_the_highest_rated_gathering()
    #superhero_themed_event()
    #generate_superhero_party_theme_image()
    #generate_image_from_hub_tool()
    #testing_langchain_tool()


