###Functiontool - Convery any python function into a tool that an agent can use.####    

from llama_index.core.tools import FunctionTool
#A FunctionTool provides a simple way to wrap any Python function and make it available to an agent.
#You can pass either a synchronous or asynchronous function to the tool, along with optional name and description parameters.
#The name and description are particularly important as they help the agent understand when and how to use the tool effectively. 

def get_weather(location: str) -> str:
    """useful for getting the weather for a given location."""
    print(f"getting weather for {location}")
    return f"The weather in {location} is sunny"

tool = FunctionTool.from_defaults(
    get_weather,
    name="my_weather_tool",
    description="Useful for getting the weather for a given location."
)
tool.call("New York")
##output: getting weather for newyork 
