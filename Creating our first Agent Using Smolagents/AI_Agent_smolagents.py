#Let’s Create Our First Agent Using smolagents#
#📌learned how we can create Agents from scratch using Python code, and we saw just how tedious that process can be. Fortunately, many Agent libraries simplify this work by handling much of the heavy lifting for you.
#📌In this tutorial, you’ll create your very first Agent capable of performing actions such as image generation, web search, time zone checking and much more!
#📌You will also publish your agent on a Hugging Face Space so you can share it with friends and colleagues.

#✅What is smolagents?
#📌To make this Agent, we’re going to use smolagents, a library that provides a framework for developing your agents with ease.
#📌This lightweight library is designed for simplicity, but it abstracts away much of the complexity of building an Agent, allowing you to focus on designing your agent’s behavior.
#📌We’re going to get deeper into smolagents in the next Unit. Meanwhile, you can also check this blog post or the library’s repo in GitHub.
#📌In short, smolagents is a library that focuses on codeAgent, a kind of agent that performs “Actions” through code blocks, and then “Observes” results by executing the code.


#📌Here is an example of what we’ll build!
#📌We provided our agent with an Image generation tool and asked it to generate an image of a cat.
#📌The agent inside smolagents is going to have the same behaviors as the custom one we built previously: it’s going to think, act and observe in cycle until it reaches a final answer:
#📌Libary to smolagent: https://github.com/huggingface/smolagents
#📌Video too watch: --> https://youtu.be/PQDKcWiuln4

#📌Throughout this lesson, the only file you will need to modify is the (currently incomplete) “app.py”.
# You can see here the original one in the template[https://huggingface.co/spaces/agents-course/First_agent_template/blob/main/app.py]. To find yours, go to your copy of the space,
#Then click the Files tab and then on app.py in the directory listing.



#✅Let’s break down the code together:
#•The file begins with some simple but necessary library imports
from smolagents import CodeAgent, DuckDuckGoSearchTool, HfApiModel, load_tool, tool #As outlined earlier, we will directly use the Code Agent class from smolagents.
import datetime
import requests
import pytz
import yaml
from First_agent_template_DidrikSkjelbred  import FinalAnswerTool



#✅#The Tools
#📌now let's get into the tools! if you want a refresher about tools, don't hesitate to go back to the Tools section of the course. 
@tool
def my_custom_tool(arg1:str, arg2:int)-> str: # it's important to specify the return type
    # Keep this format for the tool description / args description but feel free to modify the tool
    """A tool that does nothing yet 
    Args:
        arg1: the first argument
        arg2: the second argument
    """
    return "What magic will you build ?"

@tool
def get_current_time_in_timezone(timezone: str) -> str:
    """A tool that fetches the current local time in a specified timezone.
    Args:
        timezone: A string representing a valid timezone (e.g., 'America/New_York').
    """
    try:
        # Create timezone object
        tz = pytz.timezone(timezone)
        # Get current time in that timezone
        local_time = datetime.datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
        return f"The current local time in {timezone} is: {local_time}"
    except Exception as e:
        return f"Error fetching time for timezone '{timezone}': {str(e)}"
    


#The Tools are what we are encouraging you to build in this section! We give you two examples:
#1.A non-working dummy Tool that you can modify to make something useful.
#2.An actually working Tool that gets the current time somewhere in the world.

#📌 To define your tool it is important to:
#1. Provide input and output types for your function, like in get_current_time_in_timezone(timezone: str) -> str:
#2. A well formatted docstring. smolagents is expecting all the arguments to have a textual description in the docstring.



#✅The Agent 
#📌 It uses Qwen/Qwen2.5-Coder-32B-Instruct as the LLM engine. This is a very capable model that we’ll access via the serverless API.
final_answer = FinalAnswerTool()
model = HfApiModel(
    max_tokens=2096,
    temperature=0.5,
    model_id='Qwen/Qwen2.5-Coder-32B-Instruct',
    custom_role_conversions=None,
)

with open("prompts.yaml", 'r') as stream:
    prompt_templates = yaml.safe_load(stream)
    
# We're creating our CodeAgent
agent = CodeAgent(
    model=model,
    tools=[final_answer], # add your tools here (don't remove final_answer)
    max_steps=6,
    verbosity_level=1,
    grammar=None,
    planning_interval=None,
    name=None,
    description=None,
    prompt_templates=prompt_templates
)

GradioUI(agent).launch()



#📌This Agent still uses the InferenceClient we saw in an earlier section behind the HfApiModel class!

#📌We will give more in-depth examples when we present the framework in Unit 2. For now, you need to focus on adding new tools to the list of tools using the tools parameter of your Agent.

#📌For example, you could use the DuckDuckGoSearchTool that was imported in the first line of the code, or you can examine the image_generation_tool that is loaded from the Hub later in the code.

#📌Adding tools will give your agent new capabilities, try to be creative here!

#📌The complete “app.py”:

from smolagents import CodeAgent, DuckDuckGoSearchTool, HfApiModel, load_tool, tool
import datetime
import requests
import pytz
import yaml
from First_agent_template_DidrikSkjelbred import FinalAnswerTool
from First_agent_template_DidrikSkjelbred import GradioUI
# Below is an example of a tool that does nothing. Amaze us with your creativity!
@tool
def my_custom_tool(arg1:str, arg2:int)-> str: # it's important to specify the return type
    # Keep this format for the tool description / args description but feel free to modify the tool
    """A tool that does nothing yet 
    Args:
        arg1: the first argument
        arg2: the second argument
    """
    return "What magic will you build ?"

@tool
def get_current_time_in_timezone(timezone: str) -> str:
    """A tool that fetches the current local time in a specified timezone.
    Args:
        timezone: A string representing a valid timezone (e.g., 'America/New_York').
    """
    try:
        # Create timezone object
        tz = pytz.timezone(timezone)
        # Get current time in that timezone
        local_time = datetime.datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
        return f"The current local time in {timezone} is: {local_time}"
    except Exception as e:
        return f"Error fetching time for timezone '{timezone}': {str(e)}"


final_answer = FinalAnswerTool()
model = HfApiModel(
max_tokens=2096,
temperature=0.5,
model_id='Qwen/Qwen2.5-Coder-32B-Instruct',
custom_role_conversions=None,
)


# Import tool from Hub
image_generation_tool = load_tool("agents-course/text-to-image", trust_remote_code=True)

with open("prompts.yaml", 'r') as stream:
    prompt_templates = yaml.safe_load(stream)
    
agent = CodeAgent(
    model=model,
    tools=[final_answer], # add your tools here (don't remove final_answer)
    max_steps=6,
    verbosity_level=1,
    grammar=None,
    planning_interval=None,
    name=None,
    description=None,
    prompt_templates=prompt_templates
)


GradioUI(agent).launch()


#Congratulations, you’ve built your first Agent! Don’t hesitate to share it with your friends and colleagues.

#Since this is your first try, it’s perfectly normal if it’s a little buggy or slow. In future units, we’ll learn how to build even better Agents.

#The best way to learn is to try, so don’t hesitate to update it, add more tools, try with another model, etc.

#In the next section, you’re going to fill the final Quiz and get your certificate!