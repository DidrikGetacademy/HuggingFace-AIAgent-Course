

#To create an agent, we start by providing it with a set of functions/tools that define its capabilities.
#Let’s look at how to create an agent with some basic tools. As of this writing, the agent will automatically use the function calling API (if available), or a standard ReAct agent loop.
#LLMs that support a tools/functions API are relatively new, but they provide a powerful way to call tools by avoiding specific prompting and allowing the LLM to create tool calls based on provided schemas.
#ReAct agents are also good at complex reasoning tasks and can work with any LLM that has chat or text completion capabilities. They are more verbose, and show the reasoning behind certain actions that they take.

from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.core.tools import FunctionTool
from llama_index.core.workflow import Context
import asyncio

#define sample Tool -- type annotations, function names, and docstrings, are all included in parsed schemas!
def multiply(a: int, b: int) -> int:
    """Multiplies two integers and returns the resulting integer"""
    return a * b

# Agents are stateless by default, add remembering past interactions is opt-in using a Context object This might be useful if you want to use an agent that needs to remember previous interactions,
# like a chatbot that maintains context across multiple messages or a task manager that needs to track progress over time.

# stateless

async def main():
    llm = HuggingFaceInferenceAPI(model_name="Qwen/Qwen2.5-Coder-32B-Instruct")
    agent = AgentWorkflow.from_tools_or_functions(
        [FunctionTool.from_defaults(multiply)],
        llm=llm
    )
    # stateless
    response = await agent.run("what is 2 times 2?")
    print(f"Response: {response}")

    # remembering state
    from llama_index.core.workflow import Context

    ctx = Context(agent)
    

    response2 = await agent.run("My name is Bob.", ctx=ctx)
    print(f"response: {response2} ")
    response3 = await agent.run("what was my name again?", ctx=ctx)
    print(f"response: {response3} ")


asyncio.run(main())
#output: 
# Response: 2 times 2 is 4.
# response: Hello Bob, it seems you asked for a multiplication, and 2 multiplied by 3 is 6. How can I assist you further? 
# response: Your name is Bob. 
#HAD TO CHANGE PYTHON VERSION TO 3.11 because python 3.10 will give asynchio problems with llamaindex 