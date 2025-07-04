# Document Analysis Graph
# ------------------------------------------------------------------------------------------------------
# let’s create a document analysis system using LangGraph to serve Mr. Wayne’s needs. This system can:

# 1.Process images document
# 2.Extract text using vision models (Vision Language Model)
# 3.Perform calculations when needed (to demonstrate normal tools)
# 4.Analyze content and provide concise summaries
# 5.Execute specific instructions related to documents


#The Butler’s Workflow

#-----------------------------------------------------------------------------------#
# Setting Up the environment: pip install langgraph langchain_openai langchain_core
# ---> IMPORTS
#------------------------------------------------------------------------------------#
import base64
from typing import List, TypedDict, Annotated, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage
from langgraph.graph.message import add_messages
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from IPython.display import Image, display



#---------------------------------------------#
#         Defining Agent’s State
#---------------------------------------------#
# This state is a little more complex than the previous ones we have seen. 
# AnyMessage is a class from Langchain that defines messages, and add_messages is an operator that adds the latest message rather than overwriting it with the latest state.
# 🔑This is a new concept in LangGraph, where you can add operators in your state to define the way they should interact together.
class AgentState(TypedDict):
    # The document provided
    input_file: Optional[str] # Contains file path (PDF/PNG)
    messages: Annotated[List[AnyMessage], add_messages]

#---------------------------------------------#
#           Preparing Tools
#---------------------------------------------#
vision_llm = ChatOpenAI(model="gpt-4o")

def extract_text(img_path: str) -> str:
    """
    Extract text from an image file using a multimodal model.
    
    Master Wayne often leaves notes with his training regimen or meal plans.
    This allows me to properly analyze the contents.
    """
    all_text = ""
    try:

        #Read image and encode as base64
        with open(img_path, "rb") as image_file:
            image_bytes = image_file.read()
        
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")

        #Prepare the prompt including the base64 image data

        message = [
            HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": (
                            "Extract all the text from this image. "
                            "Return only the extracted text, no explanations"
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64, {image_base64}"
                        },
                    },
                ]
            )
        ]

        #Call the vision-capable model
        response = vision_llm.invoke(message)

        #Append extracted text
        all_text += response.content if hasattr(response, "content") else response + "\n\n"

        return all_text.strip()
    except Exception as e:
        #A butler should handle errors gracefully
        error_msg = f"Error extracting text: {str(e)}"
        print(error_msg)
        return ""

def divide(a: int, b: int) -> float:
    """Divide a and b - for master wayne's occasional calculations."""
    return a / b 

#Equip the butler with tools
tools = [
    divide,
    extract_text
]

llm = ChatOpenAI(model="gpt-4o")
llm_with_tools = llm.bind_tools(tools, parallel_tool_calls=False)



#---------------------------------------------#
#            The nodes
#---------------------------------------------#

def assistant(state: AgentState):
    #System message
    textual_description_of_tool="""
    extract_text(img_path:str) -> str:
        Extract text from an image file using a multimodal model.

        Args:
            img_path: A local image file path (strings).

        Returns:
            A single string containing the concatenated text extracted from each image.
    divide(a: int, b: int) -> float:
        Divide a and b 
    """

    image=state["input_file"]
    sys_msg = SystemMessage(content=f"You are a helpful butler named Alfred that serves Mr. Wayne and Batman. You can analyse documents and run computations with provided tools:\n{textual_description_of_tool} \n You have access to some optional images. Currently the loaded image is: {image}")

    return {
        "messages": [llm_with_tools.invoke([sys_msg] + state["messages"])],
        "input_file": state["input_file"]
    }


#---------------------------------------------#
#    The ReAct Pattern: How I Assist Mr. Wayne
#---------------------------------------------#
# Allow me to explain the approach in this agent. The agent follows what’s known as the ReAct pattern (Reason-Act-Observe)
# 1.Reason about his documents and requests
# 2.Act by using appropriate tools
# 3.Observe the results
# 4.Repeat as necessary until I’ve fully addressed his needs

# This is a simple implementation of an agent using LangGraph:

#The graph
builder = StateGraph(AgentState)

#Define nodes: these do the work
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

#Define edges: these determine how the control flow moves
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    #If the latest message requires a tool, route to tool
    # Otherwise, provide a direct response
    tools_condition,
)
builder.add_edge("tools", "assistant")
react_graph = builder.compile()

#show the butler's thought process
display(Image(react_graph.get_graph(xray=True).draw_mermaid_png()))
# We define a tools node with our list of tools. The assistant node is just our model with bound tools. We create a graph with assistant and tools nodes.
# We add a tools_condition edge, which routes to End or to tools based on whether the assistant calls a tool.
# Now, we add one new step:
# We connect the tools node back to the assistant, forming a loop.
# ● After the assistant node executes, tools_condition checks if the model’s output is a tool call.
# ● If it is a tool call, the flow is directed to the tools node.
# ● The tools node connects back to assistant.
# ● This loop continues as long as the model decides to call tools.
# ● If the model response is not a tool call, the flow is directed to END, terminating the process.





#---------------------------------------------#
#        The Butler in Action
#---------------------------------------------#
#Example 1: Simple Calculations --> Here is an example to show a simple use case of an agent using a tool in LangGraph.
messages = [HumanMessage(content="Divide 6790 by 5")]
messages = react_graph.invoke({"messages": messages, "input_file": None})

#show the messages
for m in messages['messages']:
    m.pretty_print()

#  The conversation would proceed with OUTPUT:
#----------------------------------------------------------#
# Human: Divide 6790 by 5

# AI Tool Call: divide(a=6790, b=5)

# Tool Response: 1358.0

# Alfred: The result of dividing 6790 by 5 is 1358.0.
#----------------------------------------------------------#


#Example 2: Analyzing Master Wayne’s Training Documents
#When Master Wayne leaves his training and meal notes:

messages = [HumanMessage(content="According to the note provided by Mr. Wayne in the provided images. What's the list of items I should buy for the dinner menu?")]
messages = react_graph.invoke({"messages": messages, "input_file": "Batman_training_and_meals.png"})

#  The interaction would proceed with OUTPUT:
#----------------------------------------------------------#
# Human: According to the note provided by Mr. Wayne in the provided images. What's the list of items I should buy for the dinner menu?

# AI Tool Call: extract_text(img_path="Batman_training_and_meals.png")

# Tool Response: [Extracted text with training schedule and menu details]

# Alfred: For the dinner menu, you should buy the following items:

# 1. Grass-fed local sirloin steak
# 2. Organic spinach
# 3. Piquillo peppers
# 4. Potatoes (for oven-baked golden herb potato)
# 5. Fish oil (2 grams)

# Ensure the steak is grass-fed and the spinach and peppers are organic for the best quality meal.



#---------------------------------------------#
#           Key Takeaways
#---------------------------------------------#
# Should you wish to create your own document analysis butler, here are key considerations:

# 1. Define clear tools for specific document-related tasks
# 2. Create a robust state tracker to maintain context between tool calls
# 3. Consider error handling for tool failures
# 4. Maintain contextual awareness of previous interactions (ensured by the operator add_messages)
#🔑 With these principles, you too can provide exemplary document analysis service worthy of Wayne Manor.

