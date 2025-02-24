###Tools: 1.01###
#Textual description of what we want the LLM to know about the tools it has available.
#-The system_message sets the context for the AI, explaining its role and available tools.
#define the system message with a placeholder for tool descriptions
system_message="""You are an AI assistant designed to help users efficiently and accurately. your primary goal is to provide helpful, precise, and clear responses.

you have access to the following tools:
{tools_description}"""


# Provide a concrete description for the tools
#The placeholder {tools_description} is replaced with a concrete description.
tools_description = "Tool 1: Web Search, Tool 2: URL Opener, Tool 3: Data Analyzer"

#format the system message with the actual tools description
system_message = system_message.format(tools_description=tools_description)

#construct the conversation messages
messages = [
    {"role": "system", "content": system_message},
    {"role": "user", "content": "Can you tell me a fun fact about space?"}
]

import openai
response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=messages
)

# Print the assistant's reply
print(response.choices[0].message['content'])





###Tools: 1.02###
###Auto-Formatting tools section###


#the decorator --->
#The @tool decorator is not a built-in Python feature. It’s a custom decorator that you (or a library you use) must implement.
#Exsample of a custom decorator... Disclaimer: This example implementation is fictional but closely resembles real implementations in most libraries.
class Tool:
    """
    A class representing a reusable piece of code (Tool).
    
    Attributes:
        name (str): Name of the tool.
        description (str): A textual description of what the tool does.
        func (callable): The function this tool wraps.
        arguments (list): A list of argument.
        outputs (str or list): The return type(s) of the wrapped function.
    """
    def __init__(self, 
                 name: str,  #name (str): The name of the tool.
                 description: str, #description (str): A brief description of what the tool does.
                 func: callable, #function (callable): The function the tool executes.
                 arguments: list,#arguments (list): The expected input parameters.
                 outputs: str ##outputs (str or list): The expected outputs of the tool.
                 ):
        self.name = name 
        self.description = description 
        self.func = func
        self.arguments = arguments
        self.outputs = outputs

    #to_string(): Converts the tool’s attributes into a textual representation.
    def to_string(self) -> str: 
        """
        Return a string representation of the tool, 
        including its name, description, arguments, and outputs.
        """
        args_str = ", ".join([
            f"{arg_name}: {arg_type}" for arg_name, arg_type in self.arguments
        ])
        
        return (
            f"Tool Name: {self.name},"
            f" Description: {self.description},"
            f" Arguments: {args_str},"
            f" Outputs: {self.outputs}"
        )
    
    #__call__(): Calls the function when the tool instance is invoked.
    def __call__(self, *args, **kwargs):
        """
        Invoke the underlying function (callable) with provided arguments.
        """
        return self.func(*args, **kwargs)
    

#We could create a Tool with this class using code like the following:
calculator_tool = Tool(
    "calculator",                   # name
    "Multiply two integers.",       # description
    calculator,                     # function to call
    [("a", "int"), ("b", "int")],   # inputs (names and types)
    "int",                          # output
)

#But we can also use Python’s inspect module to retrieve all the information for us! This is what the @tool decorator does.
#Just to reiterate, with this decorator in place we can implement our tool like this:
#Note the @tool decorator before the function definition.
#we will be able to retrieve the following text  automatically from the source code via the to_string() function provided by the decorator:
@tool 
def calculator(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b  
print(calculator.to_string()) #This will print out the following text: 
#Tool Name: calculator, Description: Multiply two integers., Arguments: a: int, b: int, Outputs: int

