###Tools: 1.01###
#Textual description of what we want the LLM to know about the tools it has available.
#-The system_message sets the context for the AI, explaining its role and available tools.
#define the system message with a placeholder for tool descriptions
#✅The LLM does not inherently "know" what a tool is – it understands them because the system message injects descriptions.
#✅If tool descriptions are ambiguous, the LLM might not call the right tool.
#📌 The system message provides essential context for the LLM. It does not execute tools itself but instead generates a structured request (like a function call) based on its understanding of the tools defined in {tools_description}. A well-defined tool description ensures the LLM correctly chooses and formats function calls.
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
#the provided Tool class is a class that represents a tool.
#Generic Tool implementation
#We create a generic Tool class that we can reuse whenever we need to use a tool.
#✅ outputs can be a string or list, since some tools return multiple values.
#✅to_string() method helps dynamically document tools, which is why it’s useful.
#📌 The Tool class provides a structured way to define reusable tools. It wraps a function with metadata (name, description, arguments, and outputs), allowing the LLM to call it dynamically. The to_string() method generates a readable description, which can be used in the system message to inform the LLM about available tools.
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
    



#But we can also use Python’s inspect module to retrieve all the information for us! This is what the @tool decorator does.
#Just to reiterate, with this decorator in place we can implement our tool like this:
#Note the @tool decorator before the function definition.
#If you are interested, you can disclose the following section to look at the decorator implementation.
import inspect# Inspect libary It allows you to examine live objects, such as modules, classes, functions, methods, and code objects, retrieving information about their source code, arguments, inheritance, and more.
def tool(func):
    """
    A decorator that creates a Tool instance from the given function.
    """
    # Get the function signature
    signature = inspect.signature(func)
    
    # Extract (param_name, param_annotation) pairs for inputs
    arguments = []
    for param in signature.parameters.values():
        annotation_name = (
            param.annotation.__name__ 
            if hasattr(param.annotation, '__name__') 
            else str(param.annotation)
        )
        arguments.append((param.name, annotation_name))
    
    # Determine the return annotation
    return_annotation = signature.return_annotation
    if return_annotation is inspect._empty:
        outputs = "No return annotation"
    else:
        outputs = (
            return_annotation.__name__ 
            if hasattr(return_annotation, '__name__') 
            else str(return_annotation)
        )
    
    # Use the function's docstring as the description (default if None)
    description = func.__doc__ or "No description provided."
    
    # The function name becomes the Tool name
    name = func.__name__
    
    # Return a new Tool instance
    return Tool(
        name=name, 
        description=description, 
        func=func, 
        arguments=arguments, 
        outputs=outputs
    )


#Just to reiterate, with this decorator in place we can implement our tool like this:
#we will be able to retrieve the following text  automatically from the source code via the to_string() function provided by the decorator:
@tool 
def calculator(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b  
print(calculator.to_string())#we can use the Tool’s to_string method to automatically retrieve a text suitable to be used as a tool description for an LLM: function provided by the decorator:
#This will print out the following text: 
#Tool Name: calculator, Description: Multiply two integers., Arguments: a: int, b: int, Outputs: int

#NB##he description is injected in the system prompt. Taking the example ###Tools: 1.01###



#We could create a Tool with this class using code like the following:
calculator_tool = Tool(
    "calculator",                   # name
    "Multiply two integers.",       # description
    calculator,                     # function to call
    [("a", "int"), ("b", "int")],   # inputs (names and types)
    "int",                          # output
)


#The @tool decorator is not a built-in Python feature. It’s a custom decorator that you (or a library you use) must implement.
#Exsample of a custom decorator... Disclaimer: This example implementation is fictional but closely resembles real implementations in most libraries.
