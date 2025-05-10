#We will:
#1.Define the Tool class – This wraps functions with metadata.
#2.Use the @tool decorator – To automatically register functions as tools.
#3.Format the system message – So the LLM understands which tools are available.
#4.Simulate an LLM generating a function call – The LLM does not execute tools directly; it suggests a structured function call.
#5.Execute the tool dynamically – We parse and run the suggested function.


#📌 Step 1: Define the Tool Class
#[This class allows us to define tools dynamically]
import inspect
class Tool:
    """ 
    A class representin a reusable tool.

    Attributes:
       name (str): The Tool's name.
       Description (str): A brief description of what the tool does.
       func (callable): The function executed when the tool is used.
        arguments (list): List of argument names and types.
        outputs (str): Expected return type.
    """

    def __init__(self, name: str, description: str,  func: callable, arguments: list, outputs: str):
        self.name = name
        self.description = description
        self.func = func
        self.arguments = arguments
        self.outputs = outputs


    def to_string(self) -> str:
        """
        Returns a string representation of the tool used in system messages.
        """
        args_str = ", ".join([f"{arg_name}: {arg_type}" for arg_name, arg_type in self.arguments])
        return f"Tool name: {self.name}, Description: {self.description}, Arguments: {args_str}, Outputs: {self.outputs}"
    

    def __call__(self, *args, **kwargs):
        """
        Calls the tool's function when the instance is used. 
        """
        return self.func(*args, **kwargs)
    





#📌 Step 2: Create the @tool Decorator
#[This decorator automatically extracts metadata from functions and registers them as tools.]
def tool(func):
    """
    A decorator that creates a Tool instance from a function. 
    """
    signature = inspect.signature(func)
    
    arguments = [(param.name, param.annotation.__name__ if param.annotation != inspect._empty else "Any")
                 for param in signature.parameters.Values()]
    

    return_annotation = signature.return_annotation
    outputs = return_annotation.__name__ if return_annotation != inspect.empty else "None"

    return Tool(name=func.__name__, description=func.__doc__ or "No Description provided.",
                func=func, arguments=arguments,outputs=outputs)






#📌 Step 3: Define Tools Using the Decorator
#[These functions will be used by the LLM]
#[Now,these functions are automatically converted into tool instances.]
@tool
def calculator(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b

@tool
def greet(name: str) -> str:
    """Returns a greeting message."""
    return f"Hello, {name}!"





#📌 Step 4: Format the System Message for the LLM
#[We must inject the tool descriptions into the system message so the LLM knows what tools it can use.]

#List all available tools
tools = [calculator, greet]
tools_description = "\n".join(tool.to_string() for tool in tools)

#Define the system message with tool details
system_message = f"""
You are an AI assistant that can use the following tools:

{tools_description}

When the user asks something that requires a tool, respond with the function call in JSON format
"""






#📌 Step 5: Simulate an LLM Generating a Function Call
#Now, let's simulate what an LLM might generate when it recives a request.

#Exsample User Query:
"Can you multiply 6 and 7?"
#The LLM doesn't execute the function-it suggest a structured tool call like this:
import json 

#Simulated LLM output (it doesn't run code, just suggests this)
llm_response = json.dumps({
    "tool": "calculator",
    "arguments": {"a": 6, "b": 7}
})
print("LLM Response:", llm_response)







#📌 Step 6: Parse and Execute the Tool
#We now interpret the LLM’s response and execute the function.
llm_response_data = json.loads(llm_response)

# Find the matching tool
for tool in tools:
    if tool.name == llm_response_data["tool"]:
        result = tool(**llm_response_data["arguments"])
        print(f"Tool Execution Result: {result}")





#🎯 Final Output
#LLM Response: {"tool": "calculator", "arguments": {"a": 6, "b": 7}}
#Tool Execution Result: 42





####SUMMARY#####
#📌 Key Takeaways
#✅ The Tool class encapsulates functions with metadata for easy LLM integration.
#✅ The @tool decorator automates tool creation from functions.
#✅ The system message informs the LLM about available tools.
#✅ The LLM does NOT execute tools—it suggests function calls in JSON format.
#✅ Our script parses and executes the tool dynamically.