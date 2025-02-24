#We will:
#1.Define the Tool class – This wraps functions with metadata.
#2.Use the @tool decorator – To automatically register functions as tools.
#3.Format the system message – So the LLM understands which tools are available.
#4.Simulate an LLM generating a function call – The LLM does not execute tools directly; it suggests a structured function call.
#5.Execute the tool dynamically – We parse and run the suggested function.


#📌 Step 1: Define the Tool Class
import inspect