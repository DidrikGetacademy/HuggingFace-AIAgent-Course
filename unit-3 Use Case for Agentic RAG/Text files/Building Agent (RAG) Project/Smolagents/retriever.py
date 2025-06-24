import datasets
from langchain.docstore.document import Document

#-------------------------------------#
# Step 1: Load and Prepare the Dataset
#--------------------------------------#




# Load the dataset
guest_dataset = datasets.load_dataset("agents-course/unit3-invitees", split="train")

# Convert dataset entries into documents objects
docs = [
    Document(
        page_content="\n".join([
            f"Name: {guest['name']}",
            f"Relation: {guest['relation']}",
            f"Description: {guest['description']}",
            f"Email: {guest['email']}"
        ]),
        metadata={"name": {guest['email']}}
    )
    for guest in guest_dataset
]

#In the code above, we:
    # ● Load the dataset
    # ● Convert each guest entry into a Document object with formatted content
    # ● Store the Document objects in a list



#-------------------------------------#
# Step 2: Create the Retriever Tool
#--------------------------------------#
#We will use the BM25Retriever from the langchain_community.retrievers module to create a retriever tool.
from smolagents import Tool
from langchain_community.retrievers import BM25Retriever

class GuestInfoRetriever(Tool):
    name = "Guest_Info_Retreiver"
    description = "Retrieves detailed information about gala guests based on their name or relation."
    inputs = {
        "query": {
            "type": "string",
            "description": "The name or relation of the guest you want information about."
        }
    }
    output_type = "string"

    def __init__(self,docs):
        self.is_initialized = False
        self.retriever = BM25Retriever.from_documents(docs)

    def forward(self, query: str):
        results = self.retriever.get_relevant_documents(query)
        if results:
            return "\n\n".join([doc.page_content for doc in results[:3]])
        else:
            return "No mathing guest information found"
        
guest_info_tool = GuestInfoRetriever(docs)

#Let’s understand this tool step-by-step:

#The (name) and (description) help the agent understand when and how to use this tool
#The (inputs) define what parameters the tool expects (in this case, a search query)
#We’re using a (BM25Retriever), which is a powerful text retrieval algorithm that doesn’t require (embeddings)
#The (forward) method processes the query and returns the most relevant guest information


#---------------------------------------#
# Step 3: Integrate the Tool with Alfred
#---------------------------------------#
#Finally, let’s bring everything together by creating our agent and equipping it with our custom tool:

from smolagents import CodeAgent, InferenceClientModel


#Initalize the hugging face model
model = InferenceClientModel()

#Create Alfred, our gala agent, with the guest info tool
alfred = CodeAgent(tools=[guest_info_tool], model=model)

#Exsample query Alfred might receive during the gala
response = alfred.run("Tell me about our guest named 'Lady Ada Lovelace")

print("🎩 Alfred's Response:")
print(response)

####OUTPUT#######
│                                                                                                                                                                                                                                                                                       ││ Tell me about our guest named 'Lady Ada Lovelace                                                                                                                                                                                                                                      ││                                                                                                                                                                                                                                                                                       │╰─ InferenceClientModel - Qwen/Qwen2.5-Coder-32B-Instruct ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ Step 1 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ─ Executing parsed code: ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── 
#   lady_ada_info = Guest_Info_Retreiver(query="Lady Ada Lovelace")                                                                                                                                                                                                                        
#   print(lady_ada_info)                                                                                                                                                                                                                                                                   
#  ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── 
# c:\Users\didri\Desktop\AI-Agents\AI-Agent\Huggingface Agent Course\unit-3 Use Case for Agentic RAG\Text files\Building Agent (RAG) Project\Smolagents\retriever.py:58: LangChainDeprecationWarning: The method `BaseRetriever.get_relevant_documents` was deprecated in langchain-core 0.1.46 and will be removed in 1.0. Use :meth:`~invoke` instead.
#   results = self.retriever.get_relevant_documents(query)
# Execution logs:
# Name: Ada Lovelace
# Relation: best friend
# Description: Lady Ada Lovelace is my best friend. She is an esteemed mathematician and friend. She is renowned for her pioneering work in mathematics and computing, often celebrated as the first computer programmer due to her work on Charles Babbage's Analytical Engine.
# Email: ada.lovelace@example.com

# Name: Marie Curie
# Relation: no relation
# Description: Marie Curie was a groundbreaking physicist and chemist, famous for her research on radioactivity.
# Email: marie.curie@example.com

# Name: Dr. Nikola Tesla
# Relation: old friend from university days
# Description: Dr. Nikola Tesla is an old friend from your university days. He's recently patented a new wireless energy transmission system and would be delighted to discuss it with you. Just remember he's passionate about pigeons, so that might make for good small talk.
# Email: nikola.tesla@gmail.com

# Out: None
# [Step 1: Duration 3.59 seconds| Input tokens: 2,058 | Output tokens: 68]
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ Step 2 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ ─ Executing parsed code: ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── 
#   # Splitting the observation into lines                                                                                                                                                                                                                                                 
#   lines = lady_ada_info.split('\n')                                                                                                                                                                                                                                                      
                                                                                                                                                                                                                                                                                         
#   # Extracting the relevant information                                                                                                                                                                                                                                                  
#   name = lines[0].split(': ')[1]                                                                                                                                                                                                                                                         
#   relation = lines[1].split(': ')[1]                                                                                                                                                                                                                                                     
#   description = lines[2].split(': ')[1]                                                                                                                                                                                                                                                  
                                                                                                                                                                                                                                                                                         
#   # Formatting the final answer                                                                                                                                                                                                                                                          
#   final_answer_string = f"Name: {name}\nRelation: {relation}\nDescription: {description}"                                                                                                                                                                                                
#   final_answer(final_answer_string)                                                                                                                                                                                                                                                      
#  ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── 
# Out - Final answer: Name: Ada Lovelace
# Relation: best friend
# Description: Lady Ada Lovelace is my best friend. She is an esteemed mathematician and friend. She is renowned for her pioneering work in mathematics and computing, often celebrated as the first computer programmer due to her work on Charles Babbage's Analytical Engine.
# [Step 2: Duration 4.17 seconds| Input tokens: 4,463 | Output tokens: 198]
# 🎩 Alfred's Response:
# Name: Ada Lovelace
# Relation: best friend
# Description: Lady Ada Lovelace is my best friend. She is an esteemed mathematician and friend. She is renowned for her pioneering work in mathematics and computing, often celebrated as the first computer programmer due to her work on Charles Babbage's Analytical Engine.
# PS C:\Users\didri\Desktop\AI-Agents\AI-Agent\Huggingface Agent Course> 


