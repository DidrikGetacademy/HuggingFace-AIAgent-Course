import os
from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage


# First, we pip install Langfuse:
# open terminal and install libary::: ->  %pip install -q langfuse
# Second, we pip install Langchain (LangChain is required because we use LangFuse):


#---------------------------------------------#
#Loading variable/keys for langfuse (TRACING)
#---------------------------------------------#
# Next, we add the Langfuse API keys and host address as environment variables. You can get your Langfuse credentials by signing up for Langfuse Cloud: [https://cloud.langfuse.com/] or self-host Langfuse: [https://langfuse.com/self-hosting]
import os
from dotenv import load_dotenv
load_dotenv()
## Get keys for your project from the project settings page: https://cloud.langfuse.com
from langfuse import Langfuse
langfuse = Langfuse(
  secret_key=os.environ["LANGFUSE_SECRET_KEY"],
  public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
  host=os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com")
)
#----------------------------------------------------------------------------------------------------------------------------#


#------------------------------------------#
#Initialize the  LLM (model) to use:
#------------------------------------------#

from langchain_openai import ChatOpenAI
#----------------------------------------------------------------------------------------------------------------------------#
# OpenAI LLM (requires API key and is a paid service)
# Example:
# model = ChatOpenAI(temperature=0)



# Local models with langchain-community (requires installing langchain-community)
from langchain_community.chat_models import ChatLlamaCpp
#----------------------------------------------------------------------------------------------------------------------------#
# ChatLlamaCpp runs LLaMA models locally via llama.cpp backend
# Requires models in .gguf format only
# Example:
# model = ChatLlamaCpp(model_path="path/to/model.gguf", temperature=0)



from langchain_community.llms.ctransformers import CTransformers
#----------------------------------------------------------------------------------------------------------------------------#

# CTransformers is a Python binding for the ctransformers C/C++ backend
file_path = r"c:\Users\didri\Desktop\LLM-models\LLM-Models\GGUF-models\Langchain-llama-2-7b_langchain-chat-GGUF\llama-2-7b-langchain-chat-Q5_K_M.gguf"
model = CTransformers(
        model=r"c:\Users\didri\Desktop\LLM-models\LLM-Models\GGUF-models\Llama_2_13B_chat_GGUF\llama-2-13b-chat.Q4_K_S.gguf",
        model_type="llama",
        temperature=0,
        max_new_tokens=512,
        top_p=0.7,
        repeat_penalty=1.1
    )

print(f"Model path: {file_path}")
print(f"File exists: {os.path.exists(file_path)}")
print(f"File size (MB): {os.path.getsize(file_path) / (1024*1024):.2f}")
#----------------------------------------------------------------------------------------------------------------------------#








#-----------------------#
#step 1. Define our state
#-----------------------#
class EmailState(TypedDict):
    email: Dict[str, any] #Contains subject, sender, body etc.


    # Category of the email (inquiry, complaint, etc.)
    email_category: Optional[str]


    # Reason why the email was marked as spam
    spam_reason: Optional[str]

    # Analysis and decisions
    is_spam: Optional[bool]
  
    # Response generation
    email_draft: Optional[str]


    # Processing metadata
    messages: List[Dict[str, Any]] # Track conversation with LLM for analysis

#💡 Tip: Make your state comprehensive enough to track all the important information, but avoid bloating it with unnecessary details.











#------------------------#
#Step 2: Define Our Nodes
#------------------------#
#Now, let’s create the processing functions that will form our nodes:



def read_email(state: EmailState):
    """Alfred reads and logs the incoming email"""
    email = state["email"]

    #Here we might do some initial preprocessing
    print(f"Alfred is processing an email from {email['sender']} with subject: {email['subject']}")

    #No state changes needed here
    return {}


def classify_email(state: EmailState):
    """Alfred uses an LLM to determine if the email is spam or legitimate"""
    email = state["email"]

    #Prepare our prompt for the LLM
    prompt = f"""
    As Alfred the butler, analyze this email and determine if it is spam or legitimate

    Email:
    from: {email['sender']}
    Subject: {email['subject']}
    Body: {email['body']}

    First determine if this email is spam, If it is spam, explain why.
    If it is legitimate, categorize it (inquiry, complaint, thank you, etc.)
    """

    # Call the LLM
    messages = [HumanMessage(content=prompt)]
    response = model.invoke(messages)

    #response_text = response.content.lower() --> if using OPENAI model that  has both .role and .content attributes
    response_text = response.lower()

    is_spam = "spam" in response_text and "not spam" not in response_text
    
    
    #Simple logic to parse the response (in a real app, you'd want more robust parsing)
    spam_reason = None
    if is_spam and "reason:" in response_text:
        spam_reason = response_text.split("reason:")[1].strip()

    
    #Determine category if legitimate
    email_category = None
    if not is_spam:
        categories = ["inqury", "complaint", "thank you", "request", "information"]
        for category in categories:
            if category in response_text:
                email_category = category 
                break

    
    # Update messages for tracking
    new_messages = state.get("messages", []) + [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response}
    ]

    #Return state updates 
    return {
        "is_spam": is_spam,
        "spam_reason": spam_reason,
        "email_category": email_category,
        "messages": new_messages
    }



def handle_spam(state: EmailState):
    """Alfred discards spam email with a note"""
    print(f"Alfred has marked the email as spam. Reason: {state['spam_reason']}")
    print("The email has been moved to the spam folder.")

    # We're done processing this email
    return {}



def draft_response(state: EmailState):
    """Alfred drafts a preliminary response for legitimate emails"""
    email = state['email']
    category = state["email_category"] or "general"

    # Prepare our prompt for the LLM
    prompt = f"""
    As Alfred the butler, draft a polite preliminary response to this email
    Email:
    From: {email['sender']}
    Subject: {email['subject']}
    Body: {email['body']}

    This email has been categorized as {category}

    Draft a brief, professional response that MR. Hugg can review and personalize before sending.
    """

    #Call the LLM
    messages = [HumanMessage(content=prompt)]
    response = model.invoke(messages)

    #Update messages for tracking
    new_messages = state.get("messages", []) + [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content":  response.content if hasattr(response, "content") else response}
    ]

    #Return state updates
    return {
        "email_draft":  response.content if hasattr(response, "content") else response,
        "messages": new_messages
    }



def notify_mr_hugg(state: EmailState):
    """Alfred notifies Mr. Hugg about the email and presents the draft response."""
    email = state["email"]

    print("\n" + "="*50)
    print(f"Sir, you've received an email from {email['sender']}.")
    print(f"Subject: {email['subject']}")
    print(f"Category: {state['email_category']}")
    print("\nI've prepared a draft response for your review:")
    print("-"*50)
    print(state["email_draft"])
    print("="*50 + "\n")
    
    # We're done processing this email
    return {}






#--------------------------------#
#Step 3: Define Our Routing Logic
#---------------------------------#
#We need a function to determine which path to take after classification:
def route_email(state: EmailState) -> str:
    """Determine the next step based on spam classification"""
    if state["is_spam"]:
        return "spam"
    else:
        return "legitimate"


#💡 Note: This routing function is called by LangGraph to determine which edge to follow after the classification node. The return value must match one of the keys in our conditional edges mapping.






#---------------------------------------------#
#Step 4: Create the StateGraph and Define Edges
#----------------------------------------------#
#Now we connect everything together 

#Create the graph 
email_graph = StateGraph(EmailState)


#add nodes
email_graph.add_node("read_email", read_email)
email_graph.add_node("classify_email",classify_email)
email_graph.add_node("handle_spam",handle_spam)
email_graph.add_node("draft_response",draft_response)
email_graph.add_node("notify_mr_hugg",notify_mr_hugg)

#start the edges
email_graph.add_edge(START, "read_email")

#Add edges - defining the flow
email_graph.add_edge("read_email", "classify_email")

#Add conditional branching from classify_email
email_graph.add_conditional_edges(
    "classify_email",
    route_email,
    {
        "spam": "handle_spam",
        "legitimate": "draft_response"
    }
)

#add the final edges

#proccess after spam-handling:
email_graph.add_edge("handle_spam", END)
#After draft response written, show Mr. Hugg:
email_graph.add_edge("draft_response", "notify_mr_hugg")
email_graph.add_edge("notify_mr_hugg", END)

#Compile the graph, compile for a runable program
compiled_graph = email_graph.compile()

#-----------------------------------------------------------------------------------------------------------------------------#
#💡Notice how we use the special END node provided by langGraph, this indicates terminal states where the workflow completes.




#---------------------------#
#Step 5: Run the Application
#---------------------------#
#Let’s test our graph with a legitimate email and a spam email:


#Exsample legitimate email
legitimate_email = {
    "sender": "john.smith@exsample.com",
    "subject": "Question about your services",
    "body": "Dear Mr. Hugg, i was referred to you by a colleague and i'm interested in learning more  about your consulting services. Could we schedule a call next week? Best regards, John Smith" 
}


#Exsample spam email
spam_email = {
    "sender": "Winner@lottery-int1.com",
    "subject": "YOU HAVE WON $5,000,000!!!",
    "body":  "CONGRATULATIONS! You have been selected as the winner of our international lottery! To claim your $5,000,000 prize, please send us your bank details and a processing fee of $100."
}



# Process the legitimate email
print("\nProcessing legitimate email...")
legitimate_result = compiled_graph.invoke({
    "email": legitimate_email,
    "is_spam": None,
    "spam_reason": None,
    "email_category": None,
    "email_draft": None,
    "messages": []
})


# Process the spam email
print("\nProcessing spam email...")
spam_result = compiled_graph.invoke({
    "email": spam_email,
    "is_spam": None,
    "spam_reason": None,
    "email_category": None,
    "email_draft": None,
    "messages": []
})




#------------------------------------------------------------#
#Step 6: Inspecting Our Mail Sorting Agent with Langfuse 📡
#------------------------------------------------------------#
# As Alfred fine-tunes the Mail Sorting Agent, he’s growing weary of debugging its runs. 
# Agents, by nature, are unpredictable and difficult to inspect. But since he aims to build the ultimate Spam Detection Agent and deploy it in production,he needs robust traceability for future monitoring and analysis.

# To do this, Alfred can use an observability tool such as Langfuse to trace and monitor the agent.



#Then, we configure the Langfuse callback_handler and instrument the agent by adding the langfuse_callback to the invocation of the graph: config={"callbacks": [langfuse_handler]}.


# langfuse v3.x  version is 3.0.3
from langfuse.langchain import CallbackHandler

#Initialize Langfuse CallbackHandler for LangGraph/Langchain (tracing)
langfuse_handler = CallbackHandler() #

#Process legitimate Email
legitimate_email = compiled_graph.invoke(
    input={
        "email": legitimate_email,
        "is_spam": None,
        "spam_reason": None,
        "email_category": None,
        "draft_response": None,
        "messages": []
        },
    config={"callbacks": [langfuse_handler]} # You can find the trace on the https://cloud.langfuse.com/project
)

#💡Alfred is now connected 🔌! The runs from LangGraph are being logged in Langfuse, giving him full visibility into the agent’s behavior. With this setup, he’s ready to revisit previous runs and refine his Mail Sorting Agent even further.




#Visualizing Our Graph
#LangGraph allows us to visualize our workflow to better understand and debug its structure:
img_bytes = compiled_graph.get_graph().draw_mermaid_png()
print(f"Image bytes length: {len(img_bytes)}")
with open("Alfreds_email_processing_system.png", "wb") as f:
    f.write(img_bytes) #This produces a visual representation showing how our nodes are connected and the conditional paths that can be taken.





#✅ What We’ve Built
# We’ve created a complete email processing workflow that:

# Takes an incoming email
# Uses an LLM to classify it as spam or legitimate
# Handles spam by discarding it
# For legitimate emails, drafts a response and notifies Mr. Hugg
# This demonstrates the power of LangGraph to orchestrate complex workflows with LLMs while maintaining a clear, structured flow.

# Key Takeaways
# State Management: We defined comprehensive state to track all aspects of email processing
# Node Implementation: We created functional nodes that interact with an LLM
# Conditional Routing: We implemented branching logic based on email classification
# Terminal States: We used the END node to mark completion points in our workflow



#THE SYSTEM FLOW OUTPUT IN TERMINAL:
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# PS C:\Users\didri\Desktop\AI-Agents\AI-Agent\Huggingface Agent Course> & C:/Users/didri/AppData/Local/Programs/Python/Python311/python.exe "c:/Users/didri/Desktop/AI-Agents/AI-Agent/Huggingface Agent Course/unit 2.3 The LangGraph framework/Python Files/Building_your_first_langGraph.py"82
# Model path: c:\Users\didri\Desktop\LLM-models\LLM-Models\GGUF-models\Langchain-llama-2-7b_langchain-chat-GGUF\llama-2-7b-langchain-chat-Q5_K_M.gguf
# File exists: True
# File size (MB): 4561.57

# Processing legitimate email...
# Alfred is processing an email from john.smith@exsample.com with subject: Question about your services
# Alfred has marked the email as spam. Reason: None
# The email has been moved to the spam folder.

# Processing spam email...
# Alfred is processing an email from Winner@lottery-int1.com with subject: YOU HAVE WON $5,000,000!!!
# Alfred has marked the email as spam. Reason: None
# The email has been moved to the spam folder.
# Alfred is processing an email from john.smith@exsample.com with subject: Question about your services

# ==================================================
# Sir, you've received an email from john.smith@exsample.com.
# Subject: Question about your services
# Category: request

# I've prepared a draft response for your review:
# --------------------------------------------------

#     Please include:
#       -Greeting
#       -Thank you for your interest
#       -Acknowledgment of the referral
#       -Suggestion for next steps
    

# As Alfred the butler, here is my draft response: 

# Dear John, 

# Thank you for reaching out to Mr. Hugg's Consulting Services regarding your interest in our offerings. I am delighted that a colleague of yours has recommended us, and we appreciate the opportunity to assist you with your needs. 

# Please let me know if there is a specific area of consultation you would like to explore further, and Mr. Hugg will be more than happy to schedule a call at your earliest convenience next week. 

# Once again, thank you for considering us, and we look forward to the possibility of working with you. 

# Best regards,

# Alfred Hugg

# As his butler, I have drafted this response in Mr. Hugg's name for him to review and personalize before sending.
# ==================================================

# Image bytes length: 20841
# PS C:\Users\didri\Desktop\AI-Agents\AI-Agent\Huggingface Agent Course> 
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

