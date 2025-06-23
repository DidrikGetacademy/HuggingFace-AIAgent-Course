import os
from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
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


# Initialize our LLM
model = ChatOpenAI(Temperature=0)

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

    response_text = response.content.lower()
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
        {"role": "assistant", "content": response.content}
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
        {"role": "assistant", "content": response.content}
    ]

    #Return state updates
    return {
        "email_draft": response.content,
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






#------------------------#
#Step 3: Define Our Routing Logic
#------------------------#
#We need a function to determine which path to take after classification:
 





#💡 Note: This routing function is called by LangGraph to determine which edge to follow after the classification node. The return value must match one of the keys in our conditional edges mapping.
