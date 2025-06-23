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
   # We will use the BM25Retriever from the langchain_community.retrievers module to create a retriever tool.



from smolagents import Tool
from langchain_community.retrievers import BM25Retriever

class GuestInfoRetriever(Tool):
    name = "Guest_Info_Retreiver"
    description = "Retrieves detailed information about gala guests based on their name or relation."
    input = {
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