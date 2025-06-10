# Creating Multi-agent systems
    # The AgentWorkflow class also directly supports multi-agent systems. By giving each agent a name and description, the system maintains a single active speaker, with each agent having the ability to hand off to another agent.
    # By narrowing the scope of each agent, we can help increase their general accuracy when responding to user messages.
    # Agents in LlamaIndex can also directly be used as tools for other agents, for more complex and custom scenarios.
import asyncio
from llama_index.core.agent.workflow import (
    AgentWorkflow,
    FunctionAgent,
    ReActAgent,
)
# Tool wrapper to expose query engine to agents
from llama_index.core.tools import QueryEngineTool 

#LlamaIndex class for creating a vecto-based index
from llama_index.core import VectorStoreIndex 

#LLM inference interface for HuggingFace Inference API
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI   

#ChromaDB vector store integration for LlamaIndex
from llama_index.vector_stores.chroma import ChromaVectorStore 

# ChromaDB client for vector storage and retrieval
import chromadb  

# Embedding model interface for HuggingFace models
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

#Initialize the embedding model using BAAI/bge-small-en-v1.5 (for generating text embeddings)
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

#Define the path for the persistent ChromaDB storage (best to use a configurable path)
persist_path = "../../Vector_store/didriks_chroma_db"


#Create or connect to a persistent ChromaDB client at the given path
db = chromadb.PersistentClient(path=persist_path)


#Create or load a collection named "didrik" for storing persona data embeddings
chroma_collection = db.get_or_create_collection("didrik")

#Wrap the collection in a LlamaIndex ChromaVectorStore for use in the index
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)


# Initialize the HuggingFace Inference API with the Qwen2.5-Coder-32B-Instruct model for answer generation
llm = HuggingFaceInferenceAPI(model_name="Qwen/Qwen2.5-Coder-32B-Instruct")


#Build a vector index over the data in the Chroma collection using the embedding model
index = VectorStoreIndex.from_vector_store(vector_store,embed_model=embed_model)



# Create a query engine by combining the vector index with the LLM for response synthesis
query_engine = index.as_query_engine(llm=llm, similarity_top_k=3)


query_engine_tool = QueryEngineTool.from_defaults(
    query_engine=query_engine,
    name="PersonaQueryEngine",
    description="Tool for retrieving persona-related data from the 'didrik' ChromaDB vector store "
                "using BAAI/bge-small-en-v1.5 embeddings and generating responses with the Qwen2.5 LLM.",
    return_direct=False,
)


#Define some tools
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


def subtract(a: int, b: int) -> int:
    """Subtract two numbers"""
    return a - b 


# Create agent configs
# NOTE: we can use FunctionAgent or ReActAgent here.
# FunctionAgent works for LLMs with a function calling API.
# ReActAgent works for any LLM.

calculator_agent = ReActAgent(
    name="calculator",
    description="Performs basic arithmetic operations",
    system_prompt="You are a calculator assistant. use your tools for any math operations.",
    tools=[add, subtract],
    llm=llm,
)

query_agent = ReActAgent(
    name="info_lookup",
    description="Looks up information about XYZ",
    system_prompt="Use your tool to query a RAG system to answer information about XYZ",
    tools=[query_engine_tool],
    llm=llm
)


agent = AgentWorkflow(
    agents=[calculator_agent, query_agent], root_agent="calculator"
)


async def run_agent():

    response = await agent.run(user_msg="Can you add 5 and 3?")
    print(f"response: {response}")


if __name__ == "__main__":
    asyncio.run(run_agent())