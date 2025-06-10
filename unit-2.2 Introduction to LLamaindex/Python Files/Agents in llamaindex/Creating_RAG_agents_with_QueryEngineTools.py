# Agentic RAG is a powerful way to use agents to answer questions about your data. 
    # We can pass various tools to Alfred to help him answer questions. 
    # However, instead of answering the question on top of documents automatically, Alfred can decide to use any other tool or flow to answer the question.


# Embedding model interface for HuggingFace models
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


# Tool wrapper to expose query engine to agents
from llama_index.core.tools import QueryEngineTool 

# ChromaDB client for vector storage and retrieval
import chromadb  


#ChromaDB vector store integration for LlamaIndex
from llama_index.vector_stores.chroma import ChromaVectorStore 


#LlamaIndex class for creating a vecto-based index
from llama_index.core import VectorStoreIndex 

#LLM inference interface for HuggingFace Inference API
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI   

# AgentWorkflow is part of LlamaIndex's agent system. It allows you to create agents that can use tools or functions to perform complex, multi-step reasoning and respond to user queries.
from llama_index.core.agent.workflow import AgentWorkflow




# from transformers import AutoModelForCausalLM, AutoTokenizer
# from llama_index.llms.huggingface import HuggingFaceLLM
# llm = HuggingFaceLLM(
#     model=model,
#     tokenizer=tokenizer,
#     model_name=model_name,
#     device_map="auto",  # uses GPU if available
#     tokenizer_kwargs={"padding_side": "left"},
#     generation_kwargs={"max_new_tokens": 512}
# )



# from llama_index.llms.llama_cpp import LlamaCPP
# llm = LlamaCPP(
#     model_path="./models/llama-2-7b-chat.gguf",
#     temperature=0.7,
#     max_new_tokens=256,
#     context_window=2048,
#     model_kwargs={"n_gpu_layers": 40},  # config based on hardware
# )


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

#Build a vector index over the data in the Chroma collection using the embedding model
index = VectorStoreIndex.from_vector_store(vector_store,embed_model=embed_model)

# Initialize the HuggingFace Inference API with the Qwen2.5-Coder-32B-Instruct model for answer generation
llm = HuggingFaceInferenceAPI(
    model_name="Qwen/Qwen2.5-Coder-32B-Instruct"
    )



# Create a query engine by combining the vector index with the LLM for response synthesis
query_engine = index.as_query_engine(llm=llm, similarity_top_k=3)

#Wrap the query engine in a QueryEngineTool for use in an agent workflow
query_engine_tool = QueryEngineTool.from_defaults(
    query_engine=query_engine,
    name="PersonaQueryEngine",
    description="Tool for retrieving persona-related data from the 'didrik' ChromaDB vector store "
                "using BAAI/bge-small-en-v1.5 embeddings and generating responses with the Qwen2.5 LLM.",
    return_direct=False,
)



query_engine_agent = AgentWorkflow.from_tools_or_functions( #This method creates an agent-based workflow that uses provided tools/functions to complete tasks. It builds an agent that can dynamically decide when and how to use the tools during its reasoning process.
    [query_engine_tool],   # list of tools or functions the agent can use
    llm=llm, # the language model that drives the agent's reasoning and output
    system_prompt="You are a helpful assistant that has access to a database containing persona descriptions." # defines the agent's behavior
)

query_engine_agent.run("Hello what kind of data do you have access to ? ")