
# 2. LoadAndSearchToolSpec
# The LoadAndSearchToolSpec takes in any existing Tool as input. As a tool spec, 
# it implements to_tool_list, and when that function is called, two tools are returned: a loading tool and then a search tool.
# The load Tool execution would call the underlying Tool, and then index the output (by default with a vector index). The search Tool execution would take in a query string as input and call the underlying index.

from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.tools.tool_spec.load_and_search import (
    LoadAndSearchToolSpec,
)
from llama_index.tools.wikipedia import WikipediaToolSpec
from llama_index.core.tools import QueryEngineTool
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
wiki_spec = WikipediaToolSpec()
# Get the search wikipedia tool
tool = wiki_spec.to_tool_list()[1]


llm = HuggingFaceInferenceAPI(model_name="Qwen/Qwen2.5-Coder-32B-Instruct")

# Create the Agent with load/search tools
agent = FunctionAgent(
    llm=llm, tools=LoadAndSearchToolSpec.from_defaults(tool).to_tool_list()
)

from llama_index.core import VectorStoreIndex
from llama_index.core.tools import QueryEngineTool
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb


embed_model = HuggingFaceEmbedding("BAAI/bge-small-en-v1.5") # the embedding model from Hugging Face for text embedding.

db = chromadb.PersistentClient(path="../../Vector_store/didriks_chroma_db") # Connect to a persistent Chroma database located at the given path.

chroma_collection = db.get_or_create_collection("didrik") # Access or create a Chroma collection named "alfred" for storing embeddings.

vectorstore = ChromaVectorStore(chroma_collection=chroma_collection)  #Create a vector store interface using the Chroma collection.

index = VectorStoreIndex.from_vector_store(vectorstore, embed_model=embed_model) # Build a vector index from the vector store and embed model to support similarity search.

llm = HuggingFaceInferenceAPI(model_name="Qwen/Qwen2.5-Coder-32B-Instruct") # Load a large language model from Hugging Face to serve as the query engine's brain.

query_engine = index.as_query_engine(llm=llm) # Convert the index into a query engine using the language model to process natural language queries.

# For example, say you specify a tool:
tool = QueryEngineTool.from_defaults(
    query_engine,
    name="<name>",
    description="<description>",
    return_direct=True,
)

agent = FunctionAgent(llm=llm, tools=[tool])

async def response():
    response = await agent.run("<question that invokes tool>")



# Debugging Tools#
# Often, it can be useful to debug what exactly the tool definition is that is being sent to APIs.

# You can get a good peek at this by using the underlying function to get the current tool schema, which is levereged in APIs like OpenAI and Anthropic.
schema = tool.metadata.get_parameters_dict()
print(schema)