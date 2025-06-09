###Search Engine - a tool that lets agent use query engines.####


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

 
tool = QueryEngineTool.from_defaults(query_engine, name="",description="") # Wrap the query engine into a tool with a name and description for easy integration and use.



# This setup:
        # Embeds and stores text in a Chroma vector database.

        # Wraps this storage in an index for querying.

        # Uses a language model to understand and respond to queries.

        # Finally, it turns the query engine into a usable tool that can be plugged into larger agent systems or workflows.