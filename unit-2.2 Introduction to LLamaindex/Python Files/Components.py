#🟢Loading and embedding documents🟢##
#---------------------------------#
#As mentioned before, LlamaIndex can work on top of your own data, however, before accessing data, we need to load it. There are three main ways to load data into LlamaIndex:
#1. SimpleDirectoryReader: A built-in loader for various file types from a local directory.
#2.LlamaParse: LlamaParse, LlamaIndex’s official tool for PDF parsing, available as a managed API.
#3.LlamaHub: A registry of hundreds of data-loading libraries to ingest data from any source.
#note to get familiar with loaders and llama parse for more complex data sources: https://github.com/run-llama/llama_cloud_services/blob/main/parse.md
import asyncio
from llama_index.core import Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core import SimpleDirectoryReader
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI   



def load_data_with_SimpleDirectoryReader():
    #🔑The simplest way to load data is with SimpleDirectoryReader. This versatile component can load various file types from a folder and convert them into Document objects that LlamaIndex can work with. Let’s see how we can use SimpleDirectoryReader to load data from a folder.
    reader = SimpleDirectoryReader(input_dir="path/to/directory") #reads all files (pdf,txt,docx etc)
    documents = reader.load_data() # a list of document objects 
    #after loading documents we need to break them into smaller pices called NODE objects. a NODE is just a chunk of text from the original documents that's easier for the AI to work with, while it still has references to the original document object.








async def create_IngestionPipeline():
    #The IngestionPipeline  helps create this nodes through two key transformations 
    #----------------------------------------------------------------------------------------#
    #1.SentenceSplitter breaks down documents into manageable chunks by splitting them at natural sentence boundaries.
    #2.HuggingFaceEmbedding converts each chunk into numerical embeddings - vector representations that capture the semantic meaning in a way AI can process efficiently.
    #🔑This process helps us organise our documents in a way that’s more useful for searching and analysis.#
    # create the pipeline with transformations
    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(chunk_overlap=0), #Split documents up in meaningful textbits (sentence/paragraph)
            HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5") #converts every bit to a vector format that the model will understand
        ]
    )

    nodes = await pipeline.arun(documents=[Document.example()])
    print(nodes)









def Storing_and_indexing_documents():
    #🟢Storing and indexing documents🟢#
    #🔑After creating our Node objects we need to index them to make them searchable, but before we can do that, we need a place to store our data.
    #🔑Since we are using an ingestion pipeline, we can directly attach a vector store to the pipeline to populate it. In this case, we will use Chroma to store our documents.
    #🟢Install ChromaDB🟢#
    #🔑pip install llama-index-vector-stores-chroma

    db = chromadb.PersistentClient(path="./alfred_chroma_db")
    chroma_collection = db.get_or_create_collection("alfred")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(chunk_size=25, chunk_overlap=0),
            HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5"),
        ],
        vector_store=vector_store
    )


    #🔑An overview of the different vector stores can be found in the https://docs.llamaindex.ai/en/stable/module_guides/storing/vector_stores/
    #🔑This is where vector embeddings come in - by embedding both the query and nodes in the same vector space, we can find relevant matches. The VectorStoreIndex handles this for us, using the same embedding model we used during ingestion to ensure consistency.
    #🟢Let’s see how to create this index from our vector store and embeddings:🟢#


    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    index = VectorStoreIndex.from_vector_store(vector_store,embed_model=embed_model)

     #let’s explore how to query it in different ways.
     #🟢Querying a VectorStoreIndex with prompts and LLMs🟢#
     #Before we can query our index, we need to convert it to a query interface. The most common conversion options are:
     #🔑as_retriever: For basic document retrieval, returning a list of NodeWithScore objects with similarity scores
     #🔑as_query_engine: For single question-answer interactions, returning a written response
     #🔑as_chat_engine: For conversational interactions that maintain memory across multiple messages, returning a written response using chat history and indexed context
     #We’ll focus on the query engine since it is more common for agent-like interactions. We also pass in an LLM to the query engine to use for the response.
 

    llm = HuggingFaceInferenceAPI(model_name="Qwen/Qwen2.5-Coder-32B-Instruct")
    query_engine = index.as_query_engine(
        llm=llm,
        response_mode="tree_summarize",
    )
    result = query_engine.query("what is the meaning of life?")
    print(result)
    #The meaning of life is 42



async def full_exsample_from_local_path():
    #STEP 1 (load documents)
    reader = SimpleDirectoryReader(input_dir=r"C:\Users\didri\Desktop\bedrfiter mappe") #reads all files (pdf,txt,docx etc)
    documents = reader.load_data() # a list of document objects 
    print(f"documents : {documents}")


    #STEP 2 (setup chroma vector store)
    db = chromadb.PersistentClient(path="./didriks_chroma_db")
    chroma_collection = db.get_or_create_collection("didrik")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)


    #STEP 3 (ingest with pipeline)
    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(chunk_size=25, chunk_overlap=0),
            HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5"),
        ],
        vector_store=vector_store
    )

    #STEP 4 (run pipeline)
    await pipeline.arun(documents=documents)
    print("ingestion complete.")

    
    #STEP 5 (build index from vector store)
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    index = VectorStoreIndex.from_vector_store(vector_store,embed_model=embed_model)

    #STEP 6 (setup query engine)
    llm = HuggingFaceInferenceAPI(model_name="Qwen/Qwen2.5-Coder-32B-Instruct")
    query_engine = index.as_query_engine(llm=llm,response_mode="tree_summariz")

    #STEP 7 (ask question)
    query = "hvilke selskaper i skien jobber med AI?"
    result = query_engine.query(query)
    print(f"\n 🔍 Query: {query}")
    print(f"📘 Answer: {result}")






if __name__ == "__main__":
   # asyncio.run(main()) splits the documents, creates vectors of them, nodes is the  prepeared documents bits ready to be searched in
   #Storing_and_indexing_documents() #All information is automatically persisted within the ChromaVectorStore object and the passed directory path.
    asyncio.run(full_exsample_from_local_path())
