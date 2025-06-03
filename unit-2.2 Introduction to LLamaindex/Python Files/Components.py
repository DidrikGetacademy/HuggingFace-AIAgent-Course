#🟢Loading and embedding documents🟢##
#---------------------------------#
#As mentioned before, LlamaIndex can work on top of your own data, however, before accessing data, we need to load it. There are three main ways to load data into LlamaIndex:
#1. SimpleDirectoryReader: A built-in loader for various file types from a local directory.
#2.LlamaParse: LlamaParse, LlamaIndex’s official tool for PDF parsing, available as a managed API.
#3.LlamaHub: A registry of hundreds of data-loading libraries to ingest data from any source.
#note to get familiar with loaders and llama parse for more complex data sources: https://github.com/run-llama/llama_cloud_services/blob/main/parse.md
import asyncio
from llama_index.core import Document
import llama_index.core
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core import SimpleDirectoryReader
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI   
from llama_index.core.evaluation import FaithfulnessEvaluator,AnswerRelevancyEvaluator,CorrectnessEvaluator
import llama_index
import os
from dotenv import load_dotenv

load_dotenv()

PHOENIX_API_KEY = os.getenv('PHOENIX_API_KEY')
os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"api_key={PHOENIX_API_KEY}"
llama_index.core.set_global_handler(
    "arize_phoenix",
    endpoint="https://llamatrace.com/v1/traces"
)








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
            SentenceSplitter(chunk_size=256, chunk_overlap=20),
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
    #🟢Under the hood, the query engine doesn’t only use the LLM to answer the question but also uses a ResponseSynthesizer as a strategy to process the response. Once again, this is fully customisable but there are three main strategies that work well out of the box:
    #🔑refine: create and refine an answer by sequentially going through each retrieved text chunk. This makes a separate LLM call per Node/retrieved chunk.
    #🔑compact (default): similar to refining but concatenating the chunks beforehand, resulting in fewer LLM calls.
    #🔑tree_summarize: create a detailed answer by going through each retrieved text chunk and creating a tree structure of the answer.
    #Take fine-grained control of your query workflows with the low-level composition API (https://docs.llamaindex.ai/en/stable/module_guides/deploying/query_engine/usage_pattern/#low-level-composition-api). This API lets you customize and fine-tune every step of the query process to match your exact needs, which also pairs great with Workflows(https://docs.llamaindex.ai/en/stable/module_guides/workflow/)  
    query_engine = index.as_query_engine(
        llm=llm,
        response_mode="tree_summarize",
    )
    result = query_engine.query("what is the meaning of life?")
    print(result)
    #The meaning of life is 42





def Evaluation_and_observability():
    #🟢The language model won’t always perform in predictable ways, so we can’t be sure that the answer we get is always correct. We can deal with this by evaluating the quality of the answer
    #🔑LlamaIndex provides built-in evaluation tools to assess response quality. These evaluators leverage LLMs to analyze responses across different dimensions. Let’s look at the three main evaluators available:
    #Let’s look at the three main evaluators available:

    #🔑FaithfulnessEvaluator: Evaluates the faithfulness of the answer by checking if the answer is supported by the context.
    #🔑AnswerRelevancyEvaluator: Evaluate the relevance of the answer by checking if the answer is relevant to the question.
    #🔑CorrectnessEvaluator: Evaluate the correctness of the answer by checking if the answer is correct.

    query_engine = # from the previous section
    llm = # from the previous section

    # query index
    evaluator = FaithfulnessEvaluator(llm=llm)
    response = query_engine.query(
        "What battles took place in New York City in the American Revolution?"
    )
    eval_result = evaluator.evaluate_response(response=response)
    eval_result.passing








async def full_exsample_from_local_path():
    #STEP 1 (load documents)
    reader = SimpleDirectoryReader(input_dir=r"C:\Users\didri\Desktop\bedrfiter mappe")   # Leser alle filer (txt, pdf, docx osv) i mappen
    documents = reader.load_data() # Laster dataene og returnerer en liste av Document-objekter med rå tekst og metadata
    print(f"documents : {documents}")


    #STEP 2 (setup chroma vector store)
    db = chromadb.PersistentClient(path="./didriks_chroma_db") # Kobler til en lokal persistent Chroma database (lagrer vektorer på disk)
    chroma_collection = db.get_or_create_collection("didrik") # Lager eller henter en samling (collection) i databasen kalt "didrik"
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)# Lager en vektor-lagringsinstans som bruker Chroma collection som backend


    #STEP 3 (ingest with pipeline)
    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(chunk_size=256, chunk_overlap=50), # Del opp dokumentteksten i mindre biter (chunks) på 256 tokens med 50 tokens overlapp
            HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5"), # Konverter hver tekstbit til en numerisk vektor (embedding) med en forhåndstrent HuggingFace-modell
        ],
        vector_store=vector_store # Resultatet av embeddingene skal lagres i Chroma vector store
    )

    #STEP 4 (run pipeline)
    await pipeline.arun(documents=documents) # Sender dokumentene gjennom splitter + embedding + lagring i vector_store
    print("ingestion complete.")

    
    #STEP 5 (build index from vector store)
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")# Initialiser embedding-modellen på nytt (brukes til spørringer)
    index = VectorStoreIndex.from_vector_store(vector_store,embed_model=embed_model)# Bygger en søkeindeks over de lagrede vektorene, slik at vi kan søke effektivt

    #STEP 6 (setup query engine)
    llm = HuggingFaceInferenceAPI(model_name="Qwen/Qwen2.5-Coder-32B-Instruct")# Initialiserer en stor språkmodell via API for generering av svar
    query_engine = index.as_query_engine(llm=llm, response_mode="tree_summarize")# Konverterer indeks til en spørringsmotor som bruker LLM og "tree_summarize" som svarmodus

    #STEP 7 (ask question)
    query = "hvilke selskaper i skien jobber med AI?"
    result = query_engine.query(query)# Kjører spørsmålet mot query engine som henter og oppsummerer relevante dokumenter
    print(f"\n 🔍 Query: {query}")
    print(f"📘 Answer: {result}")


    Correctness_Evaluator = CorrectnessEvaluator(llm=llm)# : Evaluate the correctness of the answer by checking if the answer is correct.

    AnswerRelevancy_Evaluator = AnswerRelevancyEvaluator(llm=llm)#Evaluate the relevance of the answer by checking if the answer is relevant to the question.

    FaithfulnessEvaluator_evaluator = FaithfulnessEvaluator(llm=llm) #: Evaluates the faithfulness of the answer by checking if the answer is supported by the context.
    

    response = query_engine.query(
        "hvilke selskaper i skien jobber med AI?"
    )
    eval_result = FaithfulnessEvaluator_evaluator.evaluate_response(response=response)
    eval_result.passing()
    #learn more about evaluation: https://huggingface.co/learn/agents-course/bonus-unit2/introduction







if __name__ == "__main__":
   # asyncio.run(main()) splits the documents, creates vectors of them, nodes is the  prepeared documents bits ready to be searched in
   #Storing_and_indexing_documents() #All information is automatically persisted within the ChromaVectorStore object and the passed directory path.
   #asyncio.run(full_exsample_from_local_path())


  