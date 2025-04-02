import locale 
from getpass import getpass
from langchain.document_loaders import GitHubIssuesLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os 

locale.getpreferredencoding = lambda: "UTF-8"
Github_Token = os.getenv("Github_Token")
ACCESS_TOKEN = getpass(Github_Token)

loader = GitHubIssuesLoader(repo="huggingface/peft", access_token=ACCESS_TOKEN, include_prs=False, state="all")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=30)
chunked_docs = splitter.split_documents(docs)


https://huggingface.co/learn/cookbook/rag_zephyr_langchain