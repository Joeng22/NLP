'''
One of the most common ways to store and search over unstructured data is to embed it and store the resulting embedding vectors, 
and then at query time to embed the unstructured query and retrieve the embedding vectors that are 'most similar' to the embedded query. 
A vector store takes care of storing embedded data and performing vector search for you.

This walkthrough uses the chroma vector database, which runs on your local machine as a library.

'''
import os
from getpass import getpass

from langchain.embeddings import  HuggingFaceInferenceAPIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

__import__('pysqlite3') 
import sys 
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

def main():
    HF_token = getpass()
    os.environ['HUGGINGFACEHUB_API_TOKEN'] = HF_token

    HuggingFace_embeddings = HuggingFaceInferenceAPIEmbeddings(
            api_key = HF_token,model_name = "BAAI/bge-base-en-v1.5"
        )

    raw_documents = TextLoader('Alice.txt').load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=20)

    documents = text_splitter.split_documents(raw_documents)

    db = Chroma.from_documents(documents, HuggingFace_embeddings)

    query = "Who is Alice?"
    docs = db.similarity_search(query)
    print(docs[0].page_content)

if __name__=="__main__":
    main()
