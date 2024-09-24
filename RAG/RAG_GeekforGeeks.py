from langchain.document_loaders import WebBaseLoader

def main():
    #add website data
    URL = ["https://www.geeksforgeeks.org/stock-price-prediction-project-using-tensorflow/",
        "https://www.geeksforgeeks.org/training-of-recurrent-neural-networks-rnn-in-tensorflow/",
        "https://www.freecodecamp.org/news/python-list-length-how-to-get-the-size-of-a-list-in-python/"]

    #load the data
    data = WebBaseLoader(URL)
    #extract the content
    content = data.load()
    #print("Content:",content)
    print("size of list of content:",len(content))    # prints size of list. Since 3 URLs given, size of content = 3


    #Split the data
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=256,chunk_overlap=50)
    chunking = text_splitter.split_documents(content)

    print("size of chunking:",len(chunking))


    #Word embedding
    from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings

    import os
    from getpass import getpass

    # get your free access token from HuggingFace and paste it here
    HF_token = getpass()
    os.environ['HUGGINGFACEHUB_API_TOKEN'] = HF_token

    embeddings = HuggingFaceInferenceAPIEmbeddings(
        api_key = HF_token,model_name = "BAAI/bge-base-en-v1.5"
    )



    from langchain.vectorstores import Chroma
    vectorstore = Chroma.from_documents(chunking,embeddings)


    retriever = vectorstore.as_retriever(search_type="mmr",search_kwargs={"k":3})
    query = "what is recurrent neural network?"
    docs_rel = retriever.get_relevant_documents(query)
    print(docs_rel)

    prompt = f"""
    <|system|>>
    You are an AI Assistant that follows instructions extremely well.
    Please be truthful and give direct answers. Please tell 'I don't know' if user query is not in context
    </s>
    <|user|>
    {query}
    </s>
    <|assistant|>
    """

    from langchain.llms import HuggingFaceHub
    from langchain.chains import RetrievalQA

    model = HuggingFaceHub(repo_id="HuggingFaceH4/zephyr-7b-alpha",
                        model_kwargs={"temperature":0.5,
                                        "max_new_tokens":512,
                                        "max_length":64
                                        })

    qa = RetrievalQA.from_chain_type(llm=model,retriever=retriever,chain_type="stuff")
    response = qa(prompt)
    print(response['result'])

if __name__=="__main__":
    main()