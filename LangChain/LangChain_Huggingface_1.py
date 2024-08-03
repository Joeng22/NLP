'''
LLM sample use case using Langchain and Hugging 
'''

from langchain.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
import os
from getpass import getpass


# get your free access token from HuggingFace and paste it here
HF_token = getpass()
os.environ['HUGGINGFACEHUB_API_TOKEN'] = HF_token

model = HuggingFaceHub(repo_id="HuggingFaceH4/zephyr-7b-alpha",
                       model_kwargs={"temperature":0.5,
                                     "max_new_tokens":512,
                                     "max_length":64
                                    })




HF_token = getpass()

prompt = model("where is Taj Mahal")
    
print(prompt)