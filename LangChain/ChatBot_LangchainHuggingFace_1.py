'''
Chatbot tutorial using Langchain and Open model from Huggingface
'''

from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import HuggingFaceHub
import os
from getpass import getpass

import streamlit as st

input_token = st.text_input('Enter Your Hugging Face Token: ') 

if (input_token):
    st.title('My Chatbot')
    input_text = st.text_input('Enter Your Text: ') 

    from langchain.prompts import PromptTemplate
    title_template = PromptTemplate(
        input_variables = ['concept'], 
        template='{concept}'
    )
    script_template = PromptTemplate(
        input_variables = ['title', 'wikipedia_research'], 
        template='''Give me more details based on the title {title} 
        while making use of the information and knowledge obtained from the Wikipedia research:{wikipedia_research}'''
    )

    from langchain.memory import ConversationBufferMemory

    # We use the ConversationBufferMemory to can be used to store a history of the conversation between the user and the language model. 
    # This information can be used to improve the language model's understanding of the user's intent, and to generate more relevant and coherent responses.
    
    memoryT = ConversationBufferMemory(input_key='concept', memory_key='chat_history')
    memoryS = ConversationBufferMemory(input_key='title', memory_key='chat_history')


    # get your free access token from HuggingFace and paste it here
    HF_token = input_token
    os.environ['HUGGINGFACEHUB_API_TOKEN'] = HF_token

    model = HuggingFaceHub(repo_id="HuggingFaceH4/zephyr-7b-alpha",
                        model_kwargs={"temperature":0.5,
                                        "max_new_tokens":512,
                                        "max_length":64
                                        })

    from langchain.chains import LLMChain
    chainT = LLMChain(llm=model, prompt=title_template, verbose=True, output_key='title', memory=memoryT)
    chainS = LLMChain(llm=model, prompt=script_template, verbose=True, output_key='script', memory=memoryS)


    from langchain.utilities import WikipediaAPIWrapper
    wikipedia = WikipediaAPIWrapper()
    
    # Display the output if the the user gives an input
    if input_text: 
        title = chainT.run(input_text)
        wikipedia_research = wikipedia.run(input_text) 
        script = chainS.run(title=title, wikipedia_research=wikipedia_research)
    
        st.write(title) 
        st.write(script) 
    
        with st.expander('Wikipedia-based exploration: '): 
            st.info(wikipedia_research)