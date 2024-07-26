#integrate code with OPENAI API

import os 
from constants import OPENAI_KEY
from langchain.llms import OpenAI
import streamlit as st 
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain

os.environ["OPENAI_API_KEY"] = OPENAI_KEY
#streamlit 

st.title("My first LangChain Project")
input_text = st.text_input("search the required topic here")

#prompt templates 
first_input_prompt= PromptTemplate(
    input_variables=['name'],
    template="Tell me about {name}"
)


#llm model
llm = OpenAI(temperature = 0.8)
chain1 = LLMChain(llm=llm, prompt = first_input_prompt, verbose = True, output_key = 'person')

second_input_prompt= PromptTemplate(
    input_variables=['person'],
    template="when was  {person} born"
)

chain2 = LLMChain(llm=llm, prompt = second_input_prompt, verbose = True, output_key = 'dob')

main_chain = SequentialChain(chains = [chain1, chain2], input_variables = ['name'], output_variables = ['person','dob'], verbose = True)

if input_text:
    st.write(main_chain({'name':input_text}))
