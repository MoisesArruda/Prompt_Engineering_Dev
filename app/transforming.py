"""Neste notebook iremos trabalhar com o modelo explorando a possibilidade\
    de transformar o texto em um tradutor, com corretor, ajusto de tonalidade e conversação formal."""

#%%
from chat_embeddings import *
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv

load_dotenv()

#%%

## Tradutor ##

text = f"""
Hi, I would like to order a blender
"""

template = """
Translate the following English text to Spanish:

Text: {text}
"""

prompt = PromptTemplate(
    
    template=template,
    input_variables=['text']
)

llm_chain = create_chain(prompt)

response = llm_chain.run(text=text)
print(response)

# %%

text = f"""
Combien coûte le lampadaire?
"""

template = """
Tell me which language this is: 

Text: {text}
"""

prompt = PromptTemplate(
    
    template=template,
    input_variables=['text']
)

llm_chain = create_chain(prompt)

response = llm_chain.run(text=text)
print(response)

# %%

text = f"""
I want to order a basketball
"""

template = """
Translate the following  text to French and Spanish
and English pirate:

Text: {text}
"""

prompt = PromptTemplate(
    
    template=template,
    input_variables=['text']
)

llm_chain = create_chain(prompt)

response = llm_chain.run(text=text)
print(response)

# %%
text = f"""
'Would you like to order a pillow?'
"""

template = """
Translate the following text to Spanish in both the \
formal and informal forms: 

Text: {text}
"""

prompt = PromptTemplate(
    
    template=template,
    input_variables=['text']
)

llm_chain = create_chain(prompt)

response = llm_chain.run(text=text)
print(response)


