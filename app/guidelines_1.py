#%%
from chat_embeddings import *
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv

load_dotenv()


""" Princípios de Prompt:

1. Escreva instruções específicas e claras
2. Dê tempo ao modelo para "pensar"

Taticas: Use delimitadores para indicas partes do input
"""

text = f"You should express what you want a model to do by \
providing instructions that are as clear and \
specific as you can possibly make them. \
This will guide the model towards the desired output, \
and reduce the chances of receiving irrelevant \
or incorrect responses. Don't confuse writing a \
clear prompt with writing a short prompt. \
In many cases, longer prompts provide more clarity \
and context for the model, which can lead to \
more detailed and relevant outputs."

#%%
template = """
Summarize the text delimited by triple backticks \
into a single sentence.
human = {text}
AI = """

prompt = PromptTemplate(
    
    template=template,
    input_variables=['text']
)

llm_chain = create_chain(prompt)

response = llm_chain.predict(text=text)
print(response)


#%%
## Saida estruturada como HTML ou JSON ## 

text = """Generate a list of three made-up book titles along \ 
with their authors and genres. 
Provide them in JSON format with the following keys: 
book_id, title, author, genre.
"""

template = """
Do what the following instructions are saying.\
human = {text}
AI = """

prompt = PromptTemplate(
    
    template=template,
    input_variables=['text']
)

llm_chain = create_chain(prompt)

response = llm_chain.predict(text=text)
print(response)


# %%
## Verificar se as condições foram satisfazidas ##

text = f"""
Making a cup of tea is easy! First, you need to get some \ 
water boiling. While that's happening, \ 
grab a cup and put a tea bag in it. Once the water is \ 
hot enough, just pour it over the tea bag. \ 
Let it sit for a bit so the tea can steep. After a \ 
few minutes, take out the tea bag. If you \ 
like, you can add some sugar or milk to taste. \ 
And that's it! You've got yourself a delicious \ 
cup of tea to enjoy.
"""

template = """
You will be provided with text delimited by triple quotes. 
If it contains a sequence of instructions, \ 
re-write those instructions in the following format:

Step 1 - ...
Step 2 - …
…
Step N - …

If the text does not contain a sequence of instructions, \ 
then simply write \"No steps provided.\"

\"\"\"{text}\"\"\"
"""

prompt = PromptTemplate(
    
    template=template,
    input_variables=['text']
)

llm_chain = create_chain(prompt)

response = llm_chain.run(text=text)
print(response)


# %%
## Neste exemplo as condições não foram satisfeitas, então ele não retornou os steps ##

text = f"""
The sun is shining brightly today, and the birds are \
singing. It's a beautiful day to go for a \ 
walk in the park. The flowers are blooming, and the \ 
trees are swaying gently in the breeze. People \ 
are out and about, enjoying the lovely weather. \ 
Some are having picnics, while others are playing \ 
games or simply relaxing on the grass. It's a \ 
perfect day to spend time outdoors and appreciate the \ 
beauty of nature.
"""

template = """
You will be provided with text delimited by triple quotes. 
If it contains a sequence of instructions, \ 
re-write those instructions in the following format:

Step 1 - ...
Step 2 - …
…
Step N - …

If the text does not contain a sequence of instructions, \ 
then simply write \"No steps provided.\"

\"\"\"{text}\"\"\"
"""

prompt = PromptTemplate(
    
    template=template,
    input_variables=['text']
)

llm_chain = create_chain(prompt)

response = llm_chain.run(text=text)
print(response)


# %%
## few-shot learning - é um método de treinamento de modelo de inteligência artificial que envolve o treinamento de um modelo com um número muito limitado de exemplos

text = """<child>: Teach me about patience.

<grandparent>: The river that carves the deepest \ 
valley flows from a modest spring; the \ 
grandest symphony originates from a single note; \ 
the most intricate tapestry begins with a solitary thread.

<child>: Teach me about resilience."""

template = """
Your task is to answer in a consistent style.

\"\"\"{text}\"\"\"
"""

prompt = PromptTemplate(
    
    template=template,
    input_variables=['text']
)

llm_chain = create_chain(prompt)

response = llm_chain.run(text=text)
print(response)

# %%
