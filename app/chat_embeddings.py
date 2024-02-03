import os

from langchain.chains import LLMChain
from langchain.chat_models import AzureChatOpenAI
from langchain.prompts import PromptTemplate, load_prompt
from dotenv import load_dotenv


load_dotenv()

def create_chat(t=0.1):

    llm_chat = AzureChatOpenAI(openai_api_base=os.getenv("OPENAI_API_BASE"),
                    openai_api_version=os.getenv("OPENAI_API_VERSION"),
                    openai_api_key=os.getenv("OPENAI_API_KEY"),
                    openai_api_type=os.getenv("OPENAI_API_TYPE"),
                    deployment_name=os.getenv("DEPLOYMENT_NAME"),
                    temperature=t
    )
    return llm_chat

def create_embeddings(chunk_size=1):

    embeddings = OpenAIEmbeddings(
        deployment = os.getenv("EMBEDDING_DEPLOYMENT_NAME"),
        chunk_size=chunk_size
    )
       
    return embeddings

def create_chain(prompt=None):
    
    llm_chain = LLMChain(
        llm = create_chat(),
        prompt = prompt,
        verbose=True
    )

    return llm_chain


