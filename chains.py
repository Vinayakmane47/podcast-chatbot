from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.chat_models import ChatOpenAI
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
load_dotenv()
llm = ChatOpenAI()

def give_documents(text): 
    document = Document(
        page_content= text,
        metadata={"source": "audio.wav"}
    ) 
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20,
)
    documents = text_splitter.split_documents([document])
    return documents 

def give_retriver(documents): 
    db=FAISS.from_documents(documents,OpenAIEmbeddings())
    return db.as_retriever()



from langchain_core.prompts import ChatPromptTemplate

def give_prompt(): 
    system_template =  """
    Given the following context translate this to english if required and 
    answer the user question based on context. keep answers long and include every minute details .
    {context}
    """

    prompt_template = ChatPromptTemplate.from_messages(
        [("system", system_template), ("user", "{question}")]
    )

    return prompt_template 


def give_chain(): 
    prompt_template = give_prompt()
    llm = ChatOpenAI()
    chain = prompt_template | llm | StrOutputParser()
    return chain 


def invoke_chain(text,question): 
    chain = give_chain()
    result_chain = chain.invoke({"context": text , "question":question})
    return result_chain 






