
# Importing required libraries for setting up a language model (LLM) based question answering system (ChatBot)
from langchain.document_loaders import WebBaseLoader
from langchain.document_loaders import PyPDFLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma                               # to store the data in a database
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI                            # for the language model (LLM)
from langchain.prompts import PromptTemplate                            # for defining prompt templates
from langchain.chains import RetrievalQA                                # for creating retrieval-based question answering systems
import chainlit as cl                                                   # for creating the user interface


# Defining helper functions for setting up and managing components of a language model (LLM) based question answering system.
def get_openai_api_key(file_path='openai_api_key.txt'):
    """Read the OpenAI API key from a file."""
    with open(file_path, 'r') as f:
        return f.read()

def remove_blank_lines(text):
    """Remove blank lines from text content."""
    lines = text.split('\n')
    non_empty_lines = [line.strip() for line in lines if line.strip()]
    return '\n'.join(non_empty_lines)

def load_documents_from_web(url):
    """Load documents from a web URL."""
    loader = WebBaseLoader(url)
    return loader.load()
def load_document_from_pdf(file_path):
    """Load a document from a local file in PDF format."""
    loader = PyPDFLoader(file_path)
    return loader.load()
def load_document_from_csv(file_path):
    """Load a document from a local CSV file."""
    loader = CSVLoader(file_path=file_path)
    return loader.load()
def load_pdf_documents_from_directory(directory_path):
    """Load PDF documents from a directory."""
    glob_pattern = '**/*.pdf'  # Load PDF files from all subdirectories
    loader = DirectoryLoader(directory_path, glob=glob_pattern)
    return loader.load()

def split_documents_into_chunks(docs, chunk_size=700, chunk_overlap=100):
    """Split documents into chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(docs)

def create_vector_database(chunks, embeddings, persist_directory):
    """Create vector database from document chunks."""
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    vectordb.persist()
    return vectordb

def initialize_chat_model(model_name, openai_api_key):
    """Initialize the chat model."""
    return ChatOpenAI(model_name=model_name, temperature=0.5, openai_api_key=openai_api_key, streaming=True)


@cl.on_chat_start
async def on_chat_start():

    # Get OpenAI API key
    OPENAI_API_KEY = get_openai_api_key()
    # Load documents
    docs = load_documents_from_web("https://www.jioinstitute.edu.in/faq")

    # Remove blank lines from each document's text content
    for doc in docs:
        doc.page_content = remove_blank_lines(doc.page_content)

    # Split documents into chunks
    chunks = split_documents_into_chunks(docs)
    # Create OpenAI embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    # Create vector database
    persist_directory = 'docs/chroma_db/'
    vectordb = create_vector_database(chunks, embeddings, persist_directory)
    # Define retriever
    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    # Initialize chat model
    llm_name = 'gpt-3.5-turbo'
    llm = initialize_chat_model(llm_name, OPENAI_API_KEY)
    # defining template
    template = """Please utilize the provided context to respond to the question at the end. If you're unsure of the answer, it's perfectly fine to acknowledge that you don't know rather than attempting to fabricate a response. Keep your answer concise, limiting it to three sentences at most. Additionally, remember to conclude your response with "thanks for asking!
        {context}
        Question: {question}
        Helpful Answer:"""
    # creating prompt chain
    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"],template=template)

    # creating qa retrieval chain
    qa_chain = RetrievalQA.from_chain_type(llm,
                                       retriever=retriever,
                                       return_source_documents=True,
                                       chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})
    # Storing the Chain in User Session:
    cl.user_session.set("qa_pipeline", qa_chain)
    # Sending a Greeting Message:
    await cl.Message(author="JI Assistant", content="Hello ! How may I help you ? ").send()


# When the user sends a message, the @cl.on_message decorator triggers the on_message function.
@cl.on_message
async def on_message(message: cl.Message):
    # Retrieving the QA chain from the user session
    '''Uses the query to search for relevant information using the retrieval system.
    Generates a response that answers the user's question, potentially using the LLM.
    Stores the generated response in the response variable.'''
    answering_pipeline = cl.user_session.get("qa_pipeline")

    # Processing the user's query using the chain
    '''Extracts the user's query from the message (message.content).
    Creates a dictionary with the query as the key and value.
    Passes this dictionary to the answering_pipeline object.'''
    response = answering_pipeline({"query": message.content})
    
    # Creates a message object with the answer retrieved from the response dictionary (response['result'])
    msg = cl.Message(content=response['result'], author="JI Assistant")
    
    # Sending the response message
    await msg.send()



# chainlit run chatbot.py -w
    
