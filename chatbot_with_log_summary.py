
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
import csv
import time


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

@cl.on_chat_start
async def on_chat_start():
    # Initialize chat model
    llm_name = 'gpt-3.5-turbo'
    llm = initialize_chat_model(llm_name, OPENAI_API_KEY)

    # Storing the LLM in User Session:
    cl.user_session.set("llm", llm)
    cl.user_session.set("conversation_history", [])

    # Sending a Greeting Message:
    await cl.Message(author="JI Assistant", content="Hello ! How may I help you ? ").send()


# When the user sends a message, the @cl.on_message decorator triggers the on_message function.
@cl.on_message
async def on_message(message: cl.Message):
    
    # Record start time for retrieval
    start_retrieval_time = time.time()

    # Retrieve documents directly (without creating a chain)
    documents = vectordb.max_marginal_relevance_search(query=message.content, k=3)  # Adjust k for desired retrieval count
    # similarity_search
    
    # Record end time for retrieval and calculate time
    end_retrieval_time = time.time()
    retrieval_time = round(end_retrieval_time - start_retrieval_time, 3)

    # Extract relevant information from retrieved documents
    closest_similar_match1 = documents[0].page_content
    closest_similar_match2 = documents[1].page_content
    closest_similar_match3 = documents[2].page_content

    similarity_context = [closest_similar_match1, closest_similar_match2, closest_similar_match3]

    # Get conversation history
    conversation_history = cl.user_session.get("conversation_history", [])

    # Include relevant parts of conversation history in prompt and join the similarity context
    context_for_prompt = "\n".join([entry["response"] for entry in conversation_history[-3:]]+(similarity_context))  # Adjust window size as needed

    # Create prompt for LLM
    prompt = f"Please utilize the provided context to respond to the question at the end. If you're unsure of the answer, it's perfectly fine to acknowledge that you don't know rather than attempting to fabricate a response. Keep your answer concise, limiting it to three sentences at most. Additionally, you can conclude your response with 'thanks for asking!'\n{context_for_prompt}\nQuestion: {message.content}\nHelpful Answer:"

    # Get chat model
    llm = cl.user_session.get('llm')

    start_summarization_time = time.time()
    response = llm.invoke(input=prompt) # Generate response using LLM
    end_summarization_time = time.time()

    # Summarization Time
    summarization_time = round(end_summarization_time - start_summarization_time, 3)
    
    # Summarized Response
    summarized_response = dict(response).get("content", "").strip()

    # Update conversation history
    conversation_history.append({"user_query": message.content, "response": summarized_response})
    cl.user_session.set("conversation_history", conversation_history)
    
    # print("conversation_history:\n\t", conversation_history)

    # Open CSV file for writing
    with open('chat_log.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([message.content, retrieval_time, closest_similar_match1, closest_similar_match2, closest_similar_match3,  summarization_time, summarized_response])  # Placeholder for summarization time

    # Creates a message object with the answer retrieved from the response dictionary (response['result'])
    msg = cl.Message(content=summarized_response, author="JI Assistant")
    
    # Sending the response message
    await msg.send()




# chainlit run chatbot_scratch.py -w