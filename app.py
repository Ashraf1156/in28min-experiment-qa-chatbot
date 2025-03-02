import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import glob
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_google_genai import HarmCategory, HarmBlockThreshold
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader

# Load environment variables from .env file
load_dotenv()
os.getenv("GOOGLE_API_KEY")

# Prompt the user to input the Google AI API key if it is not set in the environment variables
if "GOOGLE_API_KEY" not in os.environ:
    import getpass
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google AI API key: ")

# Configure the Google Generative AI chat model with specified parameters
chat_model = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",  # Specify the model version
    temperature=0.7,  # Control response creativity
    max_tokens=500,  # Limit the number of tokens in the output
    safety_settings={
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,  # Disable blocking for harmful content
    },
    max_retries=2,  # Set maximum retries for handling errors
)

# Function to load text transcripts from a directory using TextLoader instead of DirectoryLoader
def load_transcripts_from_directory(directory_path):
    file_paths = glob.glob(os.path.join(directory_path, "*.txt"))
    documents = []
    
    for file_path in file_paths:
        try:
            loader = TextLoader(file_path)
            documents.extend(loader.load())
            print(f"Successfully loaded: {file_path}")
        except Exception as e:
            print(f"Error loading file {file_path}")
            print(f"  Error details: {str(e)}")
    
    if not documents:
        return ""
        
    text = "\n".join([doc.page_content for doc in documents])
    return text

# Function to split large text into smaller chunks for efficient processing
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=75)  # Define chunk size and overlap
    chunks = text_splitter.split_text(text)  # Split text into chunks
    return chunks

# Function to create a vector store for text embedding
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")  # Use Google Generative AI for embeddings
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)  # Create a FAISS vector store from text chunks

    try:
        vector_store.save_local("faiss_index")  # Save the vector store locally
        print("Vector store saved successfully!")
    except Exception as e:
        print(f"Error saving vector store: {e}")

# Function to set up a conversational chain using a chat prompt template
def get_conversational_chain():
    # Define a prompt template for the conversational chain
    prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert Technical Support Specialist in an Edutech Organization. "
                "Your role is to solve learners' queries by providing direct, concise, and humanized answers. "
                "If transcripts are provided, ensure the answers are based strictly on the transcript content. "
                "If the answer cannot be found in the context, respond with 'Answer not found in provided context.'"
            ),
            (
                "human",
                (
                    "Hi [Learner's Name],\n"
                    "---------------\n"
                    "Please address the following query strictly based on the context below:\n"
                    "Context: ```{context}```\n"
                    "Query: {question}\n"
                    "---------------\n"
                    "Thanks and regards,\n"
                    "Mohammed Ashraf Hussain - in28minutes Team\n"
                    "Do not forget to explore our Step-by-Step cloud demos – constantly updated for seamless learning!\n"
                    "Azure - https://www.in28minutes.com/azure-bookshelf\n"
                    "AWS - https://www.in28minutes.com/aws-bookshelf\n"
                    "Google Cloud - https://www.in28minutes.com/gcp-bookshelf\n"
                )
            ),
        ]
    )
    chain = prompt_template | chat_model  # Combine the prompt template with the chat model
    return chain

# Function to handle user input and generate responses
def user_input(user_question, context):
    if not context.strip():  # Check if the context is empty
        st.write("Reply:", "Answer is not available in the context.")
        return

    chain = get_conversational_chain()  # Set up the conversational chain
    print("Question sent to LLM:", user_question)
    print("Context sent to LLM:", context)

    response = chain.invoke({
        "question": user_question,  # Provide user query
        "context": context  # Provide relevant context
    })

    st.write("Reply:", response.content)  # Display the response

# Main function to initialize the Streamlit app
def main():
    st.set_page_config("Chat Transcripts", layout="wide")  # Configure the Streamlit page
    st.header("in28minutes Q&A ChatBot💁")  # Display the header

    transcript_dir = "Transcripts"  # Directory containing the transcript files

    # Process and index the transcript files
    with st.spinner("Processing documents..."):
        try:
            raw_text = load_transcripts_from_directory(transcript_dir)  # Load transcripts
            if raw_text.strip() == "":
                st.error("Transcripts are blank! Provide a different dir or review file content formatting in existing files if meant to be full!")
            else:
                text_chunks = get_text_chunks(raw_text)  # Split transcripts into chunks
                get_vector_store(text_chunks)  # Create and save vector store
                st.success("Transcripts Processed Successfully!")
        except FileNotFoundError:
            st.error(f"Directory not found: {transcript_dir}. Check path validity.")
        except Exception as e:
            st.error(f"A different problem was encountered loading/processing transcripts: {e}")

    # Accept user queries and respond based on the processed transcripts
    user_question = st.text_input("Ask a Question from the Transcripts")
    if user_question:
        try:
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")  # Load embeddings
            new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)  # Load vector store
            print("Vector store loaded successfully!")
            docs = new_db.similarity_search(user_question, k=5)  # Search for similar documents
            context = "\n".join([doc.page_content for doc in docs if doc.page_content.strip()])  # Extract relevant context
            user_input(user_question, context)  # Get response
        except Exception as e:
            st.error(f"Error interacting with the vector store: {e}")

# Run the app
if __name__ == "__main__":
    main()