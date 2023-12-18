from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import faiss
from langchain.chains import RetrievalQA
from langchain.llms import openai
from dotenv import load_dotenv
import os

# Load environment variables - HUGGINGFACEHUB_API_TOKEN, 
load_dotenv()

api_key = os.getenv('HUGGINGFACEHUB_API_TOKEN')
openai_key = os.getenv("OPENAI_API_KEY")

# Define the ContextualAnswers class
class ContextualAnswers():
    # Initialize the ContextualAnswers object with a repository ID
    def __init__(self) -> None:
        self.llm = None
        self.embeddings = OpenAIEmbeddings()
        self.qa = None
        self.texts = None
        self.retriever = None
    
    # Load and split documents into chunks    
    def load_and_split_documents(self, uploaded_file):
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(separators=['\n', ' ', '\n\n'], chunk_size=500, chunk_overlap=100)
        self.texts = text_splitter.create_documents([uploaded_file])
    
    # Initialize the retriever
    def init_retriever(self):
        # Create a vectorstore from documents
        print(len(self.texts))
        db = faiss.FAISS.from_documents(documents=self.texts, embedding=self.embeddings)
        # Create retriever interface
        self.retriever = db.as_retriever()
    
    # Initialize the language model
    def init_llm(self, local=False, **kwargs):
        self.llm = openai.OpenAI(temperature=0, api_key=openai_key)
     
    # Initialize the chain      
    def init_chain(self, chain_type, uploaded_file=None):
        self.init_llm()
        if uploaded_file: 
            self.load_and_split_documents(uploaded_file)
            self.init_retriever()
            self.qa = RetrievalQA.from_chain_type(llm=self.llm, chain_type=chain_type, retriever=self.retriever)
            
    # Run the chain to get the answer    
    def get_answer(self, query_text):
        if self.qa:
            print(self.retriever.get_relevant_documents(query_text))
            return self.qa({"query": query_text})
