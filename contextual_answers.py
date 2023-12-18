# from langchain.llms import HuggingFaceHub
from huggingface_hub import InferenceClient
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
# from langchain.embeddings HuggingFaceInferenceAPIEmbeddings
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.vectorstores import faiss
from langchain.chains import RetrievalQA
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from dotenv import load_dotenv
from langchain.llms import openai
from langchain.chat_models import ChatOpenAI
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.prompts import ChatPromptTemplate
from langchain import hub
from langchain.schema import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.chains.question_answering import load_qa_chain
import os

# Load environment variables - HUGGINGFACEHUB_API_TOKEN, 
load_dotenv()

api_key = os.getenv('HUGGINGFACEHUB_API_TOKEN')
openai_key = os.getenv("OPENAI_API_KEY")

# Define the ContextualAnswers class
class ContextualAnswers():
    # Initialize the ContextualAnswers object with a repository ID
    def __init__(self) -> None:
    # def __init__(self, repo_id) -> None:
        # self.repo_id = repo_id
        self.llm = None
        #self.embeddings = HuggingFaceEmbeddings()
        self.embeddings = OpenAIEmbeddings()
        self.qa = None
        self.texts = None
        self.retriever = None
    
    # Load and split documents into chunks    
    def load_and_split_documents(self, uploaded_file):
        # documents = [uploaded_file.read().decode()]
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
        # if local:
        #     tokenizer = AutoTokenizer.from_pretrained(self.repo_id, token=api_key)
        #     self.llm = AutoModelForCausalLM.from_pretrained(self.repo_id, token=api_key)
        #     pipe = pipeline("text-generation",
        #                        model=self.llm,
        #                         tokenizer=tokenizer,
        #              )
        #     hf = HuggingFacePipeline(pipeline=pipe)
        #     self.llm = hf
        # else:
            # self.llm = HuggingFaceHub(repo_id=self.repo_id, model_kwargs=kwargs)
            # self.llm = InferenceClient(model=self.repo_id)

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

        