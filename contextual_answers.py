from langchain.llms import HuggingFaceHub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from dotenv import load_dotenv
import os

os.environ['TRANSFORMERS_CACHE'] = './models/'
# Load environment variables - HUGGINGFACEHUB_API_TOKEN for HuggingFaceEmbeddings
load_dotenv()
from huggingface_hub import login

api_key = os.getenv('HUGGINGFACEHUB_API_TOKEN')
login(api_key)


# Define the ContextualAnswers class
class ContextualAnswers():
    # Initialize the ContextualAnswers object with a repository ID
    def __init__(self, repo_id, ) -> None:
        self.repo_id = repo_id
        self.llm = None
        self.embeddings = HuggingFaceEmbeddings()
        # self.embeddings = HuggingFaceInferenceAPIEmbeddings(api_key=api_key, 
        #                                                     model_name="intfloat/multilingual-e5-small")
        self.qa = None
        self.texts = None
        self.retriever = None
    
    # Load and split documents into chunks    
    def load_and_split_documents(self, uploaded_file):
        documents = [uploaded_file.read().decode()]
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        self.texts = text_splitter.create_documents(documents)
    
    # Initialize the retriever
    def init_retriever(self):
        # Create a vectorstore from documents
        db = FAISS.from_documents(documents=self.texts, embedding=self.embeddings)
        # Create retriever interface
        self.retriever = db.as_retriever()
    
    # Initialize the language model
    def init_llm(self, local=False, **kwargs):
        if local:
            tokenizer = AutoTokenizer.from_pretrained(self.repo_id, token=api_key)
            self.llm = AutoModelForCausalLM.from_pretrained(self.repo_id, token=api_key)
            pipe = pipeline("text-generation",
                               model=self.llm,
                                tokenizer=tokenizer,
                     )
            hf = HuggingFacePipeline(pipeline=pipe)
            self.llm = hf

        else:
            self.llm = HuggingFaceHub(repo_id=self.repo_id, model_kwargs=kwargs)
          
    # Initialize the chain      
    def init_chain(self, chain_type, uploaded_file=None):
        self.init_llm(local=True)
        if uploaded_file:
            self.load_and_split_documents(uploaded_file)
            self.init_retriever()
            self.qa = RetrievalQA.from_chain_type(llm=self.llm, chain_type=chain_type, retriever=self.retriever)
    
    # Run the chain to get the answer    
    def get_answer(self, query_text):
        if self.qa:
            return self.qa.run(query_text) 
          
