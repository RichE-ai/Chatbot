from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import os

# Load OpenAI API key from .env
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Load PDF document
loader = PyPDFLoader("data/VdBbuildingreg.pdf")
documents = loader.load()

# Split document into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(documents)

# Create vector store from chunks
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(docs, embeddings)

# Create retriever-based QA chain
retriever = vectorstore.as_retriever()
qa_chain = RetrievalQA.from_chain_type(
    llm=OpenAI(temperature=0),
    chain_type="stuff",
    retriever=retriever,
)

# Test query
query = "What are the height restrictions for new buildings?"
answer = qa_chain.run(query)
print("Answer:", answer)

