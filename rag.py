from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma


# Path to the PDF file
pdf_file_path = './Book.pdf'


# Initialising the loader to load the pdf
loader = PyPDFLoader(pdf_file_path)

# Loading the pdf as a Document
documents = loader.load()

# Intialise the Text Splitter Object
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

# Splitting the loaded data into chunks
chunks = text_splitter.split_documents(documents)

# Initialise a Embedder Object to embed the chunks
embedder = OllamaEmbeddings(model="mxbai-embed-large") 

# Initialising a Chroma Object to embed the chunks
# into vector and store them
vectore_store = Chroma(
    collection_name='PDF_reader',
    persist_directory = './PDFData',
    embedding_function = embedder
)

# Storing the Vectors in the Chroma DB
vectore_store.add_documents(chunks)

# Initialising a Retriever Object that retrieves text 
# from the ChromaDB
retriever = vectore_store.as_retriever(
    search_kwargs={'k':30}
)