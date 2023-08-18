from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import DirectoryLoader

from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.embeddings import HuggingFaceInstructEmbeddings

def create_db():
    '''
    Creates and saves a Chroma database for the required literature 
    '''

    embedding = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl",
                                                      model_kwargs={"device": "cuda"})
    
    loader = DirectoryLoader('data_two_sided', glob="./*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    # len(documents)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    
    persist_directory = 'outputs/two_sided_db'

    vectordb = Chroma.from_documents(documents=texts,
                                    embedding=embedding,
                                    persist_directory=persist_directory)
    # persist the db to disk
    vectordb.persist()
    
    return None

def retrieve():
    '''
    Define a retriever to fetch relevant documents from the database
    '''
    persist_directory = 'outputs/two_sided_db'
    embedding = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl",
                                                      model_kwargs={"device": "cuda"})
    
    vector_db = Chroma(persist_directory=persist_directory, embedding_function=embedding)
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})

    return retriever