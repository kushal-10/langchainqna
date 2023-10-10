from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import DirectoryLoader

from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.embeddings import HuggingFaceInstructEmbeddings

def create_db(inp):

    db = 'data_' + inp
    op = 'outputs/' + inp + '_db'

    '''
    Creates and saves a Chroma database for the required literature 
    '''

    embedding = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl",
                                                      model_kwargs={"device": "cuda"})
    
    loader = DirectoryLoader(db, glob="./*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    
    persist_directory = op

    vectordb = Chroma.from_documents(documents=texts,
                                    embedding=embedding,
                                    persist_directory=persist_directory)
    # persist the db to disk
    vectordb.persist()
    
    return None

class GetRetriever():
    def __init__(self, data='new') -> None:
        self.data_dir = 'outputs/' + str(data) + '_db'

    def retrieve_data(self, k=3):
        '''
        Define a retriever to fetch relevant documents from the database
        '''
        persist_directory = self.data_dir
        embedding = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl",
                                                        model_kwargs={"device": "cuda"})
        
        vector_db = Chroma(persist_directory=persist_directory, embedding_function=embedding)
        retriever = vector_db.as_retriever(search_kwargs={"k": k})

        return retriever
