from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import DirectoryLoader

from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.embeddings import HuggingFaceInstructEmbeddings

import os

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

def load_pdfs(file):
    loader = PyPDFLoader(file)
    pages = loader.load()

    text = ""

    for page in pages:
        text += page.page_content

    text = text.replace("\t", " ")

    return text

def get_docs(folder):
    pdf_dict = {}
    
    pdfs = os.listdir(folder)
    for pdf in pdfs:
        path = os.path.join(folder, pdf)
        # Get the whole text35k tokens > 1024 tokens
        text = load_pdfs(path)

        #Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n"], chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.create_documents([text])
        pdf_dict[pdf] = docs
    
    return pdf_dict

    
# def rename_pdfs(folder):
#     pdfs = os.listdir(folder)

#     for pdf in pdfs:
#         chars = pdf.split()
#         new_file = chars[0] + "_" + chars[1] + ".pdf"
#         new_path = os.path.join(folder, new_file)
#         old_path = os.path.join(folder, pdf)
#         os.rename(old_path, new_path)

        
    
    
    