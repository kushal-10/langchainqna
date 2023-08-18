from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from InstructorEmbedding import INSTRUCTOR
import textwrap

from lc.model import load_model, build_pipeline
from lc.database import retrieve


def wizard_chain(query="Who are the main users (participants) in the two-sided market?"):
    # Get the individual components
    model, tokenizer = load_model()
    hf_pipe = build_pipeline(model, tokenizer)
    rt = retrieve()

    # Create the chain
    qna_chain = RetrievalQA.from_chain_type(llm=hf_pipe,
                                            chain_type="stuff",
                                            retriever=rt,
                                            return_source_documents=True)
    
    response = qna_chain(query)
    print(process_llm_response(response))
    
    return None


def wrap_text_preserve_newlines(text, width=110):
    # Split the input text into lines based on newline characters
    lines = text.split('\n')

    # Wrap each line individually
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]

    # Join the wrapped lines back together using newline characters
    wrapped_text = '\n'.join(wrapped_lines)

    return wrapped_text

def process_llm_response(llm_response):
    print(wrap_text_preserve_newlines(llm_response['result']))
    print('\n\nSources:')
    for source in llm_response["source_documents"]:
        print(source.metadata['source'])