from langchain.chains import RetrievalQA, VectorDBQA
from langchain.document_loaders import TextLoader
from InstructorEmbedding import INSTRUCTOR
import textwrap

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from lc.model import load_model, build_pipeline
from lc.model import load_t5_model, build_t5, load_incite
from lc.database import retrieve, retrieve_last_mile, retrieve_new


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

def t5_chain(query="Who are the main users (participants) in the two-sided market?"):
    # Get the individual components
    model, tokenizer = load_t5_model()
    t5_pipe = build_t5(model, tokenizer)
    rt = retrieve()

    # Create the chain
    qna_chain = RetrievalQA.from_chain_type(llm=t5_pipe,
                                            chain_type="stuff",
                                            retriever=rt,
                                            return_source_documents=True)
    
    output = qna_chain(query)
    response = output["result"]
    response = response[5:]

    return response

def t5_chain_last_mile(query="Who are the main users (participants) in the two-sided market?"):
    # Get the individual components
    model, tokenizer = load_t5_model()
    t5_pipe = build_t5(model, tokenizer)
    rt = retrieve_last_mile()

    # Create the chain
    qna_chain = RetrievalQA.from_chain_type(llm=t5_pipe,
                                            chain_type="stuff",
                                            retriever=rt,
                                            return_source_documents=True)
    
    output = qna_chain(query)
    response = output["result"]
    response = response[5:]

    return response


def t5_chain_new(query="Who are the main users (participants) in the two-sided market?"):
    # Get the individual components
    model, tokenizer = load_t5_model()
    t5_pipe = build_t5(model, tokenizer)
    rt = retrieve_new()
    # Create the chain
    qna_chain = RetrievalQA.from_chain_type(llm=t5_pipe,
                                            chain_type="stuff",
                                            retriever=rt,
                                            return_source_documents=True)
    
    output = qna_chain(query)
    response = output["result"]
    response = response[5:]
    source = output["source_documents"]

    return response, source

def t5_llmchainnew(query):
    model, tokenizer = load_t5_model()
    t5_pipe = build_t5(model, tokenizer)
    rt = retrieve_new()
    # Create the chain
    qna_chain = RetrievalQA.from_chain_type(llm=t5_pipe,
                                            chain_type="stuff",
                                            retriever=rt,
                                            return_source_documents=True)
    
    output = qna_chain(query)
    source = output["source_documents"]

    context = ""
    sources = []
    for s in source:
        context += s.page_content
        sources.append(s.metadata['source'])
    print("###########################################################")
    print("Question: " + str(query))
    print("###########################################################")
    print("The answer is derived from the following Context: " + context)
    print("###########################################################")
    print("The answer is derived from the following Context: " + str(sources))
    print("###########################################################")

    template = """Given the {context} please answer the following {query}. If you don't know the answer, just say that you don't know, don't try to make up an answer."""
    prompt = PromptTemplate(template=template, input_variables=["context", "query"])
    llm_chain = LLMChain(prompt=prompt, llm=t5_pipe)

    answer = llm_chain.predict(context=context, query=query)

    if len(answer) < 10:
        answer = "<pad> Cannot answer based on the given context" 
    else:
        answer = output["result"]


    return answer


def t5_llmchainlastmile(query):
    model, tokenizer = load_t5_model()
    t5_pipe = build_t5(model, tokenizer)
    rt = retrieve_last_mile()
    # Create the chain
    qna_chain = RetrievalQA.from_chain_type(llm=t5_pipe,
                                            chain_type="stuff",
                                            retriever=rt,
                                            return_source_documents=True)
    
    output = qna_chain(query)
    source = output["source_documents"]

    context = ""
    sources = []
    for s in source:
        context += s.page_content
        sources.append(s.metadata['source'])
    print("###########################################################")
    print("Question: " + str(query))
    print("###########################################################")
    print("The answer is derived from the following Context: " + context)
    print("###########################################################")
    print("The answer is derived from the following Context: " + str(sources))
    print("###########################################################")

    template = """Given the {context} please answer the following {query}. If you don't know the answer, just say that you don't know, don't try to make up an answer."""
    prompt = PromptTemplate(template=template, input_variables=["context", "query"])
    llm_chain = LLMChain(prompt=prompt, llm=t5_pipe)

    answer = llm_chain.predict(context=context, query=query)

    if len(answer) < 10:
        answer = "<pad> Cannot answer based on the given context" 
    else:
        answer = output["result"]


    return answer


def t5_llmchaintwosided(query):
    model, tokenizer = load_t5_model()
    t5_pipe = build_t5(model, tokenizer)
    rt = retrieve()
    # Create the chain
    qna_chain = RetrievalQA.from_chain_type(llm=t5_pipe,
                                            chain_type="stuff",
                                            retriever=rt,
                                            return_source_documents=True)
    
    output = qna_chain(query)
    source = output["source_documents"]

    context = ""
    sources = []
    for s in source:
        context += s.page_content
        sources.append(s.metadata['source'])
    print("###########################################################")
    print("Question: " + str(query))
    print("###########################################################")
    print("The answer is derived from the following Context: " + context)
    print("###########################################################")
    print("The answer is derived from the following Context: " + str(sources))
    print("###########################################################")

    template = """Given the {context} please answer the following {query}. If you don't know the answer, just say that you don't know, don't try to make up an answer."""
    prompt = PromptTemplate(template=template, input_variables=["context", "query"])
    llm_chain = LLMChain(prompt=prompt, llm=t5_pipe)

    answer = llm_chain.predict(context=context, query=query)

    if len(answer) < 10:
        answer = "<pad> Cannot answer based on the given context" 
    else:
        answer = output["result"]


    return answer




def incite_chain(query="Who are the main users (participants) in the two-sided market?"):
    # Get the individual components
    model, tokenizer = load_incite()
    # Build function similar to T5
    incite_pipe = build_pipeline(model, tokenizer)

    rt = retrieve()

    # Create the chain
    qna_chain = RetrievalQA.from_chain_type(llm=incite_pipe,
                                            chain_type="stuff",
                                            retriever=rt,
                                            return_source_documents=True)
    
    
    output = qna_chain(query)
    response = output["result"]
    lines = response.split("\n<bot>")
    #print(type(lines), lines)
    #print(len(lines), lines[0])
   
    return lines[0]


# Define output functions

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