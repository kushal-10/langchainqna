from langchain.chains import RetrievalQA, VectorDBQA
from langchain.document_loaders import TextLoader
from InstructorEmbedding import INSTRUCTOR
import textwrap

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.summarize import load_summarize_chain

from lc.model import load_t5_model, build_t5
from lc.database import GetRetriever

from lc.database import get_docs

class DefineChain():
    def __init__(self, data="new") -> None:
        self.data = data

    def llm_chain(self, query, k=3):
        model, tokenizer = load_t5_model()
        t5_pipe = build_t5(model, tokenizer)
        gr = GetRetriever(self.data)
        rt = gr.retrieve_data(k=k) 

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
        answer = output["result"]


        return answer
    
    def summarizer_chain(self):
        data = 'data_' + str(self.data)
        docs_dict = get_docs(data)

        # List of PDFs
        keys = list(docs_dict.keys())

        model, tokenizer = load_t5_model()
        t5_pipe = build_t5(model, tokenizer)

        # print(t5_pipe.get_num_tokens(doc[0].page_content))
        summarized_pdfs = {}
        for pdf in keys:
            doc = docs_dict[pdf]
            chain = load_summarize_chain(llm=t5_pipe, chain_type='map_reduce')
            summary = chain.run(doc)
            summarized_pdfs[pdf] = summary

        return summarized_pdfs


    def get_summary(self, query):
        model, tokenizer = load_t5_model()
        t5_pipe = build_t5(model, tokenizer)
        summary1 = """The  articles  in  the  list  cover  various  topics  related  to  operations  research,  
        including  the  impact  of  carbon  taxes  on  retail  location  decisions,  the  optimal  location  of  retailers  in  competitive  markets  
        and  monopoly  markets,  the  impact  of  a  carbon  price  on  retail  location  decisions,  the  competitive  facility  location  problem,  
        and  the  choice  of  green  technology.  The  articles  also  discuss  the  competitive  facility  location  problem  
        with  attractiveness  adjustment  of  the  follower,  the  impact  of  environmental  regulations  on  transportation  mode  selection,  
        and  the  impact  of  a  carbon  penalty  on  retail  location  competition.  The  articles  also  provide  a  review  of  the  literature  
        on  competitive  facility  location  problems,  environmental  taxes,  and  the  choice  of  green  technology."""
        paper1 = """ Dilek_etal2017.pdf """

        summary2 = """ The  articles  and  papers  listed  in  the  summary  discuss  the  relationship  between  retail  store  density  
        and  greenhouse  gas  emissions  in  a  retail  supply  chain.  They  explore  the  impact  of  small  local  shops  on  the  supply  chain  
        and  the  cost  of  greenhouse  gas  emissions.  The  articles  and  papers  also  consider  the  cost  of  carbon  emissions  for  retailers  
        and  consumers,  and  the  trade-off  between  store  density  and  truck  density.  They  propose  models  for  balancing  the  environmental 
          impact  of  a  car  fleet  with  the  cost  of  operating  it,  and  evaluate  penalty  bounds  for  each  scenario.  
          They  also  discuss  the  relationship  between  supply  chain  design  and  greenhouse  gas  emissions  and  use  a  tessellation  
          of  right  triangles  to  evaluate  the  optimal  design.  """
        paper2 = """ Cachon_2014.pdf """

        template = """Given two papers {paper1} and {paper2}, along with their respective summaries {summary1} and {summary2} please answer the following question {query}"""
        prompt = PromptTemplate(template=template, input_variables=["paper1", "paper2", "summary1", "summary2", "query"])
        llm_chain = LLMChain(prompt=prompt, llm=t5_pipe)


        answer = llm_chain.predict(paper1=paper1, paper2=paper2, summary1=summary1, summary2=summary2, query=query)

        return answer
