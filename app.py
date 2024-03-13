import os
import streamlit as st
import pickle
import time
import unstructured
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain

from dotenv import load_dotenv

os.environ["OpenAi_API_Key"] = "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"

load_dotenv()

st.title("Make Research Easy....")
st.sidebar.title("News Article URLs")

urls = []
for i in range(3) :
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")

main_placeholder = st.empty()
llm = OpenAI(temperature=0.9 , max_tokens=500)


if process_url_clicked :
    #load url data
    url_loader = UnstructuredURLLoader(
        urls = urls 
    )
    main_placeholder.text("Data loading.... started....")
    url_data = url_loader.load()

    #split data
    text_slitter = RecursiveCharacterTextSplitter(
    separators=["\n\n","\n",".",","],
    chunk_size = 1000,
    chunk_overlap = 200
    )
    main_placeholder.text("Data splitting.... started....")
    chunks = text_slitter.split_documents(url_data)

    # create embeddings and save in as index

    embeddings = OpenAIEmbeddings()
    vector_index_openaii = FAISS.from_documents(chunks,embeddings)
    main_placeholder.text("Embedding vector.... started Building....")
    time.sleep(2)


    #save faiss index to local
    
    vector_index_openaii.save_local("vector_index_openaii")
    vectorstore = FAISS.load_local("vector_index_openaii",embeddings,allow_dangerous_deserialization=True)


query = main_placeholder.text_input("Question : ")

if query :
      
     embeddings = OpenAIEmbeddings()
     vectorstore = FAISS.load_local("vector_index_openaii",embeddings,allow_dangerous_deserialization=True)

     chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
    
     result = chain({"question": query} , return_only_outputs = True)
     st.header("Answer") 
     st.write(result["answer"])

     sources = result.get("sources","")

     if sources :
         st.subheader("Sources : ")
         sources_list = sources.split("\n")

         for source in sources_list :
             st.write(source)

                 

