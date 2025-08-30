from flask import Flask, request, jsonify, render_template

from langchain_community.vectorstores import Pinecone
from dotenv import load_dotenv
from src.prompts import prompt
from src.utils import *

from langchain_pinecone.vectorstores import PineconeVectorStore
import os
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

app=Flask(__name__)
load_dotenv()
extracted_data =load_documents("data/")
text_chunks = raw_to_chunks(extracted_data)
PINKONE_API_KEY = os.getenv("PINECONE_API_KEY")
#OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
embedings=huging_face_embeddings()
OPENAI_API_KEY="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" 
# add your openai key here
# Initialize the Flask application
search = PineconeVectorStore.from_existing_index(
    embedding=embedings,  
    index_name="medicalbot",
)
search.add_texts(texts=[text.page_content for text in text_chunks])
PROMPT=PromptTemplate(template=prompt, input_variables=["context", "question"])
chain_type_kwargs={"prompt": PROMPT}

llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7, max_tokens=1000,openai_api_key=OPENAI_API_KEY)

qa=RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=search.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True, 
    chain_type_kwargs=chain_type_kwargs)
# Define the Flask routes
@app.route('/')
def index():
    return render_template("ui.html")

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    
    print(msg)
    result=qa({"query": msg})
    print("Response : ", result["result"])
    return str(result["result"])

if __name__=="__main__":
    app.run()