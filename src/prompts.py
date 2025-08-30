#from langchain.embeddings import HuggingFaceEmbeddings

from langchain_pinecone.vectorstores import PineconeVectorStore
from src.utils import HuggingFaceEmbeddings

from langchain.chains import RetrievalQA

from langchain.prompts import PromptTemplate

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os


prompt="""
Use the following information to answer the user s question.
If you don't know the answer, just say that you don't know, don't try to create   an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and some information about it nothing else.
Helpful answer:
"""

