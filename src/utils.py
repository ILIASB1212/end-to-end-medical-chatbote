from langchain.document_loaders import PyPDFLoader, DirectoryLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.embeddings import HuggingFaceEmbeddings


def load_documents(pdf):
    loader=DirectoryLoader(pdf, glob="*.pdf", loader_cls=PyPDFLoader)
    document=loader.load()
    return document

def raw_to_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)
    chunks=text_splitter.split_documents(text)
    return chunks

def huging_face_embeddings():
    embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings
