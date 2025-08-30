from src.utils import load_documents, raw_to_chunks, huging_face_embeddings 
from langchain_pinecone.vectorstores import PineconeVectorStore
import os 

PINKONE_API_KEY = os.getenv("PINECONE_API_KEY")


extracted_data =load_documents("data/")
text_chunks = raw_to_chunks(extracted_data)
embeddings = huging_face_embeddings()

vec_db = PineconeVectorStore(
    pinecone_api_key=PINKONE_API_KEY,
    embedding=embeddings,  
    index_name="medicalbot",
)
vec_db.add_texts(texts=[text.page_content for text in text_chunks])


