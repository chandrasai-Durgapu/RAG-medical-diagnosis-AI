# import os
# import time
# import asyncio
# from pathlib import Path
# from dotenv import load_dotenv
# from pinecone import Pinecone, ServerlessSpec
# from langchain_community.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# # from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from ..config.database import reports_collection
# from typing import List
# from fastapi import UploadFile

# #Hugging face Embeddings
# from langchain_community.embeddings import HuggingFaceEmbeddings

# # from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.text_splitter import RecursiveCharacterTextSplitter


# load_dotenv()

# #GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# PINECONE_ENV = os.getenv("PINECONE_ENV", "us-east-1")
# PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "rbac-diagnosis-index")
# UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./uploaded_reports")

# #os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
# os.makedirs(UPLOAD_DIR, exist_ok=True)

# # initialize pinecone
# pc=Pinecone(api_key=PINECONE_API_KEY)
# spec=ServerlessSpec(cloud="aws",region=PINECONE_ENV)
# existing_indexes=[i["name"] for i in pc.list_indexes()]

# #dimension=768,

# if PINECONE_INDEX_NAME not in existing_indexes:
#     pc.create_index(name=PINECONE_INDEX_NAME,dimension=384,metric="dotproduct",spec=spec)
#     while not pc.describe_index(PINECONE_INDEX_NAME).status["ready"]:
#         time.sleep(1)

# index=pc.Index(PINECONE_INDEX_NAME)


# async def load_vectorstore(uploaded_files:List[UploadFile],uploaded:str,doc_id:str):
#     """
#         Save files, chunk texts, embed texts, upsert in Pinecone and write metadata to Mongo
#     """

#     #embed_model=GoogleGenerativeAIEmbeddings(model="models/embedding-001")

#     # 1. Choose your embedding model
#     embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
#     #to check the length of embeding 
#     #embedding = embedding_model.embed_query("test")
#     # the length is 384
#     print(len(embedding_model.embed_query("test")))

#     for file in uploaded_files:
#         filename=Path(file.filename).name
#         save_path=Path(UPLOAD_DIR)/ f"{doc_id}_{filename}"
#         content=await file.read()
#         with open(save_path,"wb") as f:
#             f.write(content)

#     # load pdf pages
#     loader=PyPDFLoader(str(save_path))
#     documents=loader.load()
#     # splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=100)#chunk_overlap=100)
#     # chunks= splitter.split_documents(documents)
#     splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
#     chunks = splitter.split_text(documents)

#     embeddings = [embedding_model.embed_query(chunk) for chunk in chunks]

#     #texts=[chunk.page_content for chunk in chunks]
#     texts=chunks
#     ids=[f"{doc_id}-{i}" for i in range(len(chunks))]
#     metadatas=[
#             {
#                 "source": filename,
#                 "doc_id": doc_id,
#                 "uploader": uploaded,
#                 "page": chunk.metadata.get("page", None),
#                 "text": chunk.page_content[:2000]  # store snippet in metadata (avoid huge fields)
#             }
#             for chunk in chunks
#     ]

#     # get embeddings in thread
#     embeddings=await asyncio.to_thread(embedding_model.embed_documents, texts) #embed_model.embed_documents,texts)
#     # upsert - run in thread to avoid blocking
#     def upsert():
#         index.upsert(vectors=list(zip(ids,embeddings,metadatas)))


#     await asyncio.to_thread(upsert)

#     # save report  metadata in mongo 
#     reports_collection.insert_one({
#                 "doc_id": doc_id,
#                 "filename":filename,
#                 "uploader": uploaded,
#                 "num_chunks":len(chunks),
#                 "uploaded_at":time.time()
                
#     })
import os
import time
import asyncio
from pathlib import Path
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from ..config.database import reports_collection
from typing import List
from fastapi import UploadFile
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV", "us-east-1")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "rbac-diagnosis-index")
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./uploaded_reports")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)
spec = ServerlessSpec(cloud="aws", region=PINECONE_ENV)
existing_indexes = [i["name"] for i in pc.list_indexes()]

# Delete the index if it exists, then create a fresh one with correct dimension
# if PINECONE_INDEX_NAME in existing_indexes:
#     logger.info(f"Deleting existing index '{PINECONE_INDEX_NAME}'...")
#     pc.delete_index(PINECONE_INDEX_NAME)
#     # Wait a few seconds to ensure deletion completes
#     time.sleep(5)

# logger.info(f"Creating index '{PINECONE_INDEX_NAME}' with dimension=384...")
# pc.create_index(name=PINECONE_INDEX_NAME, dimension=384, metric="dotproduct", spec=spec)
if PINECONE_INDEX_NAME not in existing_indexes:
    pc.create_index(name=PINECONE_INDEX_NAME,dimension=384,metric="dotproduct",spec=spec)
    while not pc.describe_index(PINECONE_INDEX_NAME).status["ready"]:
        logger.info("Waiting for index to be ready...")
        time.sleep(1)

index = pc.Index(PINECONE_INDEX_NAME)


async def load_vectorstore(uploaded_files: List[UploadFile], uploaded: str, doc_id: str):
    """
    Save uploaded files, split text, embed chunks using HuggingFace,
    upsert vectors to Pinecone, and save metadata to MongoDB.
    """
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    all_chunks = []
    all_metadatas = []
    all_ids = []

    try:
        for file in uploaded_files:
            filename = Path(file.filename).name
            save_path = Path(UPLOAD_DIR) / f"{doc_id}_{filename}"

            # Save uploaded file locally
            content = await file.read()
            with open(save_path, "wb") as f:
                f.write(content)
            logger.info(f"Saved file {filename} to {save_path}")

            # Load PDF documents
            loader = PyPDFLoader(str(save_path))
            documents = loader.load()

            # Split documents into chunks with metadata
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            chunks = splitter.split_documents(documents)

            start_idx = len(all_chunks)
            all_chunks.extend(chunks)

            # Prepare metadata for each chunk
            all_metadatas.extend([
                {
                    "source": filename,
                    "doc_id": doc_id,
                    "uploader": uploaded,
                    "page": chunk.metadata.get("page", None),
                    "text": chunk.page_content[:2000],  # snippet for metadata
                }
                for chunk in chunks
            ])

            all_ids.extend([f"{doc_id}-{start_idx + i}" for i in range(len(chunks))])

        # Extract texts for embedding
        texts = [chunk.page_content for chunk in all_chunks]

        logger.info(f"Computing embeddings for {len(texts)} chunks...")
        embeddings = await asyncio.to_thread(embedding_model.embed_documents, texts)

        # Upsert to Pinecone
        def upsert():
            logger.info(f"Upserting {len(embeddings)} vectors to Pinecone index '{PINECONE_INDEX_NAME}'...")
            index.upsert(vectors=list(zip(all_ids, embeddings, all_metadatas)))

        await asyncio.to_thread(upsert)

        # Save report metadata to MongoDB
        reports_collection.insert_one({
            "doc_id": doc_id,
            "filenames": [Path(f.filename).name for f in uploaded_files],
            "uploader": uploaded,
            "num_chunks": len(all_chunks),
            "uploaded_at": time.time()
        })

        logger.info(f"Finished processing document {doc_id}.")

    except Exception as e:
        logger.error(f"Error in load_vectorstore: {e}")
        raise

