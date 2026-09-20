import os
from dotenv import load_dotenv
from langchain_unstructured import UnstructuredLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore


load_dotenv()

if __name__ == '__main__':
    print("Ingesting...")
    loader = UnstructuredLoader(file_path="/root/langchain-course/langchain-course/mediumblog1.txt", chunking_strategy="basic", max_characters=1000000)
    document = loader.load()

    print("Splitting document...")
    text_splitter = CharacterTextSplitter(chunk_size = 1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(f"created {len(texts)} chunks from the document")
    
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        output_dimensionality=768,
    )
    # embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
    print("Ingesting...")
    PineconeVectorStore.from_documents(texts, embeddings, index_name=os.environ.get("INDEX_NAME"))
    print("Done!")
    