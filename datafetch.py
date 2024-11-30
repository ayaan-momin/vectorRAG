import os
import logging
from typing import List, Optional
from dotenv import load_dotenv
from langchain_community.document_loaders import  TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

# Constants
DOCS_DIR = "./docs/"
DB_DIR = "./vector_db/"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import document loaders and download NLTK data
try:
    import nltk
    nltk.download('punkt', quiet=True)
    from langchain_community.document_loaders import UnstructuredWordDocumentLoader
    WORD_SUPPORT = True
except ImportError as e:
    logger.warning(f"Error importing Word document support: {e}")
    logger.warning("Word document support is disabled.")
    WORD_SUPPORT = False

try:
    from langchain_community.document_loaders import PyPDFLoader
    PDF_SUPPORT = True
except ImportError:
    logger.warning("pypdf not installed. PDF support is disabled.")
    PDF_SUPPORT = False

class DocumentProcessor:
    def __init__(self, docs_dir: str, db_dir: str):
        self.docs_dir = docs_dir
        self.db_dir = db_dir
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    def load_documents(self) -> List[str]:
        if not os.path.exists(self.docs_dir):
            raise FileNotFoundError(f"Directory '{self.docs_dir}' does not exist.")
        
        loaders = {
            ".txt": (TextLoader, {"encoding": "utf8"}),
        }
        
        if WORD_SUPPORT:
            loaders[".docx"] = (UnstructuredWordDocumentLoader, {})
            loaders[".doc"] = (UnstructuredWordDocumentLoader, {})
        
        if PDF_SUPPORT:
            loaders[".pdf"] = (PyPDFLoader, {})
        
        documents = []
        for file in os.listdir(self.docs_dir):
            file_path = os.path.join(self.docs_dir, file)
            file_extension = os.path.splitext(file)[1].lower()
            
            if file_extension in loaders:
                loader_class, loader_args = loaders[file_extension]
                try:
                    loader = loader_class(file_path, **loader_args)
                    documents.extend(loader.load())
                    logger.info(f"Successfully loaded {file}")
                except Exception as e:
                    logger.error(f"Error loading {file}: {str(e)}")
            else:
                logger.warning(f"Unsupported file type: {file}")
        
        if not documents:
            raise ValueError(f"No supported files found or loaded in '{self.docs_dir}'.")
        
        logger.info(f"Loaded {len(documents)} documents.")
        return [doc.page_content for doc in documents]

    def split_documents(self, documents: List[str]) -> List[str]:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
        splits = text_splitter.split_text("\n\n".join(documents))
        logger.info(f"Created {len(splits)} splits.")
        return splits

    def create_vector_store(self, splits: List[str]) -> Optional[Chroma]:
        if not splits:
            logger.error("No text splits to create vector store.")
            return None
        
        vector_store = Chroma.from_texts(
            texts=splits,
            embedding=self.embeddings,
            persist_directory=self.db_dir
        )
        logger.info(f"Vector store created with {vector_store._collection.count()} embeddings.")
        logger.info(f"Vector store persisted to: {self.db_dir}")
        return vector_store

    def process(self):
        try:
            logger.info("Loading documents...")
            documents = self.load_documents()

            logger.info("Splitting documents...")
            splits = self.split_documents(documents)

            logger.info("Creating vector store...")
            vector_store = self.create_vector_store(splits)
            if not vector_store:
                logger.error("Failed to create vector store.")
        except Exception as e:
            logger.exception(f"An error occurred: {str(e)}")

def main():
    processor = DocumentProcessor(DOCS_DIR, DB_DIR)
    processor.process()

if __name__ == "__main__":
    main()