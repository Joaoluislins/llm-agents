from datasets import Dataset
### Loading index
from usearch.index import Index
import numpy as np
import usearch

class BookDataIndex:
    def __init__(self, index_path: str = None, data_path: str = None, embeddings_path: str = None):
        """Class intended to loading the data, index and embeddings of search methods within RAG functions
        in the chatbot
        Args:
            index_path (str, optional): _description_. Defaults to None.
            data_path (str, optional): _description_. Defaults to None.
            embeddings_path (str, optional): _description_. Defaults to None.
        """
        
        self.data_path = data_path
        self.index_path = index_path
        self.embeddings_path = embeddings_path

    def load_index(self) -> usearch.index.Index:
        try:
            index = usearch.index.Index(ndim=128, metric='hamming', dtype="i8")
            index.load(self.index_path)
            return index
        except Exception as e:
            print(f"Error loading index: {e}")
            raise

    def load_data(self) -> Dataset:
        try:
            data = Dataset.load_from_disk(self.data_path)
            return data
        except Exception as e:
            print(f"Error loading data: {e}")
            raise

    def load_embeddings(self) -> np.ndarray:
        try:
            embeddings = np.load(self.embeddings_path)
            return embeddings
        except Exception as e:
            print(f"Error loading embeddings: {e}")
            raise
