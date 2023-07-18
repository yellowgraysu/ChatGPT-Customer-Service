import os
import numpy as np
from src.models import OpenAIModel
from hyperdb import HyperDB
from hyperdb.galaxy_brain_math_shit import (
    cosine_similarity,
    euclidean_metric,
    derridaean_similarity,
    hyper_SVM_ranking_algorithm_sort,
)

DATA_DIR = "./data/"
DOCUMENTS_NAME = "info.txt"
DB_FILE_NAME = "hyperdb.pickle.gz"


class CustomizeHyperDB(HyperDB):
    def __init__(self, similarity_metric="cosine"):
        self.documents = []
        self.vectors = None
        if similarity_metric.__contains__("cosine"):
            self.similarity_metric = cosine_similarity
        elif similarity_metric.__contains__("euclidean"):
            self.similarity_metric = euclidean_metric
        elif similarity_metric.__contains__("derrida"):
            self.similarity_metric = derridaean_similarity
        else:
            raise Exception(
                "Similarity metric not supported. Please use either 'cosine', 'euclidean' or 'derrida'."
            )

    def query(self, query_vector, top_k=5):
        ranked_results = hyper_SVM_ranking_algorithm_sort(
            self.vectors, query_vector, top_k=top_k, metric=self.similarity_metric
        )
        return [self.documents[index] for index in ranked_results]


def initHyperDB(db: CustomizeHyperDB, model: OpenAIModel):
    if DB_FILE_NAME in os.listdir(DATA_DIR):
        print("loading data")
        db.load(DATA_DIR + DB_FILE_NAME)
    else:
        if DOCUMENTS_NAME not in os.listdir(DATA_DIR):
            raise FileNotFoundError(f"{DOCUMENTS_NAME} not found")
        print("data preprocessing...")
        documents, vectors = data_preprocessing(DATA_DIR + DOCUMENTS_NAME, model)
        print("data preprocessed")
        db.add_documents(documents, vectors)
        print("saving data")
        db.save(DATA_DIR + DB_FILE_NAME)


def data_preprocessing(file_path, model: OpenAIModel):
    documents = []
    vectors = []
    with open(file_path, "r") as file:
        for line in file:
            documents.append(line)
            vectors.append(model.embedding(line))
    return documents, np.array(vectors)


# def data_preprocessing(csv_file, model: OpenAIModel):
#     df = pd.read_csv(csv_file)
#     documents = []
#     vectors = []
#     for idx, r in df.iterrows():
#         documents.append(
#             {"category": r.category, "subcategory": r.subcategory, "content": r.content}
#         )
#         vectors.append(model.embedding(f"{r.category}: {r.subcategory}\n{r.content}"))
#     return documents, np.array(vectors)


def getHyperDocuments(db: CustomizeHyperDB, model: OpenAIModel, query: str):
    query_vector = np.array(model.embedding(query))
    documents = db.query(query_vector, top_k=20)
    return documents
