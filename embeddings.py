from chromadb.config import Settings
from langchain.vectorstores import Chroma
from langchain_community.embeddings.gigachat import GigaChatEmbeddings

from config import Config


def create_embeddings(documents):
    embeddings = GigaChatEmbeddings(credentials=Config.GIGA_API_TOKEN, scope=Config.SCOPE, verify_ssl_certs=False
                                    )

    db = Chroma.from_documents(
        documents,
        embeddings,
        client_settings=Settings(anonymized_telemetry=False),
    )
    return db
