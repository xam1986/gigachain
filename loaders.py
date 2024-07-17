import os

from langchain.document_loaders import TextLoader
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
)
from langchain_community.document_loaders import UnstructuredMarkdownLoader, GitLoader

from config import Config


def split_text_document(file_path: str):
    loader = TextLoader(file_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=200,
    )
    documents = text_splitter.split_documents(documents)
    print(f"Total documents: {len(documents)}")
    return documents


def split_markdown_documents(directory_path: str):
    documents = []
    for filename in os.listdir(directory_path):
        filepath = os.path.join(directory_path, filename)

        # Check if it's a file and not a subdirectory
        if os.path.isfile(filepath) and filepath.endswith(".md"):
            loader = UnstructuredMarkdownLoader(filepath)
            documents.append(loader.load()[0])

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=200,
    )
    documents = text_splitter.split_documents(documents)
    print(f"Total documents: {len(documents)}")
    return documents


def extract_text_from_file(path: str):
    with open(path, 'r') as file:
        text = file.read()

    return text


def load_git_repository():
    repo_path = f"./repository/{Config.REPOSITORY.split('/')[-1]}"

    loader = GitLoader(
        clone_url=Config.REPOSITORY,
        repo_path=repo_path,
        branch="develop",
        file_filter=lambda file_path: file_path.endswith(Config.FILE_EXT)
    )
    data = loader.load()

    return data


def load_context_from_git():
    data = load_git_repository()
    filtered_data = list(filter(
        lambda it: it.metadata['file_path'].startswith("packages/modal") or it.metadata['file_path'].startswith(
            "packages/dropdown"), data))

    d_sorted = sorted(filtered_data, key=lambda x: x.metadata["source"])
    d_reversed = list(reversed(d_sorted))
    concatenated_content = "\n\n\n --- \n\n\n".join(
        [doc.page_content for doc in d_reversed]
    )

    return concatenated_content


def split_from_git():
    data = load_git_repository()

    filtered_data = list(filter(
        lambda it: it.metadata['file_path'].startswith("packages/modal") or it.metadata['file_path'].startswith(
            "packages/dropdown"), data))

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
                                                   chunk_overlap=200, )

    documents = text_splitter.split_documents(filtered_data)
    print(f"Total documents: {len(documents)}")
    return documents
