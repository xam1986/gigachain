from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chat_models.gigachat import GigaChat
from langchain_core.prompts import ChatPromptTemplate

from config import Config
from embeddings import create_embeddings
from loaders import split_markdown_documents


def ask_question(question: str):
    docs = db.similarity_search(question, k=4)

    print(len(docs))

    print(f"... {str(docs[0])} ...")

    return rag_chain.invoke({"input": question})


if __name__ == "__main__":
    llm = GigaChat(model=Config.CHAT_MODEL, temperature=Config.TEMPERATURE, credentials=Config.GIGA_API_TOKEN,
                   scope=Config.SCOPE,
                   verify_ssl_certs=False)

    system_prompt = """Ты эксперт в области конфигурирования yaml файлов.
    В ответе сначала укажи пример yaml файла с описанием параметров и пример запроса к сервису.
    При построении yaml используй документацию. Провалидируй yaml согласно схеме . \n
    {context}"""

    question = "Опиши пример простого прокси сервиса на порту 8091, проксирующего на ya.ru."
    # answer0 = llm([HumanMessage(content=question)]).content
    # print(answer0)

    # documents = split_from_git()

    # documents = split_text_document()

    documents = split_markdown_documents("content/developer-guide")

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )
    db = create_embeddings(documents)

    retriever = db.as_retriever(k=4)
    qa_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, qa_chain)
    # qa_chain = RetrievalQA.from_chain_type(llm_chain, chain_type="stuff", retriever=)

    answer1 = ask_question(question)

    print(answer1)
