from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chat_models.gigachat import GigaChat
from langchain_core.prompts import ChatPromptTemplate

from config import Config
from embeddings import create_embeddings
from loaders import split_from_git


def ask_question(question: str):
    docs = db.similarity_search(question, k=4)

    print(len(docs))

    print(f"... {str(docs[0])} ...")

    return rag_chain.invoke({"input": question})


if __name__ == "__main__":
    llm = GigaChat(model=Config.CHAT_MODEL, temperature=Config.TEMPERATURE, credentials=Config.GIGA_API_TOKEN,
                   scope=Config.SCOPE,
                   verify_ssl_certs=False)

    system_prompt = """Ты ассистент с экспертизой в javascript, typescript и react. \n 
    Используй компоненты и свойства только из @v-uik. В ответах опирайся на предоставленную документацию и примеры.  
    Предоставляемый код должен корректно исполняться со всеми обязательными импортами и переменными. 
    Структурируй свой ответ описанием решения кода. \n
    Затем перечисли импорты и далее укажи работающий блок кода. \n
    {context}"""

    question = "Напиши с использованием react пример страницы с модальным окном"
    # answer0 = llm([HumanMessage(content=question)]).content
    # print(answer0)

    documents = split_from_git()

    # documents = split_text_document()

    #documents = split_markdown_documents("content/user-guide/md")

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
