
from langchain.chains.llm import LLMChain
from langchain.chat_models.gigachat import GigaChat
from langchain_core.prompts import ChatPromptTemplate

from config import Config
from loaders import load_context_from_git


def ask_question(chain, db, question: str):
    relevants = db.similarity_search(question)
    doc = relevants[0].dict()['metadata']

    return chain.invoke(doc)


if __name__ == "__main__":
    llm = GigaChat(model=Config.CHAT_MODEL, temperature=Config.TEMPERATURE, credentials=Config.GIGA_API_TOKEN,
                   scope=Config.SCOPE,
                   verify_ssl_certs=False)

    context = load_context_from_git()

    # db = create_embeddings(documents)

    template = """You are a coding assistant with expertise in javascript, typescript and react. \n 
    Here is a full set of frontend components:  \n ------- \n  {context} \n ------- \n Answer the user 
    question based on the above provided documentation. Ensure any code you provide can be executed \n 
    with all required imports and variables defined. Structure your answer with a description of the code solution. \n
    Then list the imports. And finally list the functioning code block."""

    question = "Напиши с использованием react пример страницы с модальным окном"
    prompt = ChatPromptTemplate.from_messages([("system", template), ("human", "{input}")])

    # цепочка с кастомным промтом
    llm_chain = LLMChain(
        llm=llm,
        prompt=prompt)

    generated = llm_chain.invoke(input={"context": context, "input": question})
    print(generated["text"])
