from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def format_docs(docs):
    return "\n\n".join(f"[source={d.metadata.get('source','local')}] {d.page_content}" for d in docs)

def build_retriever():
    # Minimal sample knowledge base
    docs = [
        Document(page_content="Our refund policy: refunds are allowed within 30 days of purchase with receipt.",
                 metadata={"source": "policy.md"}),
        Document(page_content="Shipping times: standard shipping is 5-7 business days within the US.",
                 metadata={"source": "shipping.md"}),
        Document(page_content="Support hours: Monday to Friday, 9 AM to 6 PM EST.",
                 metadata={"source": "support.md"}),
    ]

    splitter = RecursiveCharacterTextSplitter(chunk_size=350, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings()
    vectordb = Chroma.from_documents(chunks, embedding=embeddings, collection_name="mini_rag")
    return vectordb.as_retriever(search_kwargs={"k": 3})

def main():
    retriever = build_retriever()

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful assistant. Answer ONLY using the provided context. "
         "If the answer is not in the context, say: 'I don't know based on the provided context.'"),
        ("human",
         "Question: {question}\n\nContext:\n{context}")
    ])

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
    )

    while True:
        q = input("\nAsk a question (or 'exit'): ").strip()
        if q.lower() in ("exit", "quit"):
            break
        res = rag_chain.invoke(q)
        print("\nAnswer:\n", res.content)

if __name__ == "__main__":
    main()
