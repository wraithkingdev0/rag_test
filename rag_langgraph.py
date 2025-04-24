from typing import TypedDict, List
from langgraph.graph import StateGraph, END

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class State(TypedDict):
    question: str
    docs: List[Document]
    context: str
    answer: str

def format_docs(docs: List[Document]) -> str:
    return "\n\n".join(f"[source={d.metadata.get('source','local')}] {d.page_content}" for d in docs)

def build_retriever():
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
    vectordb = Chroma.from_documents(chunks, embedding=OpenAIEmbeddings(), collection_name="mini_rag_graph")
    return vectordb.as_retriever(search_kwargs={"k": 3})

retriever = build_retriever()

prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful assistant. Answer ONLY using the provided context. "
     "If the answer is not in the context, say: 'I don't know based on the provided context.'"),
    ("human", "Question: {question}\n\nContext:\n{context}")
])
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def retrieve_node(state: State) -> State:
    q = state["question"]
    docs = retriever.invoke(q)
    return {**state, "docs": docs}

def build_context_node(state: State) -> State:
    ctx = format_docs(state["docs"]) if state["docs"] else ""
    return {**state, "context": ctx}

def generate_node(state: State) -> State:
    ctx = state["context"].strip()
    if not ctx:
        return {**state, "answer": "I don't know based on the provided context."}

    msg = prompt.invoke({"question": state["question"], "context": ctx})
    resp = llm.invoke(msg)
    return {**state, "answer": resp.content}

graph = StateGraph(State)
graph.add_node("retrieve", retrieve_node)
graph.add_node("context", build_context_node)
graph.add_node("generate", generate_node)

graph.set_entry_point("retrieve")
graph.add_edge("retrieve", "context")
graph.add_edge("context", "generate")
graph.add_edge("generate", END)

app = graph.compile()

def main():
    while True:
        q = input("\nAsk a question (or 'exit'): ").strip()
        if q.lower() in ("exit", "quit"):
            break
        out = app.invoke({"question": q, "docs": [], "context": "", "answer": ""})
        print("\nAnswer:\n", out["answer"])

if __name__ == "__main__":
    main()
