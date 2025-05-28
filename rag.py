from langchain_google_vertexai.chat_models import ChatVertexAI
from langchain_google_vertexai.embeddings import VertexAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain import hub
from langchain_core.documents import Document
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict

from utils.documents import get_documents


def get_graph():
    """Get answer to a question using RAG."""
    llm = ChatVertexAI(
        model_name="gemini-2.5-flash-preview-05-20",
        location="us-central1",
        temperature=0.0,
    )
    embeddings = VertexAIEmbeddings(
        model_name="text-embedding-005",
        location="us-central1",
    )

    vector_store = InMemoryVectorStore(embeddings)

    # 1 - Create splits
    docs = get_documents()
    # print(f"=== LOADED {len(docs)} DOCUMENTS ===", end="\n\n")
    # for doc in docs:
    #     print(doc.metadata["source"])

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits = text_splitter.split_documents(docs)

    _ = vector_store.add_documents(documents=all_splits)

    # 2 - Define RAG with LangGraph
    prompt = hub.pull("rlm/rag-prompt")


    # Define state for application
    class State(TypedDict):
        question: str
        context: List[Document]
        answer: str


    # Define application steps
    def retrieve(state: State):
        retrieved_docs = vector_store.similarity_search(state["question"])
        # print("=== DOCUMENTS RETRIEVED ===", end="\n\n")
        # for doc in retrieved_docs:
        #     print(doc.page_content[:250].strip() + "...", end="\n\n")
        return {"context": retrieved_docs}


    def generate(state: State):
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        messages = prompt.invoke({"question": state["question"], "context": docs_content})
        response = llm.invoke(messages)
        return {"answer": response.content}


    # Compile application and test
    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    graph = graph_builder.compile()
    return graph

if __name__ == "__main__":
    graph = get_graph()
    response = graph.invoke(
        {"question": "What does Melexis do in terms of sustainability?"},
    )
    print("=== FINAL ANSWER ===", end="\n\n")
    print(response["answer"].strip())
