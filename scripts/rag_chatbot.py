
#streamlit run rag_chatbot.py

import streamlit as st

import oracledb
import os
import uuid
from dotenv import load_dotenv

from langgraph.graph import MessagesState, StateGraph, START, END
from langchain_core.messages import HumanMessage

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_oracledb import OracleVS

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langgraph.checkpoint.memory import MemorySaver

print("✅ Successfully imported libaries and modules!")

# ==============
# Get OpenAI key
# ==============

load_dotenv("../config/.env")

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise RuntimeError("❌ OPENAI_API_KEY missing in .env")

# ===========================
# Establish Oracle connection
# ===========================

ur = os.getenv("USER")
pw = os.getenv("PASSWORD")
cs = os.getenv("CONNECT_STRING")

try:
    con26ai = oracledb.connect(user=ur, password=pw, dsn=cs)
    print("✅ Successfully connected to Oracle Database!")
    print (f"Database version: {con26ai.version}")
except Exception as e:
    print("Connection to Oracle Database failed!")

# ================
# Create Variables
# ================

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
EMBEDDING_DIMENSION = os.getenv("EMBEDDING_DIMENSION")
ORACLE_TABLE_NAME = os.getenv("ORACLE_TABLE_NAME")
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME")
print("Successfully created variables!")

# 1. State RAGState inheriting from MessagesState
class RAGState(MessagesState): # list of messages and... 
    context: str # ...context as a string (overwritten in each turn)

# 2. Setup embeddings + Oracle vector store + Retriever

# Initialize the embedding model
embeddings = OpenAIEmbeddings(
    model=EMBEDDING_MODEL,
    dimensions=EMBEDDING_DIMENSION,
    api_key=openai_api_key)
print("✅ OpenAI embeddings initialized")

# Initialize the OracleVS handle
vector_store = OracleVS(
    con26ai,
    embeddings,
    table_name=ORACLE_TABLE_NAME,
    distance_strategy=DistanceStrategy.COSINE,
    mutate_on_duplicate=True
)
print("✅ OracleVS initialized")

# Generate a retriever of type mmr to fetch relevant documents from the vector store in response to a user’s query.
# Maximal Marginal Relevance(MMR) is an informational retrieval algorithm designed to reduce redundancy in the retrieved results 
# while maintainig high relevance to the user's query.
# Its goal is to pick results of a query, which are not only relevant to the query but also different from each other, ie. unique and relevant response.
# MMR helps in RAG pipeline where we want our context window to contain diverese but relevant information.
# It is helpful in case where documents are semantically overlapping to each other.
retriever = vector_store.as_retriever(
    search_type="mmr", # It can be "similarity" (default), "mmr", or "similarity_score_threshold".
    search_kwargs={"k": 8, # Amount of documents to return (Default: 4)
                   "fetch_k": 40, # Amount of documents to pass to MMR algorithm (Default: 20)
                   "lambda_mult": 0.5}  # Diversity of results returned by MMR; 1 for minimum diversity and 0 for maximum. (Default: 0.5)
)

# 3. LLM
llm = init_chat_model(
    model=OPENAI_MODEL_NAME,
    temperature=0,
    model_provider="openai",
    streaming=True
)

# 4. Prompts
# In our system_prompt we have three placeholders or three keys: history, context, and question
system_prompt = ChatPromptTemplate.from_messages([
    # Behavior
    ("system",
     "You are a helpful professional financial analyst providing detailed, accurate answers."
     "Do not speculate or invent facts, rely only on the provided context including the conversation history."
     "If the context is insufficient, say you don't know as this is not covered in the Documents."),

    # Conversation history (kept short)
    # MessagesPlaceholder is a placeholder which can be used to pass a list of messages.
    # So we use here the variable name "history" and this is actually the key.
    # And then we can replace the placeholder with the key messages and pass here any list of messages.
    MessagesPlaceholder(variable_name="history", n_messages=20), # last 10 turns

    # RAG context (string you build from retrieved docs)
    ("system",
     "Context (may be empty). Use it if relevant; do not mention these headers in your answer:\n{context}"),

    # Latest user question
    MessagesPlaceholder(variable_name="question", n_messages=1)
])

# 5. chain the system_prompt defined above with the llm
chain = system_prompt | llm # retruns an AI message

# 6. Helper‑Function: Documents → Context‑String
# Concatenates the page content for all top docs.
def concat_docs(docs) -> str:
    return "\n\n---\n\n".join(d.page_content for d in docs)

# 7. Two nodes: retriever + chat
def retriever_node(state: RAGState):
    # In the retriever node we get all messages and we extract the query and actually only the string: content.
    msgs = state.get("messages", [])
    query = msgs[-1].content

    # We get the top_docs with the query from the retriever.
    # Then we get the context as a full string, passing top_docs to the helper function concat_docs
    # And finally we override context in our state.
    # context it's just a string overridden each time.
    top_docs = retriever.invoke(query)
    context = concat_docs(top_docs) 
    return {"context": context}   

def chat_node(state: RAGState):
    msgs = state.get("messages", []) # get all message (latest and history)
    history = msgs[:-1] # all messages except the last one are history messages.
    question = msgs[-1:] # the question message is actually the very last message in a list.
    context = state.get("context", "") # we also get the context from the state.

    # With these inputs we can invoke our chain passing history to history questions to question and context to context.
    # And then we get an AI message, which we concatenate to our state.
    ai = chain.invoke({"history": history, "question": question, "context" : context})
    return {"messages": [ai]}

# 8. Build LangGraph Workflow
# Two nodes : retriever node and chat node.
# One linear graph : From the START to the retriever node, retriever to chat node, and from the chat node to the END.
# checkpointer=MemorySaver() to store states in memory using a thread_id
workflow = StateGraph(RAGState)
workflow.add_node("retriever_node", retriever_node)
workflow.add_node("chat_node", chat_node)
workflow.add_edge(START, "retriever_node")
workflow.add_edge("retriever_node", "chat_node")
workflow.add_edge("chat_node", END)
graph = workflow.compile(checkpointer=MemorySaver())

# to display the graph
# display(Image(graph.get_graph().draw_mermaid_png()))

# 9. Interactive chat using streamlit
class RAGChatbot:
    def __init__(self):
        st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="centered")
        st.title("🤖 RAG Chatbot (OpenAI-powered)")
        st.divider()
        self.init_session()

    def init_session(self):
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

    def run(self):
        # Display chat history
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])                

        # Chat Input Handling
        if user_prompt := st.chat_input("Type a question (or 'quit, q or exit' to exit)"):
            st.session_state.chat_history.append({"role": "user", "content": user_prompt})

            if user_prompt.lower() in ['quit', 'exit', 'q']:
                response = "👋 Chat ended."
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                with st.chat_message("assistant"):
                    st.markdown(response)
                st.stop()

            with st.chat_message("user"):
                st.markdown(user_prompt)

            with st.chat_message("assistant"):
                response_container = st.empty()
                config = {"configurable": {"thread_id": str(uuid.uuid4())}}

                response = graph.invoke({"messages": [HumanMessage(content=user_prompt)]}, config=config)
                ai_response = response["messages"][-1].content

                response_container.markdown(ai_response)
                st.session_state.chat_history.append({"role": "assistant", "content": ai_response})

def main():
    app = RAGChatbot()
    app.run()

if __name__ == "__main__":
    main()
