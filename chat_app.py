import streamlit as st

from chunking import chunk_folder
from rag_pipeline import RAGPipeline


st.set_page_config(
    page_title="RAG Assistant",
    page_icon="🔎",
    layout="wide"
)


@st.cache_resource
def load_pipeline():

    chunks = chunk_folder(
        "data/documents",
        chunk_size=1000,
        overlap=150
    )

    return RAGPipeline(chunks)


st.title("🔎 RAG Assistant")
st.caption(
    "Hybrid retrieval using BM25 + FAISS Dense Retrieval + "
    "RRF + Cross-Encoder Reranking"
)


pipeline = load_pipeline()


if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


query = st.chat_input(
    "Ask a question about the documents..."
)


if query:

    st.session_state.messages.append({
        "role": "user",
        "content": query
    })

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):

        with st.spinner("Searching documents..."):

            result = pipeline.answer(
                query,
                strategy="hybrid_reranked"
            )

        st.markdown(result["answer"])

        with st.expander("📚 Retrieved Sources"):

            for chunk in result["retrieved_chunks"]:

                st.markdown(
                    f"**{chunk['chunk_id']}** — "
                    f"{chunk['source_doc']}"
                )

                st.write(chunk["text"])

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"]
    })