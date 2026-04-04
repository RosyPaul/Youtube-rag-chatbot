import re
import os
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()
load_dotenv()
# os.environ["HUGGINGFACE_HUB_TOKEN"] = os.getenv("HUGGINGFACE_HUB_TOKEN", "")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "")
# -------------------------
# Extract Video ID
# -------------------------
def extract_video_id(url: str):
    pattern = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
    match = re.search(pattern, url)
    return match.group(1) if match else None

# -------------------------
# Load Transcript
# -------------------------
def load_transcript(url: str):
    video_id = extract_video_id(url)
    if not video_id:
        raise ValueError(f"Invalid YouTube URL: {url}")

    try:
        fetched_transcript = YouTubeTranscriptApi().fetch(video_id, languages=['en'])
        transcript_list = fetched_transcript.to_raw_data()

        if not transcript_list:
            fetched_transcript = YouTubeTranscriptApi().fetch(video_id)
            transcript_list = fetched_transcript.to_raw_data()

        if not transcript_list:
            raise ValueError("Transcript is empty.")

    except (TranscriptsDisabled, NoTranscriptFound):
        raise ValueError(f"No captions available for video: {video_id}")
    except Exception as e:
        raise ValueError(f"Unexpected error fetching transcript: {e}")

    documents = [
        Document(
            page_content=chunk["text"],
            metadata={"timestamp": chunk["start"], "video_id": video_id, "url": url}
        )
        for chunk in transcript_list
    ]
    return documents

# -------------------------
# Create / Load FAISS for a single URL
# -------------------------
def get_vector_store_for_url(url: str, embeddings):
    video_id = extract_video_id(url)
    index_path = f"/tmp/faiss_index_{video_id}"

    if os.path.exists(index_path):
        return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)

    docs = load_transcript(url)
    if not docs:
        raise ValueError(f"No transcript documents found for: {url}")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=400)
    chunks = splitter.split_documents(docs)
    if not chunks:
        raise ValueError(f"Failed to split transcript for: {url}")

    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(index_path)
    return vector_store

# -------------------------
# Multi-URL: merge all vector stores
# -------------------------
def get_merged_vector_store(urls: list):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    merged_store = None
    errors = []

    for url in urls:
        url = url.strip()
        if not url:
            continue
        try:
            store = get_vector_store_for_url(url, embeddings)
            if merged_store is None:
                merged_store = store
            else:
                merged_store.merge_from(store)
        except Exception as e:
            errors.append(f"⚠️ Skipped '{url}': {e}")

    if merged_store is None:
        raise ValueError("No valid videos could be processed.\n" + "\n".join(errors))

    return merged_store, errors

# -------------------------
# Build RAG Chain with Chat History
# -------------------------
def build_qa_chain(vector_store, k=10):

    llm = ChatGroq(
        model_name="llama-3.1-8b-instant",
        temperature=0.5,
        api_key=os.getenv("GROQ_API_KEY")
    )

    retriever = vector_store.as_retriever(search_kwargs={"k": k})

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant that answers questions about YouTube videos based on their transcripts.

You will be given transcript excerpts from the video as context. Use them to answer the user's question as best as you can.

- If the user asks to summarize the video, summarize based on all the context provided.
- If the user asks a specific question, answer it using relevant parts of the context.
- If the context is insufficient, say what you can infer and mention that only partial transcript is available.
- Never say you "cannot" answer if there is any relevant information in the context.

Context from transcript:
{context}"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])

    def format_docs(docs):
        if not docs:
            return "No context available in the video."
        return "\n\n".join(
            f"[Video: {doc.metadata.get('video_id', 'unknown')} | t={doc.metadata.get('timestamp', '?')}s]\n{doc.page_content}"
            for doc in docs
        )

    rag_chain = (
        {
            "context": (lambda x: x["question"]) | retriever | format_docs,
            "question": lambda x: x["question"],
            "chat_history": lambda x: x.get("chat_history", []),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain

