"""
Minimal RAG pipeline over the Kestrel Data Labs HR policy pack.

Loader (pypdf) -> RecursiveCharacterTextSplitter -> Embeddings (OpenAI or open-source BGE)
-> FAISS vector store -> retriever (top-k) -> Gemini LLM with a citation-enforcing prompt (P1 or P2)
-> answer + citations. ConversationBufferMemory keeps the chat history so follow-up questions work.

The chain is written as a LangChain Runnable graph (LCEL):
    condense follow-up -> retrieve -> format context -> prompt -> LLM -> parse citations
"""

import os

# macOS: faiss-cpu and PyTorch (used by the open-source embedding model) each bundle an OpenMP runtime, and
# loading both aborts the process ("OMP: Error #15"). Allowing the duplicate must happen before either loads.
# It has no effect on Linux / Google Colab.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import re
import time
from pathlib import Path

from dotenv import load_dotenv
from pypdf import PdfReader

from langchain_core.documents import Document
from langchain_core.globals import set_llm_cache
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.cache import SQLiteCache
from langchain_classic.embeddings import CacheBackedEmbeddings
from langchain_classic.storage import LocalFileStore
from langchain_classic.memory import ConversationBufferMemory

load_dotenv()                                   # reads GOOGLE_API_KEY / OPENAI_API_KEY from .env

HERE = Path(__file__).parent
CORPUS_DIR = HERE / "corpus"

# DRY RUN (RAG_DRY_RUN=1): exercises the whole pipeline with NO API calls. The LLM is replaced by a
# simple extractive stand-in and OpenAI embeddings by the local BGE model. All files then go to dryrun/,
# so real results are never mixed with test output.
DRY_RUN = os.getenv("RAG_DRY_RUN") == "1"
WORK_DIR = HERE / "dryrun" if DRY_RUN else HERE
CACHE_DIR = WORK_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Fixed configuration (recorded in every log row for reproducibility)
# ---------------------------------------------------------------------------
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")   # 2.5-flash is retired for new API keys
OPENAI_EMBED_MODEL = "text-embedding-3-small"
OPEN_SOURCE_EMBED_MODEL = "BAAI/bge-small-en-v1.5"
TOP_K = 4
TEMPERATURE = 0.0
SEED = 42
OPENAI_EMBED_PRICE_PER_1M = 0.02        # USD per 1M tokens, text-embedding-3-small list price

# Every LLM response is cached on disk (keyed by model + exact prompt). Re-running the
# notebook therefore makes no new API calls and gives identical answers.
set_llm_cache(SQLiteCache(database_path=str(CACHE_DIR / "llm_cache.sqlite")))


# ---------------------------------------------------------------------------
# 1. Loading: one Document per PDF page, with title / source / page metadata
# ---------------------------------------------------------------------------
def load_corpus(corpus_dir=CORPUS_DIR):
    docs = []
    for pdf in sorted(Path(corpus_dir).glob("*.pdf")):
        reader = PdfReader(pdf)
        title = reader.metadata.title or pdf.stem
        for page_no, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            # drop the running footer line ("<title> - Kestrel ... - Page n") so it doesn't pollute chunks
            lines = [l for l in text.splitlines() if "Kestrel Data Labs (fictional) - Page" not in l]
            text = re.sub(r"[ \t]+", " ", "\n".join(lines)).strip()
            docs.append(Document(page_content=text, metadata={
                "doc_title": title, "source": str(pdf.relative_to(HERE)), "page": page_no}))
    return docs


# ---------------------------------------------------------------------------
# 2. Splitting: split page by page so every chunk belongs to exactly one page
# ---------------------------------------------------------------------------
def split_docs(pages, chunk_size, overlap_pct):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=int(chunk_size * overlap_pct),
        separators=["\n\n", "\n", ". ", " ", ""])
    chunks = []
    for page in pages:
        for i, piece in enumerate(splitter.split_text(page.page_content)):
            slug = Path(page.metadata["source"]).stem
            md = dict(page.metadata, chunk_id=f"{slug}-p{page.metadata['page']}-c{i}")
            chunks.append(Document(page_content=piece, metadata=md))
    return chunks


# ---------------------------------------------------------------------------
# 3. Embeddings (cached on disk, so OpenAI is only charged once per unique chunk)
# ---------------------------------------------------------------------------
_EMB = {}

def get_embeddings(name):
    if name in _EMB:
        return _EMB[name]
    if name == "openai" and DRY_RUN:            # stand-in: same local model, separate cache namespace
        from langchain_huggingface import HuggingFaceEmbeddings
        base = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        model_id = "dryrun-openai-standin"
    elif name == "openai":
        from langchain_openai import OpenAIEmbeddings
        base, model_id = OpenAIEmbeddings(model=OPENAI_EMBED_MODEL), OPENAI_EMBED_MODEL
    elif name == "bge":
        from langchain_huggingface import HuggingFaceEmbeddings
        base = HuggingFaceEmbeddings(model_name=OPEN_SOURCE_EMBED_MODEL,
                                     encode_kwargs={"normalize_embeddings": True})
        model_id = OPEN_SOURCE_EMBED_MODEL
    else:
        raise ValueError(name)
    store = LocalFileStore(str(CACHE_DIR / "embeddings"))
    _EMB[name] = CacheBackedEmbeddings.from_bytes_store(
        base, store, namespace=model_id.replace("/", "_"), query_embedding_cache=True)
    return _EMB[name]


def build_vectorstore(chunks, embedding_name):
    return FAISS.from_documents(chunks, get_embeddings(embedding_name))


# ---------------------------------------------------------------------------
# 4. LLM: Gemini, temperature 0, rate-limited for the free tier
# ---------------------------------------------------------------------------
_rate_limiter = InMemoryRateLimiter(requests_per_second=0.15, check_every_n_seconds=0.2, max_bucket_size=1)

def _dry_run_llm(prompt_value):
    """Extractive stand-in for the LLM (dry run only): copies the best-matching passage with its citation."""
    from langchain_core.messages import AIMessage
    msgs = prompt_value.to_messages()
    system, question = msgs[0].content, msgs[-1].content
    usage = {"input_tokens": len(system.split()) * 4 // 3, "output_tokens": 0, "total_tokens": 0}
    if "Context:" not in system:                     # condense step -> return the question unchanged
        return AIMessage(content=question, usage_metadata=usage)
    words = {w for w in re.findall(r"[a-z]{4,}", question.lower())}
    best, best_score = None, 0
    for block in system.split("Context:", 1)[1].strip().split("\n\n"):
        m = re.match(r"\[([^\]]+)\]", block)
        score = len(words & set(re.findall(r"[a-z]{4,}", block.lower())))
        if m and score > best_score:
            best, best_score = (m.group(1), block.split("\n", 1)[-1]), score
    if not best or best_score < 2:
        text = "I don't know based on the provided policies."
    else:
        sent = re.split(r"(?<=\.) ", best[1].replace("\n", " "))
        text = " ".join(sent[:2]).strip() + f" [{best[0]}]"
        if "Evidence:" in system:
            text = f"Evidence:\n- {sent[0].strip()} [{best[0]}]\nAnswer: {text}"
    usage["output_tokens"] = len(text.split()) * 4 // 3
    usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]
    return AIMessage(content=text, usage_metadata=usage)


def get_llm():
    if DRY_RUN:
        return RunnableLambda(_dry_run_llm)
    from langchain_google_genai import ChatGoogleGenerativeAI
    # timeout + retries: Gemini returns 503/504 when it is overloaded; the client backs off and retries
    kwargs = dict(model=GEMINI_MODEL, temperature=TEMPERATURE, seed=SEED, max_retries=6, timeout=120,
                  rate_limiter=_rate_limiter)
    if "2.5" in GEMINI_MODEL:
        kwargs["thinking_budget"] = 0      # 2.5 models: switch hidden reasoning off (cheaper, faster)
    # Gemini 3.x models always "think" a little first; those hidden reasoning tokens are reported separately
    # and are excluded from answer_tokens below, so the <=120-token conciseness rule measures the visible answer.
    return ChatGoogleGenerativeAI(**kwargs)


# ---------------------------------------------------------------------------
# 5. Prompt templates
# ---------------------------------------------------------------------------
SYSTEM_RULES = (
    "You are an HR policy assistant for Kestrel Data Labs. "
    "Use ONLY the context passages below; do not use outside knowledge. "
    "Every factual sentence must end with a citation in the exact form [Document Title#page], "
    "copied from the passage labels. "
    "If the context does not contain the answer, reply exactly: \"I don't know based on the provided policies.\" "
    "and add no citations."
)

P1 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_RULES + " Answer in at most 120 tokens (2-4 short sentences).\n\nContext:\n{context}"),
    MessagesPlaceholder("chat_history"),
    ("human", "{question}"),
])

P2 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_RULES +
     " Format your reply exactly as:\n"
     "Evidence:\n- <short quote or paraphrase from a passage> [Title#page]\n- <second supporting snippet> [Title#page]\n"
     "Answer: <final answer in at most 60 words, with citations>\n"
     "Use at most 2 evidence bullets. Do not show any other reasoning. "
     "If the answer is not in the context, reply only with the don't-know sentence.\n\nContext:\n{context}"),
    MessagesPlaceholder("chat_history"),
    ("human", "{question}"),
])

PROMPTS = {"P1": P1, "P2": P2}

# Rewrites a follow-up ("how many of those can I carry forward?") into a standalone question
CONDENSE = ChatPromptTemplate.from_messages([
    ("system", "Rewrite the user's latest question as a standalone question that can be understood without the "
               "conversation, resolving words like 'those', 'it' or 'that trip' from the history. If it is already "
               "standalone, return it unchanged. Return only the question."),
    MessagesPlaceholder("chat_history"),
    ("human", "{question}"),
])


# ---------------------------------------------------------------------------
# 6. The Runnable graph and a conversational session with memory
# ---------------------------------------------------------------------------
CITATION_RE = re.compile(r"\[([^\[\]#]+?)#(\d+)\]")

def format_context(docs):
    # each passage is labelled with the exact citation the model must use
    return "\n\n".join(f"[{d.metadata['doc_title']}#{d.metadata['page']}] (chunk {d.metadata['chunk_id']})\n"
                       f"{d.page_content}" for d in docs)


def build_chain(vectorstore, prompt_id):
    llm = get_llm()
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

    def condense(x):
        if not x["chat_history"]:
            return {"text": x["question"], "usage": {}}
        msg = (CONDENSE | llm).invoke(x)
        return {"text": msg.text.strip(), "usage": msg.usage_metadata or {}}

    def generate(x):
        msg = (PROMPTS[prompt_id] | llm).invoke({"context": format_context(x["docs"]),
                                                 "chat_history": x["chat_history"], "question": x["question"]})
        return msg

    return (
        RunnablePassthrough.assign(condensed=RunnableLambda(condense))
        | RunnablePassthrough.assign(docs=lambda x: retriever.invoke(x["condensed"]["text"]))
        | RunnablePassthrough.assign(llm_msg=RunnableLambda(generate))
    )


class RAGSession:
    """One conversation: the chain + a ConversationBufferMemory that is reset between triples."""

    def __init__(self, vectorstore, prompt_id):
        self.chain = build_chain(vectorstore, prompt_id)
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    def reset(self):
        self.memory.clear()

    def ask(self, question):
        t0 = time.perf_counter()
        history = self.memory.load_memory_variables({})["chat_history"]
        out = self.chain.invoke({"question": question, "chat_history": history})
        latency = time.perf_counter() - t0
        answer = out["llm_msg"].text.strip()
        self.memory.save_context({"input": question}, {"output": answer})

        usage = out["llm_msg"].usage_metadata or {}
        cu = out["condensed"]["usage"]
        citations = sorted({f"{t.strip()}#{p}" for t, p in CITATION_RE.findall(answer)})
        return {
            "answer": answer,
            "citations": citations,
            "standalone_question": out["condensed"]["text"],
            "used_chunks": [d.metadata["chunk_id"] for d in out["docs"]],
            "used_pages": sorted({f"{d.metadata['doc_title']}#{d.metadata['page']}" for d in out["docs"]}),
            "input_tokens": usage.get("input_tokens", 0) + cu.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0) + cu.get("output_tokens", 0),
            "answer_tokens": usage.get("output_tokens", 0) - (usage.get("output_token_details") or {}).get("reasoning", 0),
            "reasoning_tokens": (usage.get("output_token_details") or {}).get("reasoning", 0),
            "latency_s": round(latency, 2),
        }
