import os
import time
import traceback
from dataclasses import dataclass, field
from enum import Enum
from llama_index.core import Settings
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.base.llms.types import ChatMessage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.google_genai import GoogleGenAI
# from llama_index.llms.gemini import Gemini
from config import LOCAL_EMBEDDING_MODEL, MODEL, NEW_API_KEY
from google.api_core import exceptions

# ─────────────────────────────────────────────────────────────────────────────
# 1. PROMPTS
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """
You are a Technical Document Analyst specialising in lottery systems for skilrock Technologies.
Your job is to interpret text, images, diagrams, and screenshots from
the provided PCSO (Philippine Charity Sweepstakes Office) document.

When answering:
• State the type of visual (Architecture Diagram, Dashboard, Hardware Table …)
• Relate it to PCSO requirements (High Availability, Data Replication, Help Desk …)
• For diagrams — describe data flow between primary and mirror facilities.
• For dashboards — detail metrics: sales, player demographics, system alerts.
• For hardware tables — summarise server modules and scaling strategies.
• Conclude with what the visual confirms about the system's "Complied" status.

STRICT RULES
1. Only use the retrieved document context supplied to you.
2. If the answer is not in the context, return exactly: NOT_IN_DOCUMENT
3. Never invent technical specifications not shown in the document.
"""

RELEVANCE_PROMPT = """You are a strict relevance grader.
Given a question and a list of text chunks, identify which chunks are RELEVANT to answering the question.

Question: {question}

Chunks:
{chunks_list}

Reply ONLY with the ID numbers of the relevant chunks separated by commas (e.g., 0, 2, 4).
If none are relevant, reply with 'NONE'."""

REFLECTION_PROMPT = """You are a quality-control reviewer.
Given the question and the draft answer, decide:
  GOOD    – accurate, grounded in the document, and complete
  REFINE  – partially correct but needs more detail
  FAIL    – not grounded, incorrect, or the model could not find the information

Question: {question}
Answer:   {draft}
Verdict (reply with one word only):"""

DOMAIN_CHECK_PROMPT = """You are a strict domain classifier for a PCSO lottery system assistant.

Is the question related to ANY of these domains?
- Lottery systems, operations, or technology
- PCSO (Philippine Charity Sweepstakes Office)
- Government IT infrastructure or compliance
- Data replication, high availability, disaster recovery
- Help desk or IT operations for regulated systems
- Hardware / software architecture for enterprise systems
- Data analytics or reporting for regulated industries

Question: {question}

Reply with ONE word only:
  IN_DOMAIN     – related to the domains above
  OUT_OF_DOMAIN – unrelated"""


# ─────────────────────────────────────────────────────────────────────────────
# 2. TYPES
# ─────────────────────────────────────────────────────────────────────────────

class QueryType(str, Enum):
    OFF_TOPIC = "off_topic"
    VISUAL    = "visual"
    STANDARD  = "standard"


class AnswerSource(str, Enum):
    DOCUMENT        = "document"
    GEMINI_FALLBACK = "gemini_fallback"


@dataclass
class AgentState:
    query:           str
    query_type:      QueryType    = QueryType.STANDARD
    chunks:          list         = field(default_factory=list)
    relevant_chunks: list         = field(default_factory=list)
    draft_answer:    str          = ""
    final_answer:    str          = ""
    reflection:      str          = ""
    answer_source:   AnswerSource = AnswerSource.DOCUMENT
    iterations:      int          = 0
    max_iterations:  int          = 2


# ─────────────────────────────────────────────────────────────────────────────
# 3. QUERY ROUTER
# ─────────────────────────────────────────────────────────────────────────────

OFF_TOPIC_KEYWORDS = {
    "weather", "movie", "film", "actor", "iron man", "batman",
    "recipe", "cook", "doctor", "hospital", "bitcoin", "cricket",
}
VISUAL_KEYWORDS = {
    "diagram", "image", "screenshot", "chart", "figure",
    "architecture", "block diagram", "dashboard", "table",
}


def route_query(query: str) -> QueryType:
    lower = query.lower()
    if any(kw in lower for kw in OFF_TOPIC_KEYWORDS):
        return QueryType.OFF_TOPIC
    if any(kw in lower for kw in VISUAL_KEYWORDS):
        return QueryType.VISUAL
    return QueryType.STANDARD


# ─────────────────────────────────────────────────────────────────────────────
# 4. GEMINI DOMAIN-SPECIFIC FALLBACK
# ─────────────────────────────────────────────────────────────────────────────

GEMINI_FALLBACK_PROMPT = """You are a Technical Expert specialising in lottery and gaming systems,
IT infrastructure, and government compliance frameworks — specifically PCSO
(Philippine Charity Sweepstakes Office) standards.

The user's question could NOT be answered from the internal document.
Answer using your general knowledge, BUT follow these rules strictly:
1. Stay within lottery systems, PCSO operations, IT infrastructure, high availability,
   data replication, disaster recovery, help desk operations, or government compliance.
2. Prefix your answer with: [General Knowledge]
3. If the question is outside these domains, reply only with:
   "Invalid Question: This system only answers lottery and PCSO-related questions."
4. Keep the answer concise and technical.

Question: {question}
Answer:"""


class GeminiFallback:
    """
    Fires only when RAG cannot find the answer in the document.

    Flow:
      1. Domain gate  — Gemini classifies IN_DOMAIN / OUT_OF_DOMAIN (one cheap call).
      2. If in-domain — a second Gemini call answers with a strict PCSO/lottery prompt.
      3. Every answer is prefixed with [General Knowledge] for UI differentiation.

    No external API key needed — same Gemini instance used throughout.
    """

    def __init__(self, llm):
        self.llm = llm

    def _is_in_domain(self, query: str) -> bool:
        prompt  = DOMAIN_CHECK_PROMPT.format(question=query)
        verdict = str(self.llm.complete(prompt)).strip().upper()
        print(f"[GeminiFallback] Domain check → {verdict}")
        return "IN_DOMAIN" in verdict

    def ask(self, query: str) -> str:
        # ① Domain gate
        if not self._is_in_domain(query):
            return (
                "Invalid Question: This system only answers based on "
                "the provided lottery terms document."
            )

        # ② General-knowledge answer — domain-locked via prompt
        print("[GeminiFallback] 🌐 Calling Gemini (general knowledge fallback) …")
        try:
            prompt = GEMINI_FALLBACK_PROMPT.format(question=query)
            answer = str(self.llm.complete(prompt)).strip()
            # Ensure prefix is present even if the model forgets it
            if not answer.startswith("[General Knowledge]"):
                answer = f"[General Knowledge] {answer}"
            print(f"[GeminiFallback] ✅ {answer[:100]} …")
            return answer
        except Exception as exc:
            print(f"[GeminiFallback] ❌ {exc}")
            return (
                "The requested information is not available in the "
                "provided lottery terms document."
            )


# ─────────────────────────────────────────────────────────────────────────────
# 5. AGENTIC RAG PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

class RAGAgent:
    """
    Agentic loop:
      retrieve → grade → generate → reflect
        ↳ REFINE  → expand retrieval & retry (max 2×)
        ↳ FAIL    → OpenAI domain-specific fallback
    """

    def __init__(self, index, llm, fallback: GeminiFallback, retriever_top_k: int = 5):
        self.index      = index
        self.llm        = llm
        self.fallback   = fallback
        self.base_top_k = retriever_top_k
        self.retriever  = index.as_retriever(similarity_top_k=retriever_top_k)

    # ── Step 1: Retrieve ─────────────────────────────────────────────
    def retrieve(self, state: AgentState) -> AgentState:
        print(f"[RAGAgent] 📥 Retrieving for: '{state.query[:80]}'")
        state.chunks = self.retriever.retrieve(state.query)
        print(f"[RAGAgent] Retrieved {len(state.chunks)} chunk(s).")
        return state

    # ── Step 2: Batch-grade relevance ────────────────────────────────
    def grade_chunks(self, state: AgentState) -> AgentState:
        print("[RAGAgent] 🔍 Batch grading …")
        numbered = "".join(
            f"ID {i}: {node.node.text[:500]}\n\n"
            for i, node in enumerate(state.chunks)
        )
        prompt = RELEVANCE_PROMPT.format(question=state.query, chunks_list=numbered)
        try:
            resp = str(self.llm.complete(prompt)).strip().upper()
            print(f"[RAGAgent] Grader → {resp}")
            if "NONE" in resp:
                state.relevant_chunks = state.chunks
            else:
                ids = [int(x.strip()) for x in resp.split(",") if x.strip().isdigit()]
                state.relevant_chunks = [state.chunks[i] for i in ids if i < len(state.chunks)]
                if not state.relevant_chunks:
                    state.relevant_chunks = state.chunks
        except Exception as exc:
            print(f"[RAGAgent] Grading error: {exc}")
            state.relevant_chunks = state.chunks
        print(f"[RAGAgent] {len(state.relevant_chunks)} relevant chunk(s).")
        return state

    # ── Step 3: Generate ─────────────────────────────────────────────
    def generate(self, state: AgentState) -> AgentState:
        print("[RAGAgent] ✍️  Generating …")
        context    = "\n\n".join(n.node.text for n in state.relevant_chunks)
        style_hint = (
            "Pay special attention to diagrams, images, or visual elements "
            "referenced in the context."
            if state.query_type == QueryType.VISUAL else ""
        )
        prompt = (
            f"{SYSTEM_PROMPT}\n{style_hint}\n\n"
            f"--- DOCUMENT CONTEXT ---\n{context}\n------------------------\n\n"
            f"Question: {state.query}\n\nAnswer:"
        )
        state.draft_answer = str(self.llm.complete(prompt)).strip()
        print(f"[RAGAgent] Draft: {state.draft_answer[:120]} …")
        return state

    # ── Step 4: Self-reflect ─────────────────────────────────────────
    def reflect(self, state: AgentState) -> AgentState:
        print("[RAGAgent] 🪞 Reflecting …")
        # Fast-path: generator itself signalled it couldn't find the answer
        if "NOT_IN_DOCUMENT" in state.draft_answer.upper():
            state.reflection = "FAIL"
            print("[RAGAgent] Generator flagged NOT_IN_DOCUMENT → FAIL")
            return state

        prompt  = REFLECTION_PROMPT.format(question=state.query, draft=state.draft_answer)
        verdict = str(self.llm.complete(prompt)).strip().upper()
        state.reflection = (
            "GOOD"   if "GOOD"   in verdict else
            "REFINE" if "REFINE" in verdict else
            "FAIL"
        )
        print(f"[RAGAgent] Verdict → {state.reflection}")
        return state

    # ── Agentic loop ─────────────────────────────────────────────────
    def run(self, query: str, query_type: QueryType) -> tuple[str, AnswerSource]:
        state = AgentState(query=query, query_type=query_type)

        state = self.retrieve(state)
        if not state.chunks:
            print("[RAGAgent] No chunks → Gemini fallback")
            return self.fallback.ask(query), AnswerSource.GEMINI_FALLBACK

        state = self.grade_chunks(state)
        time.sleep(1.5)
        state = self.generate(state)
        time.sleep(1.5)
        state = self.reflect(state)

        # Refinement loop
        while state.reflection == "REFINE" and state.iterations < state.max_iterations:
            state.iterations += 1
            print(f"[RAGAgent] 🔄 Refine pass {state.iterations} …")
            self.retriever = self.index.as_retriever(
                similarity_top_k=self.base_top_k + state.iterations * 2
            )
            state = self.retrieve(state)
            state = self.grade_chunks(state)
            time.sleep(1.5)
            state = self.generate(state)
            time.sleep(1.5)
            state = self.reflect(state)

        # FAIL → Gemini general-knowledge fallback
        if state.reflection == "FAIL":
            print("[RAGAgent] FAIL → Gemini domain fallback …")
            return self.fallback.ask(query), AnswerSource.GEMINI_FALLBACK

        state.final_answer = state.draft_answer
        return state.final_answer, AnswerSource.DOCUMENT


# ─────────────────────────────────────────────────────────────────────────────
# 6. PUBLIC FACTORY
# ─────────────────────────────────────────────────────────────────────────────

def get_chat_engine(index):
    """Returns agentic_chat(user_query) -> str. Drop-in replacement."""
 
    print(f"[chat_logic] Loading Gemini: {MODEL}")
    google_api_key = os.getenv("GOOGLE_API_KEY") or NEW_API_KEY
    if not google_api_key:
        raise RuntimeError("Set GOOGLE_API_KEY in your environment before starting the chat.")
    os.environ["GOOGLE_API_KEY"] = google_api_key

    llm = GoogleGenAI(model=MODEL, temperature=0)
    Settings.llm         = llm
    Settings.embed_model = HuggingFaceEmbedding(model_name=LOCAL_EMBEDDING_MODEL)

    try:
        print(f"[chat_logic] Smoke-test: {str(llm.complete('Say OK'))[:30]}")
    except Exception as exc:
        print(f"[chat_logic] ❌ LLM unreachable: {exc}")

    memory   = ChatMemoryBuffer.from_defaults(token_limit=2000)
    fallback = GeminiFallback(llm=llm)
    agent    = RAGAgent(index=index, llm=llm, fallback=fallback, retriever_top_k=5)

    def agentic_chat(user_query: str) -> str:
        print(f"\n{'='*60}\n[USER] {user_query}\n{'='*60}")

        # ① Hard keyword gate
        query_type = route_query(user_query)
        if query_type == QueryType.OFF_TOPIC:
            return (
                "Invalid Question: This system only answers based on "
                "the provided lottery terms document."
            )

        # ② Enrich with recent memory
        stored = memory.get()
        history_text = (
            "\n".join(f"{m.role}: {m.content}" for m in stored[-4:])
            if stored else ""
        )
        enriched_query = (
            f"Conversation so far:\n{history_text}\n\nNew question: {user_query}"
            if history_text else user_query
        )

        # ③ Agentic pipeline
        try:
            answer, source = agent.run(enriched_query, query_type)
            tag = "🌐 Gemini fallback" if source == AnswerSource.GEMINI_FALLBACK else "📄 Document"
            print(f"[chat_logic] Source → {tag}")
        except Exception as exc:
            print(f"[chat_logic] ❌ {exc}")
            traceback.print_exc()
            answer = f"System error: {type(exc).__name__}: {str(exc)[:200]}"
        except exceptions.ResourceExhausted:
            return "I am currently at my usage limit. Please try again later or contact support."

        # ④ Persist turn
        memory.put(ChatMessage(role="user",      content=user_query))
        memory.put(ChatMessage(role="assistant", content=answer))

        return answer

    return agentic_chat