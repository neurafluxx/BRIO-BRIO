import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

# Point HuggingFace cache to /tmp so it is writable on Railway's read-only filesystem.
# Must be set before importing sentence-transformers / transformers.
os.environ.setdefault("HF_HOME", "/tmp/hf_cache")
os.environ.setdefault("TRANSFORMERS_CACHE", "/tmp/hf_cache")
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", "/tmp/st_cache")

from langchain_core.messages import HumanMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq.chat_models import ChatGroq

from src.config import BrioConfig

INTENT_LABELS = [
    "BOOKING_REQUEST",
    "PRICE_INQUIRY",
    "COMPLAINT",
    "GENERAL_INQUIRY",
    "ESCALATION_NEEDED",
]


class BrioAgent:
    def __init__(self, config: BrioConfig):
        self.config = config
        try:
            self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        except Exception as exc:
            raise RuntimeError(
                "Failed to load embedding model 'all-MiniLM-L6-v2'. "
                "Ensure the container has internet access and enough disk/tmp space. "
                f"Original error: {exc}"
            ) from exc
        self.llm = ChatGroq(
            api_key=config.groq_api_key,
            model="llama-3.3-70b-versatile",
            temperature=0.2,
        )
        self.rag_documents = self._load_rag_documents()
        self.document_embeddings = self._embed_documents(
            [doc["page_content"] for doc in self.rag_documents]
        )
        # Use a writable temp dir for conversation history (Railway filesystem is read-only for src/)
        self.conversation_history_path = Path(
            os.environ.get("CONVERSATION_HISTORY_PATH", tempfile.gettempdir())
        ) / "conversation_history.json"
        self.conversation_memory: List[Dict[str, str]] = []

    def _load_rag_documents(self) -> List[Dict[str, Any]]:
        data_path = Path(__file__).resolve().parent / "brio-data" / "brio_rag_knowledge_base.json"
        with data_path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        documents: List[Dict[str, Any]] = []
        for business in data.get("businesses", []):
            page_content = self._flatten_business_data(business)
            documents.append(
                {
                    "page_content": page_content,
                    "metadata": {
                        "business_id": business.get("business_id"),
                        "type": business.get("type"),
                        "name": business.get("name"),
                    },
                }
            )

        return documents

    def _flatten_business_data(self, business: Dict[str, Any]) -> str:
        lines: List[str] = []
        lines.append(f"Business: {business.get('name', 'Unknown')}")
        lines.append(f"Tagline: {business.get('tagline', '')}")
        lines.append(f"Personality: {business.get('ai_persona', '')}")
        lines.append(f"Language preference: {business.get('language_preference', '')}")

        services = business.get("services", [])
        if services:
            lines.append("Services:")
            for service in services:
                lines.append(
                    f"- {service.get('name', '')}: {service.get('description', '')} {service.get('note', '')}".strip()
                )

        menu_highlights = business.get("menu_highlights", [])
        if menu_highlights:
            lines.append("Menu highlights:")
            for item in menu_highlights:
                if isinstance(item, dict):
                    lines.append(
                        f"- {item.get('item', '')} ({item.get('price', '')}): {item.get('description', '')}".strip()
                    )
                else:
                    lines.append(f"- {item}")

        hours = business.get("hours", {})
        if hours:
            lines.append("Hours:")
            if isinstance(hours, dict):
                for key, value in hours.items():
                    lines.append(f"- {key.replace('_', ' ').title()}: {value}")
            else:
                lines.append(f"- {hours}")

        faqs = business.get("faq", []) or business.get("faqs", [])
        if faqs:
            lines.append("FAQ:")
            for item in faqs:
                q = item.get("question") or item.get("q", "")
                a = item.get("answer") or item.get("a", "")
                lines.append(f"- Q: {q}")
                lines.append(f"  A: {a}")

        policies = business.get("policies", {})
        if policies:
            lines.append("Policies:")
            if isinstance(policies, dict):
                for key, value in policies.items():
                    lines.append(f"- {key.replace('_', ' ').title()}: {value}")
            elif isinstance(policies, list):
                for policy in policies:
                    if isinstance(policy, dict):
                        lines.append(f"- {policy.get('title', '')}: {policy.get('description', '')}")
                    else:
                        lines.append(f"- {policy}")

        return "\n".join(lines)

    def _embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        return self.embeddings.embed_documents(texts)

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def _detect_language(self, text: str) -> str:
        if re.search(r"[\u0600-\u06FF]", text):
            return "urdu"
        lower_text = text.lower()
        urdu_tokens = [
            "ji", "aap", "kya", "hai", "ka", "ke", "ko", "nahi", "haan",
            "aur", "kaise", "kab", "kis", "mujhe", "tum", "aapka", "theek",
            "shukriya", "mein",
        ]
        for token in urdu_tokens:
            if re.search(rf"\b{re.escape(token)}\b", lower_text):
                return "urdu"
        return "english"

    def save_memory(self, user_message: str, assistant_response: str, metadata: Optional[Dict[str, Any]] = None):
        row = {
            "message": user_message,
            "response": assistant_response,
            "metadata": metadata or {},
        }

        if self.conversation_history_path.exists():
            with self.conversation_history_path.open("r", encoding="utf-8") as history_file:
                try:
                    history = json.load(history_file)
                except ValueError:
                    history = []
        else:
            history = []

        history.append(row)
        try:
            with self.conversation_history_path.open("w", encoding="utf-8") as history_file:
                json.dump(history, history_file, indent=2)
        except OSError:
            # On read-only filesystems just skip persistence; in-memory is still kept
            pass

    def add_documents(self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None):
        if metadatas is None:
            metadatas = [{} for _ in texts]

        for text, metadata in zip(texts, metadatas):
            page_content = text.strip()
            if not page_content:
                continue
            self.rag_documents.append({"page_content": page_content, "metadata": metadata})
            self.document_embeddings.extend(self._embed_documents([page_content]))

    def retrieve_knowledge(self, query: str) -> List[Dict[str, Any]]:
        if not self.rag_documents:
            return []

        query_embedding = self.embeddings.embed_query(query)
        scored_docs = [
            (self._cosine_similarity(query_embedding, doc_emb), doc)
            for doc_emb, doc in zip(self.document_embeddings, self.rag_documents)
        ]
        scored_docs.sort(key=lambda item: item[0], reverse=True)
        return [doc for score, doc in scored_docs[: self.config.top_k_docs] if score > 0]

    def classify_intent(self, message: str) -> Dict[str, Any]:
        prompt = (
            "Classify the following customer message into one of the labels: "
            + ", ".join(INTENT_LABELS)
            + ".\nReturn ONLY a raw JSON object (no markdown, no code fences) with keys: intent, confidence."
            + f"\nMessage: {message}"
        )
        classification = self.llm.invoke([HumanMessage(content=prompt)])
        raw = classification.content if hasattr(classification, "content") else str(classification)
        # Strip markdown code fences if the model wraps its JSON
        raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
        raw = re.sub(r"\s*```$", "", raw.strip())
        try:
            parsed = json.loads(raw)
        except (ValueError, TypeError):
            parsed = {"intent": "GENERAL_INQUIRY", "confidence": 0.5}
        intent = parsed.get("intent", "GENERAL_INQUIRY")
        confidence = float(parsed.get("confidence", 0.0))
        if intent not in INTENT_LABELS:
            intent = "GENERAL_INQUIRY"
        return {"intent": intent, "confidence": confidence}

    def build_rag_context(self, question: str) -> str:
        docs = self.retrieve_knowledge(question)
        if not docs:
            return ""
        context = "\n\n".join([f"Source snippet:\n{doc['page_content']}" for doc in docs])
        return f"Use these knowledge snippets to answer the customer question:\n{context}\n"

    def respond(self, customer_message: str) -> Dict[str, Any]:
        intent_data = self.classify_intent(customer_message)
        rag_context = self.build_rag_context(customer_message)

        # Build conversation history for context
        memory_context = ""
        if self.conversation_memory:
            recent_history = self.conversation_memory[-self.config.memory_window :]
            memory_lines = []
            for item in recent_history:
                memory_lines.append(f"Customer: {item['customer_message']}")
                memory_lines.append(f"BRIO: {item['assistant_response']}")
            memory_context = "Recent conversation history:\n" + "\n".join(memory_lines) + "\n\n"

        system_prompt = self.config.business_system_prompt
        language_instruction = (
            "If the customer is speaking in English, reply fully in English. "
            "If the customer is speaking in Urdu or Roman Urdu, reply fully in Urdu. "
            "Do not mix languages within the same response."
        )
        user_prompt = (
            f"{system_prompt}\n\n"
            f"{language_instruction}\n\n"
            f"{memory_context}"
            f"{rag_context}\n"
            f"Customer: {customer_message}\n"
            "Respond clearly, concisely, and politely."
        )

        messages = [HumanMessage(content=user_prompt)]
        response = self.llm.invoke(messages)
        answer = response.content if hasattr(response, "content") else str(response)

        should_escalate = (
            intent_data["intent"] == "ESCALATION_NEEDED"
            or intent_data["confidence"] < self.config.min_confidence
        )

        self.conversation_memory.append({
            "customer_message": customer_message,
            "assistant_response": answer,
        })
        self.save_memory(customer_message, answer, metadata={"intent": intent_data["intent"]})

        return {
            "answer": answer,
            "intent": intent_data["intent"],
            "confidence": intent_data["confidence"],
            "escalation_required": should_escalate,
            "knowledge_used": rag_context.strip() != "",
        }
