import os
from dataclasses import dataclass


@dataclass
class BrioConfig:
    groq_api_key: str = None
    api_key: str = None
    supabase_url: str = None
    supabase_key: str = None
    business_system_prompt: str = "You are BRIO, a friendly customer support assistant. Answer politely and accurately."
    memory_window: int = 10
    rag_chunk_size: int = 300
    rag_chunk_overlap: int = 50
    top_k_docs: int = 3
    min_confidence: float = 0.7

    def __post_init__(self):
        self.groq_api_key = os.getenv("GROQ_API_KEY", "")
        self.supabase_url = os.getenv("SUPABASE_URL", "")
        self.supabase_key = os.getenv("SUPABASE_KEY", "")
        self.business_system_prompt = os.getenv("BUSINESS_SYSTEM_PROMPT", self.business_system_prompt)
        self.memory_window = int(os.getenv("BRIO_MEMORY_WINDOW", self.memory_window))
        self.rag_chunk_size = int(os.getenv("BRIO_RAG_CHUNK_SIZE", self.rag_chunk_size))
        self.rag_chunk_overlap = int(os.getenv("BRIO_RAG_CHUNK_OVERLAP", self.rag_chunk_overlap))
        self.top_k_docs = int(os.getenv("BRIO_TOP_K_DOCS", self.top_k_docs))
        self.min_confidence = float(os.getenv("BRIO_MIN_CONFIDENCE", self.min_confidence))
        self.api_key = os.getenv("API_KEY", self.api_key)

    def validate(self):
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY is required in environment")
        if not self.api_key:
            raise ValueError("API_KEY is required in environment")
