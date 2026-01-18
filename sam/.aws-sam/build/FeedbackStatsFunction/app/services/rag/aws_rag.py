from __future__ import annotations
from typing import Any, Dict, List, Optional
import json, re

from app.models import Evidence
from app.services.llm import get_llm
from app.services.rag.opensearch_store import OpenSearchHybridStore

class AwsAdvancedRAG:
    def __init__(self) -> None:
        self.llm = get_llm()
        self.store = OpenSearchHybridStore()

    def decompose_query(self, query: str) -> List[str]:
        system = "Return ONLY JSON."
        prompt = f"""Break this question into 2-4 search sub-queries. Return JSON array.

Question: {query}"""
        try:
            resp = self.llm.generate(prompt, system=system, temperature=0.1).text
            m = re.search(r"(\[[\s\S]*\])", resp)
            arr = json.loads(m.group(1) if m else resp)
            return [str(x).strip() for x in arr if str(x).strip()][:4] or [query]
        except Exception:
            return [query]

    def search(self, query: str, filters: Optional[Dict[str, Any]] = None) -> List[Evidence]:
        subs = self.decompose_query(query)
        gathered: Dict[str, Evidence] = {}
        for sq in subs:
            for e in self.store.search(sq, top_k=10, filters=filters):
                if e.chunk_id not in gathered or e.score > gathered[e.chunk_id].score:
                    gathered[e.chunk_id] = e
        out = list(gathered.values())
        out.sort(key=lambda e: e.score, reverse=True)
        return out[:10]
