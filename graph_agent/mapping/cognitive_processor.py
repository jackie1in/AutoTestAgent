from typing import List, Dict, Any
import numpy as np


class CognitiveProcessor:
    def __init__(self, api_key: str):
        self.api_key = api_key

    def compute_fingerprint(self, envelopes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Computes a semantic fingerprint for a list of semantic envelopes."""
        # 1. Serialize envelopes to text representation
        text_parts = []
        for env in envelopes:
            role = env.get("role", "")
            name = env.get("name", "")
            if role and name:
                text_parts.append(f"{role}:{name}")

        text_rep = " | ".join(sorted(text_parts))

        # 2. Get embedding (Mockable)
        vector = self._get_embedding(text_rep)

        return {"text_representation": text_rep, "vector": vector}

    def _get_embedding(self, text: str) -> List[float]:
        """Placeholder for OpenAI Embedding API call."""
        # In real implementation, this would call OpenAI
        # For now, return a dummy vector or raise NotImplementedError if not mocked
        return [0.0] * 1536

    def calculate_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculates cosine similarity between two vectors."""
        v1 = np.array(vec1)
        v2 = np.array(vec2)

        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(np.dot(v1, v2) / (norm1 * norm2))

    def is_similar(
        self, vec1: List[float], vec2: List[float], threshold: float = 0.95
    ) -> bool:
        """Determines if two vectors are similar based on a threshold."""
        return self.calculate_similarity(vec1, vec2) >= threshold
