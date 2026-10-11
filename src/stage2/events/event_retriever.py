"""Event Sub-Index Retrieval and Category Snapping to Commons Database."""

from __future__ import annotations
import math
import re
from typing import List, Dict, Any, Optional, Set, Tuple

from src.common.types import ScoredCandidate
from src.data.category_vocab import CategoryVocabulary


def tokenize(text: str) -> List[str]:
    """Tokenizes text into lowercase alphanumeric words, stripping Category prefix."""
    clean = text.replace("Category:", "").lower()
    return [w for w in re.findall(r"\b\w+\b", clean) if len(w) > 1]


class BM25Index:
    """Fast, in-memory BM25 index over category names for lexical snapping."""

    def __init__(self, categories: List[Tuple[int, str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = len(categories)
        self.doc_ids = [cat_id for cat_id, _ in categories]
        self.doc_names = [name for _, name in categories]

        self.doc_len: List[int] = []
        self.inverted_index: Dict[str, List[Tuple[int, int]]] = {}  # term -> [(doc_idx, tf)]
        self.df: Dict[str, int] = {}  # term -> doc_freq

        total_len = 0
        for idx, (_, name) in enumerate(categories):
            tokens = tokenize(name)
            t_len = len(tokens)
            self.doc_len.append(t_len)
            total_len += t_len

            counts: Dict[str, int] = {}
            for t in tokens:
                counts[t] = counts.get(t, 0) + 1

            for t, tf in counts.items():
                if t not in self.inverted_index:
                    self.inverted_index[t] = []
                    self.df[t] = 0
                self.inverted_index[t].append((idx, tf))
                self.df[t] += 1

        self.avg_doc_len = (total_len / self.corpus_size) if self.corpus_size > 0 else 1.0

        # Precompute IDF
        self.idf: Dict[str, float] = {}
        for term, freq in self.df.items():
            # Standard Lucene/BM25 IDF with smoothing
            self.idf[term] = math.log(1.0 + (self.corpus_size - freq + 0.5) / (freq + 0.5))

    def query(self, query_text: str, top_k: int = 20) -> List[Tuple[int, float]]:
        """Queries the BM25 index, returning list of (doc_idx, score)."""
        tokens = tokenize(query_text)
        if not tokens or self.corpus_size == 0:
            return []

        scores: Dict[int, float] = {}
        for t in tokens:
            if t not in self.inverted_index:
                continue
            idf_t = self.idf.get(t, 0.0)
            for doc_idx, tf in self.inverted_index[t]:
                dl = self.doc_len[doc_idx]
                numerator = tf * (self.k1 + 1.0)
                denominator = tf + self.k1 * (1.0 - self.b + self.b * (dl / self.avg_doc_len))
                term_score = idf_t * (numerator / denominator)
                scores[doc_idx] = scores.get(doc_idx, 0.0) + term_score

        if not scores:
            return []

        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return sorted_scores


class EventIndexRetriever:
    """Snaps event hypotheses to verified Wikimedia Commons categories."""

    def __init__(
        self,
        vocab: CategoryVocabulary,
        bm25_weight: float = 0.6,
        prototype_weight: float = 0.4,
        min_score_threshold: float = 0.20,
    ):
        self.vocab = vocab
        self.bm25_weight = bm25_weight
        self.prototype_weight = prototype_weight
        self.min_score_threshold = min_score_threshold

        # Build in-memory lexical index from vocabulary
        corpus = [(info.category_id, info.name) for info in self.vocab.id_to_cat.values()]
        self.bm25 = BM25Index(corpus)

    def retrieve_and_snap(
        self,
        hypotheses: List[str],
        top_k: int = 20,
    ) -> List[ScoredCandidate]:
        """Snaps generated hypothesis strings against verified database categories."""
        if not hypotheses or len(self.vocab) == 0:
            return []

        candidate_scores: Dict[int, float] = {}
        candidate_meta: Dict[int, Dict[str, Any]] = {}

        for hypothesis in hypotheses:
            clean_hyp = self.vocab.normalize_name(hypothesis)

            # 1. Exact match check
            exact_id = self.vocab.get_id(clean_hyp)
            if exact_id is not None:
                exact_score = 0.95
                if exact_id not in candidate_scores or exact_score > candidate_scores[exact_id]:
                    candidate_scores[exact_id] = exact_score
                    candidate_meta[exact_id] = {
                        "matched_hypothesis": hypothesis,
                        "match_type": "exact",
                        "lexical_score": 1.0,
                    }

            # 2. BM25 / token similarity query
            bm25_hits = self.bm25.query(hypothesis, top_k=10)
            if bm25_hits:
                max_bm25 = max(score for _, score in bm25_hits)
                for doc_idx, raw_score in bm25_hits:
                    cat_id = self.bm25.doc_ids[doc_idx]
                    norm_score = (raw_score / max_bm25) if max_bm25 > 0 else 0.0

                    # Jaccard bonus for token overlap
                    cand_name = self.bm25.doc_names[doc_idx]
                    hyp_tokens = set(tokenize(hypothesis))
                    cand_tokens = set(tokenize(cand_name))
                    jaccard = (
                        len(hyp_tokens & cand_tokens) / len(hyp_tokens | cand_tokens)
                        if (hyp_tokens or cand_tokens) else 0.0
                    )

                    combined_score = 0.7 * norm_score + 0.3 * jaccard
                    if combined_score >= self.min_score_threshold:
                        if cat_id not in candidate_scores or combined_score > candidate_scores[cat_id]:
                            candidate_scores[cat_id] = combined_score
                            candidate_meta[cat_id] = {
                                "matched_hypothesis": hypothesis,
                                "match_type": "lexical_bm25",
                                "bm25_score": float(norm_score),
                                "jaccard": float(jaccard),
                            }

        # Sort and select top_k
        sorted_cands = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        results: List[ScoredCandidate] = []
        for cat_id, score in sorted_cands:
            name = self.vocab.get_name(cat_id) or f"Category:{cat_id}"
            results.append(
                ScoredCandidate(
                    category_id=cat_id,
                    name=name,
                    score=float(score),
                    source="event_retrieval",
                    metadata=candidate_meta.get(cat_id, {}),
                )
            )

        return results
