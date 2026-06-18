"""
Stage 3: Semantic re-ranking using all-MiniLM-L6-v2.

Applied only to the top-K candidates from Stage 2 (default 500).
Encodes candidate text blobs and computes cosine similarity against
the pre-computed JD embedding.

Falls back gracefully to composite-score ordering if:
  - jd_embedding.npy not found (run generate_jd_artifacts.py)
  - sentence-transformers not installed
"""

from pathlib import Path
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def semantic_rerank(top_k: list, jd_config: dict, artifacts_dir: str = "artifacts") -> list:
    """
    Re-rank top_k candidates using semantic similarity to the JD embedding.
    Returns the top 100 candidates with final_score set.

    Args:
        top_k:         list of feature dicts from Stage 2
        jd_config:     loaded jd_config.json dict
        artifacts_dir: directory containing jd_embedding.npy and model_cache/

    Returns:
        list of 100 feature dicts, sorted by final_score descending
    """
    blend = jd_config.get("semantic_blend", {})
    composite_w = float(blend.get("composite_weight", 0.70))
    semantic_w  = float(blend.get("semantic_weight",  0.30))

    jd_emb_path  = Path(artifacts_dir) / "jd_embedding.npy"
    model_path   = Path(artifacts_dir) / "model_cache"

    semantic_scores = _compute_semantic_scores(top_k, jd_emb_path, model_path)

    for i, cand in enumerate(top_k):
        cand["semantic_score"] = float(semantic_scores[i])
        cand["final_score"] = (
            composite_w * cand["composite_score"]
            + semantic_w * float(semantic_scores[i])
        )

    top_k.sort(key=lambda x: (
        -x["final_score"],
        x["candidate_id"]          # tie-break: ascending candidate_id
    ))
    return top_k[:100]


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _compute_semantic_scores(candidates: list,
                              jd_emb_path: Path,
                              model_path: Path) -> np.ndarray:
    """
    Return a 1-D numpy array of cosine similarities (one per candidate).
    Falls back to zeros if dependencies or artifacts are unavailable.
    """
    n = len(candidates)

    # ── Dependency check ──────────────────────────────────────────────────────
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("  [Stage 3] WARNING: sentence-transformers not installed. "
              "Skipping semantic re-ranking (composite scores used as-is).")
        print("  Install: pip install sentence-transformers")
        return np.zeros(n, dtype=np.float32)

    # ── Load JD embedding ─────────────────────────────────────────────────────
    if not jd_emb_path.exists():
        print(f"  [Stage 3] WARNING: {jd_emb_path} not found. "
              "Run generate_jd_artifacts.py to enable semantic re-ranking.")
        return np.zeros(n, dtype=np.float32)

    jd_emb = np.load(jd_emb_path)

    # ── Load model ────────────────────────────────────────────────────────────
    if model_path.exists():
        print(f"  [Stage 3] Loading model from {model_path} (offline)...")
        model = SentenceTransformer(str(model_path))
    else:
        print("  [Stage 3] Loading all-MiniLM-L6-v2 (may require network)...")
        model = SentenceTransformer("all-MiniLM-L6-v2")

    # ── Encode candidates ─────────────────────────────────────────────────────
    texts = [c["candidate_text"] for c in candidates]
    print(f"  [Stage 3] Encoding {n} candidates in batches...")
    cand_embs = model.encode(
        texts,
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    # Both jd_emb and cand_embs are L2-normalized → dot product = cosine similarity
    sims = cand_embs @ jd_emb
    return sims.astype(np.float32)
