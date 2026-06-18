#!/usr/bin/env python3
"""
Offline JD artifact generator.

Run this ONCE before the competition to produce:
  artifacts/jd_embedding.npy   — 384-dim JD vector for Stage 3
  artifacts/model_cache/       — all-MiniLM-L6-v2 weights for offline use

The jd_config.json is already committed to artifacts/ and does not need
regeneration unless the role changes.

Usage:
  python generate_jd_artifacts.py

Requirements:
  pip install sentence-transformers numpy
  (network access allowed — offline artifact generation phase)
"""

import json
import sys
from pathlib import Path

ARTIFACTS = Path(__file__).parent / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# JD text (distilled from job_description.docx for optimal embedding)
# ─────────────────────────────────────────────────────────────────────────────
# This text is deliberately written to encode what the role MEANS,
# not just what the JD says — including disqualifiers as negative framing.

JD_EMBEDDING_TEXT = """
Senior AI Engineer founding team Redrob AI talent intelligence platform.
Pune Noida India hybrid. Five to nine years experience, ideal six to eight years.

Production experience required: embeddings-based retrieval systems deployed to real users.
Sentence transformers, BGE, E5, OpenAI embeddings. Handled embedding drift, index refresh,
retrieval quality regression in production. Real users, real traffic.

Production experience required: vector databases and hybrid search infrastructure.
Pinecone, Weaviate, Qdrant, Milvus, OpenSearch, Elasticsearch, FAISS.
Operational experience with search infrastructure at scale.

Strong Python. Hands-on evaluation frameworks for ranking systems.
NDCG, MRR, MAP, A/B testing, offline-to-online correlation.
Designed and ran experiments to improve retrieval quality.

Shipped at least one end-to-end ranking, search, or recommendation system
to real users at meaningful scale. Product company experience, not consulting.
Scrappy product-engineering attitude: ship working systems, learn from users.
Strong opinions on retrieval: hybrid versus dense search.
LLM integration: when to fine-tune versus prompt engineering.

Nice to have: LoRA QLoRA PEFT fine-tuning. Learning to rank XGBoost LambdaMART.
Distributed systems, large scale inference optimization. Open source contributions.

NOT a good fit: pure research without production deployment.
NOT a good fit: LangChain calling OpenAI only, no prior ML production experience.
NOT a good fit: entire career at TCS Infosys Wipro Accenture Cognizant Capgemini HCL services firms.
NOT a good fit: computer vision speech robotics without NLP information retrieval experience.
NOT a good fit: senior title but no production code written in eighteen months.
"""


def generate_embedding() -> bool:
    """Compute and save JD embedding. Returns True on success."""
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
    except ImportError:
        print("ERROR: Missing dependencies.")
        print("Run:  pip install sentence-transformers numpy")
        return False

    model_cache = ARTIFACTS / "model_cache"
    model_cache.mkdir(exist_ok=True)

    print("Loading all-MiniLM-L6-v2 ...")
    print("(~23 MB download on first run; cached to artifacts/model_cache/ after)")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Saving model to artifacts/model_cache/ for fully offline use...")
    model.save(str(model_cache))

    print("Computing JD embedding...")
    jd_emb = model.encode(JD_EMBEDDING_TEXT.strip(), normalize_embeddings=True)

    emb_path = ARTIFACTS / "jd_embedding.npy"
    np.save(emb_path, jd_emb)
    print(f"Saved: {emb_path}  (shape: {jd_emb.shape}, dtype: {jd_emb.dtype})")
    return True


def verify_config() -> bool:
    """Check jd_config.json exists and is valid JSON."""
    cfg_path = ARTIFACTS / "jd_config.json"
    if not cfg_path.exists():
        print(f"ERROR: {cfg_path} not found.")
        return False
    try:
        with open(cfg_path, encoding="utf-8") as f:
            cfg = json.load(f)
        n_must = len(cfg.get("must_have_skills", []))
        print(f"jd_config.json OK  ({n_must} must-have skills defined)")
        return True
    except json.JSONDecodeError as e:
        print(f"ERROR: jd_config.json is invalid JSON: {e}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("Redrob — JD Artifact Generator")
    print("=" * 60)

    if not verify_config():
        sys.exit(1)

    if generate_embedding():
        print()
        print("All artifacts ready. You can now run:")
        print()
        print("  python rank.py --candidates candidates.jsonl --out submission.csv")
        print()
    else:
        print()
        print("Embedding generation failed.")
        print("Stage 3 semantic re-ranking will be skipped during ranking.")
        print("Stage 1 + Stage 2 will still run and produce a valid submission.")
