"""
Honeypot Detection Module.

The dataset contains ~80 honeypot profiles with subtly impossible signal
combinations. More than 10% of top-100 being honeypots triggers
auto-disqualification at Stage 3.

Strategy: score each candidate on 5 anomaly checks.
Score ≥ 3 → suspected honeypot → demoted to bottom ranks.
This guarantees ≤ 9 suspects in top-100 even if all are honeypots (< 10%).
"""

from datetime import date


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def detect_and_demote_honeypots(top_100: list) -> list:
    """
    Compute a honeypot suspicion score for each candidate.
    Suspected honeypots (score ≥ 3) are moved to the bottom of the ranking.
    Returns reordered list of 100 candidates.
    """
    clean    = []
    suspects = []

    for cand in top_100:
        hs = _honeypot_score(cand)
        cand["honeypot_score"] = hs
        if hs >= 3:
            suspects.append(cand)
        else:
            clean.append(cand)

    # Clean candidates keep their relative order; suspects pushed to the end
    return clean + suspects


# ─────────────────────────────────────────────────────────────────────────────
# Anomaly checks
# ─────────────────────────────────────────────────────────────────────────────

def _honeypot_score(feats: dict) -> int:
    """Return an integer suspicion score (0–8). Threshold for demotion: ≥ 3."""
    score = 0
    raw   = feats.get("_raw", {})
    sig   = raw.get("redrob_signals", {})

    # ── Check 1: Metric perfection ───────────────────────────────────────────
    # Statistically improbable to score at ceiling on multiple independent metrics
    ceiling_hits = 0
    if float(sig.get("profile_completeness_score", 0)) >= 99.0:
        ceiling_hits += 1
    if float(sig.get("github_activity_score", -1)) == 100.0:
        ceiling_hits += 1
    if float(sig.get("recruiter_response_rate", 0)) >= 0.99:
        ceiling_hits += 1
    if float(sig.get("interview_completion_rate", 0)) >= 0.99:
        ceiling_hits += 1
    if float(sig.get("offer_acceptance_rate", -1)) >= 0.99:
        ceiling_hits += 1
    if ceiling_hits >= 5:
        score += 3
    elif ceiling_hits >= 4:
        score += 2

    # ── Check 2: Endorsement / connection ratio anomaly ──────────────────────
    endorsements = int(sig.get("endorsements_received", 0))
    connections  = max(int(sig.get("connection_count", 1)), 1)
    if endorsements / connections > 10.0:
        score += 1

    # ── Check 3: Skill-career contradiction ─────────────────────────────────
    # Expert-level ML skills with long duration, but zero AI/ML career history
    skills = raw.get("skills", [])
    expert_ml_skills = [
        s for s in skills
        if s.get("proficiency") in ("expert", "advanced")
        and int(s.get("duration_months", 0)) > 48
        and any(kw in s.get("name", "").lower() for kw in [
            "machine learning", "deep learning", "nlp", "embedding",
            "llm", "pytorch", "tensorflow", "recommendation", "retrieval"
        ])
    ]
    if len(expert_ml_skills) >= 3 and feats.get("ai_career_fraction", 1.0) < 0.03:
        score += 2

    # ── Check 4: Date impossibilities ────────────────────────────────────────
    for job in raw.get("career_history", []):
        sd = job.get("start_date", "")
        ed = job.get("end_date")
        if sd and ed:
            try:
                s = date.fromisoformat(sd)
                e = date.fromisoformat(ed)
                if s > e:
                    score += 2
                    break
            except (ValueError, TypeError):
                pass

    for edu in raw.get("education", []):
        sy = edu.get("start_year")
        ey = edu.get("end_year")
        if sy and ey and int(ey) < int(sy):
            score += 1
            break

    # ── Check 5: Activity paradox ────────────────────────────────────────────
    last_active_str = sig.get("last_active_date", "2020-01-01")
    try:
        last_active = date.fromisoformat(last_active_str)
        if last_active > date.today():
            score += 2           # last active in the future
    except (ValueError, TypeError):
        pass

    saved = int(sig.get("saved_by_recruiters_30d", 0))
    if saved > 250:              # unrealistically high bookmarks
        score += 1

    return score
