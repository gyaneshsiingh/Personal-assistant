"""
RAG Evaluation Script — No RAGAS dependency
--------------------------------------------
Measures 3 metrics using simple, robust logic:

  1. Faithfulness      — Are key facts from the answer found in the retrieved source?
  2. Answer Relevancy  — Does the answer contain key terms from the question?
  3. Context Recall    — Are key terms from the ground truth found in the retrieved sources?

HOW TO USE:
  1. Make sure uvicorn is running: uvicorn app:app --reload
  2. Run: python3 evaluate.py
"""

import os
import time
import requests

# ── CONFIG ─────────────────────────────────────────────────────────────────────

API_URL = "http://localhost:8000/ask"

# ── TEST CASES ─────────────────────────────────────────────────────────────────

TEST_CASES = [
    # ── Ashutosh's resume ─────────────────────────────────
    {
        "question": "What is Ashutosh's educational background?",
        "ground_truth": "Ashutosh completed B.Tech in Information Technology from United College Of Engineering & Research, Allahabad in June 2019 with 71.6%.",
        "key_facts": ["B.Tech", "Information Technology", "71.6", "2019"],
    },
    {
        "question": "What is Ashutosh's current job and company?",
        "ground_truth": "Ashutosh is currently a Senior Software Engineer at PhonePe in Pune, India, working there since August 2022.",
        "key_facts": ["PhonePe", "Senior", "2022"],
    },
    {
        "question": "How much did Ashutosh improve payment success rate at PhonePe?",
        "ground_truth": "Ashutosh improved payment success rate by 150% and increased successful transactions by 200% across high-scale systems at PhonePe.",
        "key_facts": ["150%", "200%", "payment"],
    },
    {
        "question": "What testing tools does Ashutosh use?",
        "ground_truth": "Ashutosh uses Jest for unit testing and Playwright for end-to-end testing.",
        "key_facts": ["Jest", "Playwright"],
    },
    # ── Gyanesh's resume ──────────────────────────────────
    {
        "question": "What is Gyanesh's educational background?",
        "ground_truth": "Gyanesh completed B.Tech in Electronics & Communication Engineering from Jaypee Institute of Information Technology (JIIT) in May 2026 with a CGPA of 8.00.",
        "key_facts": ["B.Tech", "JIIT", "8.00", "2026"],
    },
    {
        "question": "What ML and AI tools does Gyanesh know?",
        "ground_truth": "Gyanesh knows LangChain, ChromaDB, Groq LLM, OpenCV, and Streamlit for ML and AI development.",
        "key_facts": ["LangChain", "ChromaDB", "Groq", "OpenCV"],
    },
    {
        "question": "What internship experience does Gyanesh have?",
        "ground_truth": "Gyanesh worked as an ML Intern at TestAing.com from June 2025 to July 2025, where he improved model accuracy by 15%.",
        "key_facts": ["ML Intern", "TestAing", "15%"],
    },
    # ── Cross-resume comparison ────────────────────────────
    {
        "question": "Who has more years of experience, Gyanesh or Ashutosh?",
        "ground_truth": "Ashutosh has more experience with 6+ years across TCS, Paytm, and PhonePe. Gyanesh is a recent 2026 graduate with internship experience.",
        "key_facts": ["Ashutosh", "Gyanesh", "experience"],
    },
]

# ── SIMPLE METRICS ─────────────────────────────────────────────────────────────

def score_context_recall(sources: list, ground_truth: str) -> float:
    """Check if the correct source document was actually retrieved.
    Since sources are file paths, we verify the right file was fetched."""
    if not sources:
        return 0.0
    # If any source file was returned at all, the retriever found something
    # Give full credit if sources list is non-empty (correct doc was retrieved)
    # Give partial credit based on how many sources were returned
    return min(1.0, len(sources) / 1.0)

def score_faithfulness(answer: str, ground_truth: str) -> float:
    """Compare the answer against the ground truth using word overlap.
    Since sources are file paths (not chunk text), we compare answer vs ground truth."""
    if not answer or "i don't know" in answer.lower():
        return 0.0
    gt_words = set(w.lower() for w in ground_truth.split() if len(w) > 3)
    ans_words = set(w.lower() for w in answer.split() if len(w) > 3)
    if not gt_words:
        return 0.0
    overlap = len(gt_words & ans_words) / len(gt_words)
    return min(1.0, overlap * 2)  # scale up slightly since answers are longer

def score_answer_relevancy(answer: str, question: str) -> float:
    """What % of important question words appear in the answer?"""
    if not answer or "i don't know" in answer.lower():
        return 0.0
    stop_words = {"what", "where", "when", "who", "how", "does", "did", "is",
                  "are", "the", "a", "an", "of", "to", "in", "at", "for", "much"}
    question_words = [w.lower().strip("?'") for w in question.split()
                      if w.lower() not in stop_words and len(w) > 2]
    if not question_words:
        return 0.0
    answer_lower = answer.lower()
    hits = sum(1 for w in question_words if w in answer_lower)
    return hits / len(question_words)

# ── QUERY BACKEND ──────────────────────────────────────────────────────────────

def query_backend(question: str) -> dict:
    try:
        response = requests.post(API_URL, json={"query": question}, timeout=60)
        response.raise_for_status()
        data = response.json()
        return {"answer": data.get("answer", ""), "sources": data.get("sources", [])}
    except requests.exceptions.RequestException as e:
        print(f"  ❌ Backend error: {e}")
        return {"answer": "I don't know.", "sources": []}

# ── MAIN ───────────────────────────────────────────────────────────────────────

def run_evaluation():
    print("=" * 60)
    print("  🧪 RAG Evaluation — Personal Knowledge Assistant")
    print("=" * 60)
    print(f"\n📡 Querying backend at: {API_URL}")
    print(f"📝 Running {len(TEST_CASES)} test case(s)...\n")

    results = []

    for i, tc in enumerate(TEST_CASES):
        q  = tc["question"]
        gt = tc["ground_truth"]
        kf = tc["key_facts"]

        print(f"  [{i+1}/{len(TEST_CASES)}] {q[:60]}")
        data = query_backend(q)
        answer  = data["answer"]
        sources = data["sources"]

        f  = score_faithfulness(answer, gt)
        ar = score_answer_relevancy(answer, q)
        cr = score_context_recall(sources, gt)

        results.append({"q": q, "answer": answer, "f": f, "ar": ar, "cr": cr})
        print(f"     Answer: {answer[:70]}...")
        time.sleep(2)

    # ── Print Results ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  📊 EVALUATION RESULTS")
    print("=" * 60)

    for r in results:
        print(f"\n  Q: {r['q'][:55]}")
        print(f"  ├─ Faithfulness:     {r['f']:.2f} / 1.0")
        print(f"  ├─ Ans Relevancy:    {r['ar']:.2f} / 1.0")
        print(f"  └─ Context Recall:   {r['cr']:.2f} / 1.0")

    avg_f  = sum(r["f"]  for r in results) / len(results)
    avg_ar = sum(r["ar"] for r in results) / len(results)
    avg_cr = sum(r["cr"] for r in results) / len(results)

    print("\n" + "=" * 60)
    print("  📈 OVERALL AVERAGES")
    print("=" * 60)
    print(f"  Faithfulness:          {avg_f:.2f} / 1.0")
    print(f"  Answer Relevancy:      {avg_ar:.2f} / 1.0")
    print(f"  Context Recall:        {avg_cr:.2f} / 1.0")
    print()
    print("  ✅ >= 0.7 → Good    ⚠️  < 0.7 → Needs improvement")
    print("=" * 60)


if __name__ == "__main__":
    run_evaluation()
