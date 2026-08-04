"""
RAG Evaluation Pipeline — RAGAS.

Đánh giá pipeline RAG với 4 metrics: faithfulness, answer_relevancy,
context_recall, context_precision.
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).parent.parent / ".env")

EVAL_DIR = Path(__file__).parent
GOLDEN_DATASET_PATH = EVAL_DIR / "golden_dataset.json"
RESULTS_PATH = EVAL_DIR / "results.md"


def load_golden_dataset():
    with open(GOLDEN_DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def generate_answer(question: str, contexts: list[str]) -> str:
    """Generate answer from retrieved contexts using OpenAI."""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    context_text = "\n\n---\n\n".join(contexts[:5])

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": (
                "Bạn là chatbot hỗ trợ khách hàng Shopee. Trả lời câu hỏi "
                "DỰA TRÊN TÀI LIỆU được cung cấp. Nếu tài liệu không đủ thông tin, "
                "hãy nói 'Tôi không tìm thấy thông tin này trong tài liệu.' "
                "Trả lời bằng tiếng Việt, ngắn gọn, rõ ràng."
            )},
            {"role": "user", "content": f"Tài liệu tham khảo:\n{context_text}\n\nCâu hỏi: {question}"},
        ],
        temperature=0,
        max_tokens=500,
    )
    return response.choices[0].message.content


def run_eval():
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_recall,
        context_precision,
    )
    from datasets import Dataset

    # Import RAG pipeline
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.task9_retrieval_pipeline import retrieve

    golden = load_golden_dataset()
    print(f"Loaded {len(golden)} test cases\n")

    eval_data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [],
    }

    for item in golden:
        print(f"Evaluating: {item['id']} — {item['question'][:60]}...")

        # Step 1: Retrieve contexts
        retrieval_results = retrieve(item["question"], top_k=5)
        contexts = [r["content"] for r in retrieval_results]
        context_sources = [r["metadata"].get("source", "?") for r in retrieval_results]

        # Step 2: Generate answer
        answer = generate_answer(item["question"], contexts)

        eval_data["question"].append(item["question"])
        eval_data["answer"].append(answer)
        eval_data["contexts"].append(contexts)
        eval_data["ground_truth"].append(item["expected_answer"])

        print(f"  Sources: {context_sources[:3]}")
        print(f"  Answer: {answer[:100]}...")
        print()

    # Step 3: Run RAGAS evaluation
    print("Running RAGAS evaluation...")
    dataset = Dataset.from_dict(eval_data)

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
    )

    df = result.to_pandas()
    print("\n" + "=" * 60)
    print("RAGAS Evaluation Results")
    print("=" * 60)
    print(df[["faithfulness", "answer_relevancy", "context_recall", "context_precision"]].describe())

    # Step 4: Export results
    export_results(df, golden)
    return df


def export_results(df, golden):
    avg = df[["faithfulness", "answer_relevancy", "context_recall", "context_precision"]].mean()

    content = "# RAG Evaluation Results\n\n"
    content += "## Pipeline Config\n\n"
    content += "- Embedding: `text-embedding-3-small` (OpenAI, 1536-dim)\n"
    content += "- Chunking: `RecursiveCharacterTextSplitter` (800/100)\n"
    content += "- Retrieval: Semantic + BM25 → RRF (k=60)\n"
    content += "- Generation: `gpt-4o-mini`\n"
    content += "- Datasets: 3 chính sách Shopee (trả hàng, vận chuyển, bảo mật)\n\n"

    content += "## Overall Scores\n\n"
    content += "| Metric | Score |\n"
    content += "|--------|-------|\n"
    for col in ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]:
        content += f"| {col} | {avg[col]:.4f} |\n"

    content += "\n## Per-Question Scores\n\n"
    content += "| ID | Question | Faithfulness | Answer Relevancy | Context Recall | Context Precision |\n"
    content += "|---|----------|-------------|-----------------|---------------|------------------|\n"
    for i, row in df.iterrows():
        q = golden[i]["question"][:50]
        content += f"| {golden[i]['id']} | {q}... | "
        content += f"{row['faithfulness']:.3f} | {row['answer_relevancy']:.3f} | "
        content += f"{row['context_recall']:.3f} | {row['context_precision']:.3f} |\n"

    content += "\n## Worst Performers (bottom 3)\n\n"
    worst = df.nsmallest(3, "faithfulness")
    for i, row in worst.iterrows():
        content += f"- **{golden[i]['id']}**: *{golden[i]['question'][:80]}...*\n"
        content += f"  - faithfulness={row['faithfulness']:.3f}, context_recall={row['context_recall']:.3f}\n"

    content += "\n## Nhận Xét\n\n"
    content += "- Pipeline hybrid (semantic + BM25) cho kết quả retrieval tốt\n"
    content += "- Generation dùng gpt-4o-mini cho câu trả lời tiếng Việt tự nhiên\n"
    content += "- Các câu hỏi lệch domain có thể có score thấp do thiếu context phù hợp\n"

    RESULTS_PATH.write_text(content, encoding="utf-8")
    print(f"\nResults exported to {RESULTS_PATH}")


if __name__ == "__main__":
    run_eval()
