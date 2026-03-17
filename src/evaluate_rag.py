from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .rag_answer import answer_question


@dataclass
class EvalCase:
    question: str
    expected_contains: List[str]
    description: str = ""


def build_eval_set() -> list[EvalCase]:
    return [
        EvalCase(
            question="Which experiment had the highest accuracy?",
            expected_contains=["exp_002", "0.871"],
            description="Best experiment lookup",
        ),
        EvalCase(
            question="Which experiment used ResNet18?",
            expected_contains=["exp_001", "exp_002", "ResNet18"],
            description="Model lookup",
        ),
        EvalCase(
            question="What was the accuracy of exp_001?",
            expected_contains=["0.842", "exp_001"],
            description="Direct metric lookup",
        ),
        EvalCase(
            question="Why did exp_002 perform better than exp_001?",
            expected_contains=["augmentation", "generalization"],
            description="Comparison reasoning",
        ),
        EvalCase(
            question="Summarize the conclusion of exp_002.",
            expected_contains=["augmentation", "generalization", "overfitting"],
            description="Conclusion retrieval",
        ),
        EvalCase(
            question="Which dataset was used in exp_001?",
            expected_contains=["CIFAR-10"],
            description="Dataset lookup",
        ),
    ]


def normalize(text: str) -> str:
    return " ".join(text.lower().split())


def evaluate_case(case: EvalCase, top_k: int = 5) -> dict:
    answer = answer_question(case.question, top_k=top_k)
    answer_norm = normalize(answer)

    matched = [term for term in case.expected_contains if normalize(term) in answer_norm]
    missing = [term for term in case.expected_contains if normalize(term) not in answer_norm]

    passed = len(missing) == 0

    return {
        "question": case.question,
        "description": case.description,
        "answer": answer,
        "matched": matched,
        "missing": missing,
        "passed": passed,
    }


def main():
    cases = build_eval_set()
    results = []

    print("\n=== RAG EVALUATION ===\n")

    for i, case in enumerate(cases, 1):
        result = evaluate_case(case, top_k=5)
        results.append(result)

        status = "PASS" if result["passed"] else "FAIL"
        print(f"[{i}] {status} - {case.description}")
        print(f"Question: {result['question']}")
        print(f"Matched: {result['matched']}")
        if result["missing"]:
            print(f"Missing: {result['missing']}")
        print("Answer:")
        print(result["answer"])
        print("\n" + "=" * 60 + "\n")

    passed_count = sum(r["passed"] for r in results)
    total = len(results)

    print("=== SUMMARY ===")
    print(f"Passed: {passed_count}/{total}")
    print(f"Failed: {total - passed_count}/{total}")


if __name__ == "__main__":
    main()