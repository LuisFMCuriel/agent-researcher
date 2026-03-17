import argparse
from .agent import run_agent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("question", type=str, help="Question to research")
    args = parser.parse_args()

    out = run_agent(args.question)
    print("\n=== ANSWER ===\n")
    print(out["answer"])
    print("\n=== CITATIONS ===\n")
    for c in out["citations"]:
        print(f"- {c['title']}: {c['url']}")
    print("\n=== TRACE ===\n")
    for i, t in enumerate(out["trace"], 1):
        print(f"{i}. {t['kind']}: {t['payload']}")

if __name__ == "__main__":
    main()
