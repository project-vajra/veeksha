import os
import json
from typing import List, Dict
from veeksha.core.response import Response


def store_generated_texts(output_dir: str, generated_responses: List[Response]) -> None:
    """Store generated responses in a text file."""
    with open(os.path.join(output_dir, "generated_texts.txt"), "w") as f:
        f.write(("\n" + "-" * 30 + "\n").join([i.text for i in generated_responses]))


def store_lmeval_results(output_dir: str, lmeval_results: Dict) -> None:
    """Store LMEval results in a JSON file."""
    with open(os.path.join(output_dir, "lmeval_results.json"), "w") as f:
        json.dump(lmeval_results, f, indent=4)


def load_corpus() -> List[str]:
    """Load the corpus lines from the corpus.txt file."""
    corpus_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "data", "corpus.txt")
    )
    with open(corpus_path, "r") as f:
        corpus_lines = f.readlines()
    return corpus_lines
