import json
import os
from importlib.resources import files as _pkg_files
from typing import Dict, List

from veeksha.core.response import ChannelResponse


def store_generated_texts(
    output_dir: str, generated_responses: List[ChannelResponse]
) -> None:
    """Store generated responses in a text file."""
    with open(os.path.join(output_dir, "generated_texts.txt"), "w") as f:
        f.write(
            ("\n" + "-" * 30 + "\n").join([str(i.content) for i in generated_responses])
        )


def store_lmeval_results(output_dir: str, lmeval_results: Dict) -> None:
    """Store LMEval results in a JSON file."""
    with open(os.path.join(output_dir, "lmeval_results.json"), "w") as f:
        json.dump(lmeval_results, f, indent=4)


def load_corpus() -> List[str]:
    """Load corpus lines from packaged corpus.txt file"""
    corpus_resource = _pkg_files("veeksha.data").joinpath("corpus.txt")
    with corpus_resource.open("r", encoding="utf-8") as f:
        return f.readlines()
