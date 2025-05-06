import matplotlib

COLORS = list(matplotlib.colors.TABLEAU_COLORS)
# swap 1st and 3rd colors
COLORS[0], COLORS[2] = COLORS[2], COLORS[0]
HATCHES = ['\\\\', '-', '//', 'x', '.', '',]
MARKERS = ['o', 's', '^', 'D', 'v', 'p']
OPACITY = 0.7

FONT = 'Sans Serif'

PRETTY_NAMES = {
    "akasha": "Heimdall",
    "sglang": "SGLang",
    "vllm": "vLLM",
    "sglang_wb": "SGLang-WB",
    "sglang_wt": "SGLang-WT",
    "sglang_wts": "SGLang-WTS",
}

SYSTEM_ID_MAP = {
    "akasha": 0,
    "sglang_wb": 1,
    "sglang_wt": 2,
    "sglang_wts": 3,
    "vllm": 4,
}

SYSTEM_NAME_MAP = {
    "akasha": "Heimdall",
    "sglang_wb": "SGLang-WB",
    "sglang_wt": "SGLang-WT",
    "sglang_wts": "SGLang-WTS",
    "vllm": "vLLM",
}