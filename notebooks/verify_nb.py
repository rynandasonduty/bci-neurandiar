import json, re, sys

NB_PATH = "BCI Analysis and Results.ipynb"
with open(NB_PATH, encoding="utf-8") as f:
    nb = json.load(f)

cells = nb["cells"]
code_n = sum(1 for c in cells if c["cell_type"] == "code")
md_n   = sum(1 for c in cells if c["cell_type"] == "markdown")

print(f"nbformat : {nb['nbformat']}")
print(f"Cells    : {len(cells)}  (code={code_n}, markdown={md_n})")
print()

# List cells
for i, c in enumerate(cells):
    src = "".join(c["source"])[:75].replace("\n", " ")
    safe_src = src.encode('ascii', errors='replace').decode('ascii')
    print(f"[{i:02d}] {c['cell_type']:10} id={c['id']:12} | {safe_src}")

print()

# Check for emojis
EMOJI_RE = re.compile(
    "[\U0001F300-\U0001FFFF"
    "\U00002600-\U000027BF"
    "\U0001F000-\U0001F02F"
    "\U0001F0A0-\U0001F0FF"
    "\U0001F100-\U0001F1FF"
    "\U0001F200-\U0001F2FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FA6F"
    "✂-➰]+"
)
emoji_found = []
for i, c in enumerate(cells):
    text = "".join(c["source"])
    if EMOJI_RE.search(text):
        snippet = EMOJI_RE.search(text).group()
        emoji_found.append((i, c["cell_type"], snippet))

if emoji_found:
    print("EMOJI VIOLATIONS:")
    for idx, ct, em in emoji_found:
        print(f"  Cell {idx} ({ct}): {repr(em)}")
    sys.exit(1)
else:
    print("Emoji check: PASS (none found)")

# Check required sections
REQUIRED = [
    "Main Analysis 1",
    "Main Analysis 2",
    "Main Analysis 3",
    "Main Analysis 4",
    "Main Analysis 5",
    "Stage 6",
    "Wilcoxon",
    "SHAP",
    "Friedman",
    "Champion",
    "Subject Rescue",
    "Inference Latency",
]
full_text = "\n".join("".join(c["source"]) for c in cells)
missing = [k for k in REQUIRED if k not in full_text]
if missing:
    print("MISSING SECTIONS:", missing)
    sys.exit(1)
else:
    print("Section check : PASS (all required sections present)")

print()
print("Verification complete. Notebook is structurally valid.")
