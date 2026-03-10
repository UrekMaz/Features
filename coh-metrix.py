"""
check_cohmetrix_output.py
Run this to see exactly which features your CohMetrix CLI outputs.
"""
import subprocess
import tempfile
import os

COH_METRIX_CLI = os.path.join(os.path.dirname(__file__), "CohMetrixCore", "CohMetrixCoreCLI.exe")
COH_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "cohmetrix_output")
os.makedirs(COH_OUTPUT_DIR, exist_ok=True)

sample = (
    "The mitochondria is the powerhouse of the cell. "
    "It produces energy through a process called cellular respiration. "
    "This process converts glucose and oxygen into ATP, carbon dioxide, and water. "
    "The ATP produced is then used by the cell to perform various functions. "
    "Without mitochondria, cells would not be able to produce the energy they need."
)

with tempfile.NamedTemporaryFile(
    mode="w", suffix=".txt", dir=COH_OUTPUT_DIR,
    delete=False, encoding="utf-8"
) as f:
    f.write(sample)
    tmp_path = f.name

subprocess.run([COH_METRIX_CLI, tmp_path, COH_OUTPUT_DIR], check=True, capture_output=True, timeout=60)

output_csv = tmp_path + ".csv"
cli_features = {}
with open(output_csv, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split(",")
        if len(parts) == 2:
            try:
                cli_features[parts[0].strip()] = float(parts[1].strip())
            except ValueError:
                pass

os.remove(tmp_path)
os.remove(output_csv)

print(f"CLI outputs {len(cli_features)} features:\n")
for k, v in sorted(cli_features.items()):
    print(f"  {k}: {v}")

# Check which FS5 CohMetrix features are missing
fs5_cohmetrix = [
    'SMCAUSwn', 'WRDPOLc', 'CNCTemp', 'PCNARp', 'SYNSTRUTt', 'CRFCWOad',
    'LSAGNd', 'WRDHYPnv', 'WRDFAMc', 'WRDCNCc', 'WRDMEAc', 'WRDFRQa',
    'WRDFRQmc', 'WRDPRP1p', 'LSAGN', 'LSASSpd', 'LSASS1', 'PCTEMPp',
    'CNCTempx', 'SMCAUSlsa', 'DRNEG', 'WRDHYPn','PCREFp'
]

print(f"\n── FS5 CohMetrix features in CLI output ──")
for f in fs5_cohmetrix:
    status = "✓ in CLI" if f in cli_features else "✗ MISSING — needs manual"
    print(f"  {f}: {status}")