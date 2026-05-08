"""
Stage 6: analyse
Master script that runs all analysis modules in sequence.
"""
import subprocess
import sys

scripts = [
    "src/analyse_descriptive.py",
    "src/analyse_tests.py",
    "src/analyse_lme.py",
    "src/analyse_rope.py",
    "src/analyse_clustering.py",
]

for script in scripts:
    print(f"\n{'='*60}")
    print(f"Running {script}...")
    print('='*60)
    subprocess.run([sys.executable, script], check=True)

print("\n✅ All analyses complete!")