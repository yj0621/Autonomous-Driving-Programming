from pathlib import Path
import time,subprocess,sys
r=Path(__file__).resolve().parent
while not (r/'results/feature_preparation.json').exists():time.sleep(5)
raise SystemExit(subprocess.call([sys.executable,str(r/'train.py'),'--epochs','100']))
