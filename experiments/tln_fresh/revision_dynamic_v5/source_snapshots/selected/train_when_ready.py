"""Start only after all requested train and val episode metadata are closed."""
from pathlib import Path
import time,subprocess,sys
r=Path(__file__).resolve().parent
while True:
 train=list((r/'data').glob('train_*.json'));val=list((r/'data').glob('val_*.json'))
 if len(train)>=80 and len(val)>=16:break
 time.sleep(5)
subprocess.check_call([sys.executable,str(r/'prepare.py')])
raise SystemExit(subprocess.call([sys.executable,str(r/'train.py'),'--epochs','100']))
