"""Standalone figures for the presentation; no invented or interpolated success metrics."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/tln_mpcc_matplotlib')
import json
from pathlib import Path
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from world import World
out=Path(__file__).parent/'results';w=World()
path=out/'evaluation_final.json'
if not path.exists():path=out/'evaluation_bc.json'
rows=json.loads(path.read_text());models=list(dict.fromkeys(r['model'] for r in rows));fig,axs=plt.subplots(1,len(models),figsize=(4*len(models),5),squeeze=False)
for ax,name in zip(axs[0],models):
 s=np.linspace(0,w.length,900);p=w.xy(s);ax.plot(p[:,0],p[:,1],'k--',lw=1,label='Raceline')
 for r in rows:
  if r['model']!=name:continue
  t=np.array(r['trajectory']);ax.plot(t[:,0],t[:,1],lw=1,alpha=.7,label=str(r['seed']))
  if not r['success']:ax.scatter(t[-1,0],t[-1,1],marker='x',s=30)
 ax.set_title(name);ax.set_aspect('equal');ax.set_xlabel('x [m]');ax.set_ylabel('y [m]');ax.legend(fontsize=6)
fig.tight_layout();fig.savefig(out/'trajectories.png',dpi=160);plt.close(fig)
fig,ax=plt.subplots(figsize=(7,4));rate=[sum(r['success'] for r in rows if r['model']==name)/sum(r['model']==name for r in rows)*100 for name in models];ax.bar(models,rate,color=['#707070','#b0b0b0','#3878c7','#21815d','#e58f2c'][:len(models)]);ax.set_ylim(0,100);ax.set_ylabel('No-reset full-lap success [%]')
for i,v in enumerate(rate):ax.text(i,v+2,f'{v:.1f}%',ha='center')
fig.tight_layout();fig.savefig(out/'success_rates.png',dpi=160);plt.close(fig)
print('figures written',path)
