"""Post-hoc failure measurements; no policy changes and no test data training."""
from pathlib import Path
import json
import numpy as np
from world import World
from policy import basic_action
root=Path(__file__).parent;rows=json.loads((root/'results/evaluation_strict_final_test.json').read_text());w=World();analysis=[]
for row in rows:
 if row['success']:continue
 scenario=json.loads((root/'data'/f"test_{row['scenario_mode']}_{row['seed']}.json").read_text());trace=row['trace'];last=np.asarray(trace[-1]['state']);obs=scenario['obstacles'];b=min(obs,key=lambda b:np.linalg.norm(last[:2]-[b['x'],b['y']])) if obs else None
 entry={'controller':row['controller'],'seed':row['seed'],'outcome':'collision' if row['collision'] else 'stopped','kind':row['collision'],'progress_m':row['progress_m'],'last_xy':last[:2].tolist(),'last_speed_state_mps':float(last[3]),'last_command':trace[-1]['command'],'last_guard':trace[-1]['supervisor'],'closest_obstacle_proxy':b}
 if b:
  reaction=None
  for t in trace:
   st=np.asarray(t['state']);dist=float(np.linalg.norm(st[:2]-[b['x'],b['y']]))
   if dist>5:continue
   baseline,_=basic_action(w,st,w.project(*st[:2]));difference=abs(t['command'][0]-baseline[0])
   if difference>=.06:reaction={'distance_to_obstacle_centre_m':dist,'steering_difference_rad':difference};break
  entry['first_deviation_from_basic_at_least_0p06rad_within5m']=reaction
  so=b['s'];car_s=w.project(*last[:2]);dcar=float(np.dot(last[:2]-w.xy(car_s),w.normal(car_s)));entry['car_lateral_offset_m']=dcar;entry['obstacle_lateral_offset_m']=b['d'];entry['centre_lateral_separation_m']=abs(dcar-b['d']);entry['approx_required_separation_m']=b['half_w']+.1016+.05
 entry['interpretation_limit']='Closest obstacle and lateral separation are geometric proxies; this does not prove a unique cause or global path infeasibility.'
 analysis.append(entry)
(root/'results/failure_analysis.json').write_text(json.dumps(analysis,indent=2));print('실패 사례 분석',len(analysis),'건')
