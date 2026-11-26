from pathlib import Path
import json,hashlib,csv,collections,re,os
import numpy as np
os.environ.setdefault('MPLCONFIGDIR','/tmp/tln_fresh_mpl')
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from world import World
root=Path(__file__).parent;r=root/'corner_revision';out=r/'deliverables';out.mkdir(exist_ok=True)
selection=json.loads((r/'results/final_selection.json').read_text());training=json.loads((r/'results_dagger/training_corner.json').read_text());rows=json.loads((r/'results/evaluation_final_test.json').read_text());summary=json.loads((r/'results/evaluation_final_test_summary.json').read_text());assert len(rows)==52
stats={}
for c in ['neural','protected']:
 stats[c]={}
 for group in ['all','corner_exit','full_lap']:
  rr=[x for x in rows if x['controller']==c and (group=='all' or (x['mode']=='corner_exit' if group=='corner_exit' else x['required_progress_m']>50))]
  ok=[x for x in rr if x['success']];stats[c][group]={'trials':len(rr),'successes':len(ok),'wall':sum(x['collision']=='wall' for x in rr),'obstacle':sum(x['collision']=='obstacle' for x in rr),'stop':sum(not x['success'] and not x['collision'] for x in rr),'successful_mean_s':float(np.mean([x['sim_time_s'] for x in ok])) if ok else None}
ros=[]
for folder in ['ros_recordings','ros_recordings_safe','ros_recordings_serial','ros_recordings_sync_fixed','ros_recordings_background','ros_recordings_full']:
 p=r/folder/'manifest.json'
 if p.exists():
  for x in json.loads(p.read_text()):ros.append({'folder':folder,**x})
speed=json.loads((r/'results/realtime_basic_speed_check.json').read_text());mode_counts=collections.Counter(e['mode'] for e in training['train_episodes']);records=[]
for folder in ['data','dagger_data']:
 for p in (r/folder).glob('*.json'):
  e=json.loads(p.read_text());records.append({'folder':folder,'split':e['split'],'mode':e['mode'],'seed':e['seed'],'success':e['success'],'collision':e['collision'],'samples':e['samples']})
(r/'results/recording_inventory.json').write_text(json.dumps(records,indent=2));(r/'results/ros_recording_inventory.json').write_text(json.dumps(ros,indent=2))
lines=['# 급코너 출구 녹화·새 TLN 학습·정지 문제 점검','', '**화면의 느린 주행은 실시간 실행으로 복구했다. 새 회피·MPCC 데이터로 기존 TLN 가중치 없이 모델을 다시 만들었다. 다만 모든 배치에서 충돌과 정지를 없애는 목표는 아직 달성하지 못했다.**','','## 화면 속도와 중간 정지','',f'동기 녹화에서는 전문가 계산을 기다리므로 실제 경과 시간보다 물리 시간이 느리게 진행했다. 화면 주행과 추가 녹화는 도메인 0과 42로 분리했다. 아래 속도 측정은 학생으로 전환하기 전 기본 레이스라인 주행의 결과다. 이후 사용자 요청에 따라 화면을 최종 새 TLN과 경로 보호로 전환하고 정적 장애물 3개를 추가했다. 실시간 시연에서는 벽 충돌이 확인됐으며 안정적인 회피 완주로 보고하지 않는다. 직선 목표 속도는 2.2 m/s이고 코너에서는 감속한다. 8초 재측정의 물리 시간/실제 시간 비율은 {speed["physics_to_wall_ratio"]:.4f}, 최고 측정 속도는 {speed["measured_speed_max_mps"]:.2f} m/s, 0 속도 명령은 {speed["zero_commands"]}회였다.','','확인한 사실: 초기 관측에서 충돌 없이 0 명령이 두 번 나왔고, 이후 진단 로그에서는 두 센서의 수신 시각이 함께 약 2.9초 오래된 상태에서 센서 유실 정지가 실행됐다. 수동 조작 메시지는 관측되지 않았다. 이는 사용자가 본 모든 정지의 원인이 하나라고 확정한 결과는 아니다.','','수정: 깊이 1의 센서 수신 큐, 정지 사유 기록, 제어 출력을 막을 수 있는 반복 터미널 출력 제거, 실행 재개 시 대기 중인 센서 콜백을 처리할 최대 0.05초의 제한된 유예를 적용했다. 정상적인 센서 유실의 0.3초 정지 조건은 유지한다. 새 센서 없이 반복 지연이 나도 유예를 계속 연장하지 않는다. 해당 네 가지 경우를 단위 시험으로 확인했다.','','시뮬레이터 실행을 직렬화하고 실제 물리 시간과 측정 조향각을 읽도록 했다. 원래 코드에서 물리 계산이 재진입 가능한 다중 스레드 타이머로 실행되는 것은 확인했지만, 과거 모든 실패가 실제 동시 계산 때문이었다고 입증한 것은 아니다. 물리식과 충돌 판정은 기존 구현을 상속한다. 일반 실시간 모드와 명령당 물리 계산 10회를 수행하는 녹화 모드를 구분한다.','','## 새 데이터와 학습','', '- 지도: `hangar_1905_v0`. 코너 위치 약 s=25.1, 32.1, 35.8, 54.5 m의 출구를 대상으로 출구 거리와 좌우 배치를 바꿨다. 지도상 여유가 부족한 후보는 제외했다.', '- 새 학습용 코너 출구 후보 28회 중 26회 완주, 새 MPCC 전체 랩 후보 6회 중 4회 완주, 기본 전체 랩 2회 모두 완주. 정지 실패 4회는 학습에서 제외했다.', '- 복구 상태를 늘리기 위한 행동 혼합·잡음 수집 24회 중 22회 완주. 장애물 충돌 2회는 제외했다. 실제 실행 명령과 전문가 정답은 별도로 기록했다.', f'- 최종 학습 회차 {len(training["train_episodes"])}개: '+', '.join(f'{k} {v}개' for k,v in sorted(mode_counts.items()))+'.',f'- 좌우 반전 뒤 학습 샘플 {training["training_samples_after_mirror"]:,}개, 회귀 검증용 성공 회차 {len(training["validation_episodes"])}개. 실패 배치는 폐루프 검증·시험에서 제외하지 않았다.', '- 이전 작업에서 직접 모은 새 데이터와 고속 기본주행 ROS 샘플도 포함했다. 과거의 예전 TLN 모델·가중치·주행 배열을 가져오지 않았다. 학습 목록과 해시는 `training_corner.json`이 기준이다.', '- 새 학습은 무작위 초기화부터 시작했다. 행동 혼합 녹화에서 이번 작업의 중간 학생 모델을 사용한 것은 학습 초기 가중치를 불러오는 것과 구분한다.',f'- 파라미터 {training["parameters"]:,}개, 학습 {training["epochs"]}회. 검증 25개 배치에서 신경망 단독 완주가 첫 후보 15회, 복구 데이터를 넣은 후보 16회여서 후자를 선택했다. 시험 결과를 보기 전에 선택을 고정했다.',f'- 최종 SHA256: `{selection["sha256"]}`.','','## 입력과 출력','', '입력은 LiDAR 거리 3프레임과 전경 마스크 3프레임 `(541,6)`, 지도 경로·속도·이전 조향·장애물 후보 42차원이다. 관측한 장애물 위치를 차량 이동에 맞춰 갱신하며 최대 30개 제어 단계 동안 유지한다. 정답 장애물 좌표는 학생 입력에 넣지 않는다. 지도와 GT 위치는 사용하므로 LiDAR 단독 TLN이나 실제 위치 추정 환경과 동일하지 않다.','','출력은 보조 경로 오프셋 3개, 목표 속도 비율, 조향 보정이다. 속도는 2.8 m/s, 조향 보정은 0.72 rad로 복원해 기준 경로 조향에 더한다. 최종 조향은 ±0.36 rad이며 지도 코너 전방과 조향 곡률로 속도를 제한한다. EMA와 양의 최저 속도는 없다. MPCC 전문가는 관측 가능한 장애물의 정밀 좌표를 사용하므로 학생의 LiDAR 추정값과 정보 정밀도 차이가 남는다.','','## 학습에 사용하지 않은 최종 시험','', '모든 회차는 위치 초기화 없이 평가했다. 코너 출구는 11 m 구간이며 전체 랩이 아니다. 나머지 회차는 55.2902 m 이상의 전체 경로 진행을 요구한다. 코너 배치 14개와 전체 랩 배치 12개를 평가했고, 전문가가 실패한 배치도 포함했다. 정지는 성공으로 세지 않았다.','','| 방식 | 구분 | 완주/시험 | 벽 충돌 | 장애물 충돌 | 정지·미완주 | 성공 구간 평균 시간 |','|---|---|---:|---:|---:|---:|---:|']
for c,label in [('neural','새 신경망'),('protected','새 신경망 + 경로 보호')]:
 for g,title in [('all','전체'),('corner_exit','코너 출구 11 m'),('full_lap','전체 랩')]:
  x=stats[c][g];t='구간 길이 혼합으로 생략' if g=='all' else (f'{x["successful_mean_s"]:.2f} s' if x['successful_mean_s'] is not None else '성공 없음');lines.append(f'| {label} | {title} | {x["successes"]}/{x["trials"]} | {x["wall"]} | {x["obstacle"]} | {x["stop"]} | {t} |')
lines+=['','시간은 고정 물리 시간 간격 기준이며 실제 ROS 경과 시간과 직접 비교하지 않는다. 시간 평균은 성공한 회차만 포함한다. 전체 지연 수치는 입력 특징 생성·신경망·보호 계산을 포함하지만 ROS 전송 및 실제 작동기 지연은 포함하지 않는다.','','보호 제어는 신경망과 별도의 기하 경로 탐색 및 제동을 포함한다. 그 성능을 신경망 단독의 성능으로 해석하지 않는다. 보호가 정지시킨 회차도 해결된 회피로 보고하지 않는다. 보조 경로를 직접 조향으로 쓰는 후보는 검증 성능이 낮아 선택하지 않았다.','','## 실제 ROS 녹화와 남은 한계','', '이전 중단 실험의 고아 녹화기 3개가 남아 디스크에 기록을 계속하던 것을 확인해 종료했고, 종료 처리에서 bag을 항상 닫도록 수정했다. 이 부하가 모든 실패의 유일한 원인이라고 단정하지 않는다.\n\n오프라인 원시 센서 기록이 학습용 데이터다. 추가 ROS bag은 실행·시간 정렬 검증을 위한 진단 자료로 보존하고 이번 모델 학습에 자동으로 합치지 않았다. 먼저 수집한 실패 bag도 로컬에 보존했다. 업로드하는 ROS 묶음에는 닫힌 도메인 42의 배경 녹화와 MPCC 전체 랩 원본을 넣고, 중단된 대용량 bag은 학습에 넣지 않았다. 다른 실패 녹화의 요약과 로그는 코드·결과 묶음에 넣었다. 실제 ROS 녹화 현황은 다음과 같다.','','| 녹화 폴더 | 회차 | 성공 | 진행 거리 | 충돌 |','|---|---|---|---:|---|']
for x in ros:lines.append(f'| {x["folder"]} | {x["episode"]} | {"완주" if x["success"] else "미완주"} | {x["progress_m"]:.2f} m | {", ".join(x["collision_events"]) or "없음"} |')
lines+=['','급코너 직후 일부 배치에서는 전문가도 정지하며, 학생의 폐루프 상태 오차·장애물 중심 추정 오차·짧은 가시 거리의 영향이 남아 있다. 각 원인의 기여를 분리해 확정한 것은 아니다. 현재 모델을 모든 장애물에서 충돌 없이 빠르게 완주하는 모델이라고 보고할 근거는 부족하다.','','## 현재 RViz 시연','', '최종 모델을 실제 실시간 bridge의 학생 제어기로 실행 중이다. 기본 제어기의 주행을 학생 모델이라고 표시하지 않는다. 직선 s=10.0 m, 급코너 출구 s=33.1 m, 복귀 구간 s=46.0 m에 0.3 m 정적 장애물을 배치했다. 장애물 좌표 및 실시간 충돌·복구 기록은 `rviz_three_obstacles.json`에 있다. 시연 실패를 완주 성공으로 세지 않는다. 별도 녹화 시뮬레이터는 종료했다.','', '## 파일','', '`final_model_corner.keras`, `MODEL_MANIFEST.json`, `training_corner.json`, `evaluation_final_test_summary.json`, `evaluation_trials.csv`, `outcomes.png`, `corner_trajectories.png`, `recorded_dataset.zip`, `source_and_results.zip`, `ros_recordings.zip`을 함께 제공한다. 재현 절차는 `REPRODUCE_KO.md`에 있다.']
text='\n'.join(lines)+'\n';assert not re.search('[\u3400-\u9fff\u3040-\u30ff]',text);(out/'CORNER_TLN_REPORT_KO.md').write_text(text)
fields=['controller','mode','seed','success','collision','progress_m','required_progress_m','sim_time_s','mean_abs_raceline_error_m','end_to_end_control_ms_p95','recovery_steps']
with (out/'evaluation_trials.csv').open('w') as f:
 writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
fig,axes=plt.subplots(1,2,figsize=(10,4))
for ax,(group,title) in zip(axes,[('corner_exit','Corner exits: 11 m segments'),('full_lap','Full-lap layouts')]):
 bottom=np.zeros(2)
 for key,color,label in [('successes','#2e9d63','Completed'),('wall','#d74747','Wall collision'),('obstacle','#ef9e36','Obstacle collision'),('stop','#888888','Stop / timeout')]:
  values=np.array([stats[c][group][key] for c in ['neural','protected']]);ax.bar(['Neural','Neural + guard'],values,bottom=bottom,color=color,label=label);bottom+=values
 ax.set_title(title);ax.set_ylabel('Trials');ax.set_ylim(0,max(bottom)+1);ax.set_yticks(range(0,int(max(bottom))+1,2))
axes[1].legend(fontsize=8);fig.tight_layout();fig.savefig(out/'outcomes.png',dpi=170);plt.close(fig)
w=World();seeds=sorted({x['seed'] for x in rows if x['mode']=='corner_exit'});fig,axes=plt.subplots(2,7,figsize=(21,7));s=np.linspace(0,w.length,1200);road=w.xy(s)
for ax,seed in zip(axes.flat,seeds):
 rr=[x for x in rows if x['seed']==seed];scenario=json.loads((root/rr[0]['scenario']).read_text());coords=[];ax.plot(road[:,0],road[:,1],'--',c='#777777',lw=.8)
 for x,color in zip(rr,['#2679bd','#e58b22']):
  tr=np.array([t['state'][:2] for t in x['trace']]);coords.extend(tr.tolist());ax.plot(tr[:,0],tr[:,1],c=color,label=x['controller'],lw=1.2);ax.scatter(tr[-1,0],tr[-1,1],s=18,c=color,marker='o' if x['success'] else 'x')
 for b in scenario['obstacles']:
  theta=b['theta'];v=np.array([[-.15,-.15],[.15,-.15],[.15,.15],[-.15,.15]]);rotation=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]]);poly=v@rotation.T+[b['x'],b['y']];ax.add_patch(Polygon(poly,color='#c94c4c'))
 coords=np.array(coords);lo=coords.min(0)-.8;hi=coords.max(0)+.8;ax.set_xlim(lo[0],hi[0]);ax.set_ylim(lo[1],hi[1]);ax.set_aspect('equal');ax.set_title(str(seed),fontsize=9);ax.tick_params(labelsize=7)
axes.flat[0].legend(fontsize=6);fig.suptitle('Held-out corner exits: blue neural, orange guarded; x = failed end');fig.tight_layout();fig.savefig(out/'corner_trajectories.png',dpi=150);plt.close(fig)
(out/'MODEL_MANIFEST.json').write_text(json.dumps({'selection':selection,'input_shapes':[[541,6],[42]],'mode_counts':mode_counts,'training_samples_after_mirror':training['training_samples_after_mirror'],'from_random_initialization':True,'legacy_weights_used':False,'test_summary':stats,'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.glob('*.py'))}},indent=2));print(json.dumps(stats,indent=2));print('REPORT COMPLETE')
