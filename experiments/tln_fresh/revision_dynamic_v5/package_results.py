"""Freeze measured artifacts. All failures and controller assistance remain explicit."""
from pathlib import Path
import sys,json,hashlib,shutil,zipfile,importlib.metadata
root=Path(__file__).resolve().parent;workspace=root.parents[2];sys.path.insert(0,str(root.parent));out=root/'deliverables';out.mkdir(exist_ok=True)
import numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from world import World
w=World();results=root/'results';test=json.loads((results/'evaluation_best_regression_test_raw_protected_1lap_final_holdout_summary.json').read_text());val=json.loads((results/'evaluation_best_regression_val_protected_1lap_rawpoints_summary.json').read_text());continuous=json.loads((results/'evaluation_best_regression_val_protected_3lap_final_continuous_summary.json').read_text());runtime=json.loads((results/'final_runtime_snapshot.json').read_text());training=json.loads((results/'training_moe96.json').read_text());physics=json.loads((results/'physics_verification.json').read_text());baseline=json.loads((results/'v4_baseline.json').read_text());ros=json.loads((results/'ros_teacher_audit.json').read_text());raw=test['summary']['raw'];protected=test['summary']['protected'];vr=val['summary']['protected'];cr=continuous['summary']['protected'];source=hashlib.sha256((root/'controller.py').read_bytes()).hexdigest();assert source==test['source_sha256']['controller.py'];assert runtime['model_sha256']==test['model_sha256']
controls=[]
for line in (results/'final_runtime_controls.jsonl').read_text().splitlines():
 q=json.loads(line)
 if q['event']=='control':controls.append(q)
initial=controls[0]['physics_s'];lap_lo=sum(runtime['lap_physics_s'][:2]);lap_hi=sum(runtime['lap_physics_s'][:3]);plot_rows=[q for q in controls if lap_lo<=q['physics_s']-initial<lap_hi];ss=[];prev=None
for q in plot_rows:prev=w.project(*q['state'][:2],hint=prev);ss.append(prev)
fig,ax=plt.subplots(figsize=(10,3.7));ax.scatter(ss,[q['state'][3] for q in plot_rows],s=15,label='Measured speed',color='#167d8d');ax.scatter(ss,[q['command'][1] for q in plot_rows],s=10,alpha=.6,label='Command',color='#e8962d');ax.set(xlabel='Raceline arc (m)',ylabel='Speed (m/s)',title='New model + predictive MPCC: one actual ROS lap',ylim=(0,4.7));ax.grid(alpha=.2);ax.legend();fig.tight_layout();fig.savefig(out/'actual_speed_profile.png',dpi=170);plt.close(fig)
fig,ax=plt.subplots(figsize=(7,4));labels=['Completed','Wall collision','Obstacle collision','Stop / timeout'];values=[[raw['completions'],raw['wall_collisions'],raw['obstacle_collisions'],raw['stops_or_timeouts']],[protected['completions'],protected['wall_collisions'],protected['obstacle_collisions'],protected['stops_or_timeouts']]];x=np.arange(4);ax.bar(x-.18,values[0],.36,label='Neural policy with physical output limits',color='#7996bd');ax.bar(x+.18,values[1],.36,label='With predictive MPCC assistance',color='#167d8d');ax.set(xticks=x,xticklabels=labels,ylabel='Scenarios out of 24',title='Independent held-out scenarios (94000-94023)',ylim=(0,27));ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2);fig.tight_layout();fig.savefig(out/'heldout_results.png',dpi=170);plt.close(fig)
records=[json.loads(p.read_text()) for p in (root/'data').glob('*.json')];train_ok=[q for q in records if q['split']=='train' and q['success']];basic=sum(q['mode']=='basic_mpcc' for q in train_ok);avoid=len(train_ok)-basic;interventions=runtime['mpcc_intervention_steps']/runtime['control_samples']*100
report=f'''# 새 MPCC 기반 TLN 모델 결과

## 현재 결론

기존 TLN 가중치를 사용하지 않고 새 데이터로 새 모델을 학습했고, RViz에서 실행 중이다. 새 독립 시험 24개 중 안전 제어 포함 {protected['completions']}개를 완주했다. 벽 충돌 {protected['wall_collisions']}회, 장애물 충돌 {protected['obstacle_collisions']}회, 정지 또는 시간 초과 {protected['stops_or_timeouts']}개였다. 정지 문제까지 모든 배치에서 해결된 상태는 아니다.

현재 화면의 고정 장애물 3개 배치에서는 결과를 저장한 시점까지 {runtime['full_continuous_laps']}바퀴를 연속 완주했다. 충돌 {len(runtime['collisions'])}회, 출발 이후 위치 재설정 {runtime['pose_resets_after_startup']}회, 후진 명령 {runtime['reverse_command_steps']}개였고, 출발 이후 최대 정지 시간은 물리 시간 {runtime['max_stationary_physics_s_after_startup']:.2f}초였다. 이 결과는 한 배치의 반복 시험이며 무작위 24개 배치 전체의 성공으로 확대할 수 없다.

## 사용자가 지적한 문제와 실제 변경

| 문제 | 실제 확인과 변경 | 확인 범위 |
|---|---|---|
| 잘 가다가 정지 | 교사의 실패·후진을 정상 주행 정답에서 제외했다. 새 계획이 실패하면 직전에 검증한 계획을 다시 검사해 이어가며, 실제 후처리를 적용한 명령을 검사한다. | 현재 고정 배치에서는 장시간 주행 성공. 독립 시험에서는 3개 정지 실패가 남았다. |
| 장애물 앞 급감속 | 장애물을 보았다는 이유만으로 일괄 저속 제한하던 방식 대신, 우회 경로의 곡률·제동거리로 연속적인 속도 계획을 만든다. 평균 가속도와 순간 PID 오차를 혼동한 변환도 수정했다. | 물리 동역학 검사와 실제 속도 그래프를 첨부했다. 긴급 상황에서는 충돌 방지 제동을 우선한다. |
| 급코너 벽 충돌 | 단순 자전거 운동 예측을 프로젝트 PID·타이어 동역학·RK4 예측으로 교체했다. 지도, 추정 사각형, 실제 라이다 관측점으로 전체 차체 경로를 검사한다. 실제 조향각과 센서의 시각도 맞췄다. | 최종 독립 시험에서는 벽 충돌 0회. 안전 제어를 제외한 모델에는 벽 충돌 2개가 남았다. |
| 직선 가속이 약함 | 최고 목표 3.0m/s와 명령 증가량 0.25m/s 제한에 묶인 이전 구조를 교체했다. 정상 제어는 주기 평균 가속·감속과 변화율을 제한하고, 새 MPCC 주행에서 빠른 직선 가속을 수집했다. | 실제 최고 {runtime['max_speed_mps']:.2f}m/s, 직선 평균 {runtime['straight_mean_mps']:.2f}m/s. 전 구간의 가속도 95백분위는 {runtime['actual_acceleration_p95_mps2']:.2f}m/s²로, 모든 가속 지표가 좋아졌다고 주장하지 않는다. |
| 전체 속도가 느림 | MPCC 목표 최고 4.2m/s, 곡률과 제동거리를 반영한 코너 감속을 사용한다. 계산 지연 동안 오래된 명령으로 물리 상태가 진행하지 않도록 시뮬레이터 제어 단계를 맞췄다. | 최종 독립 시험의 완주 평균 랩타임 {protected['successful_mean_lap_sim_s']:.2f}초. 실제 고정 배치 평균 {runtime['mean_lap_physics_s']:.2f}초. |

## 교사와 새 학습 데이터

맵은 `stack_master/maps/hangar_1905_v0`다. 새 학습용 주행 80개 중 충돌·후진 없이 완주한 {len(train_ok)}개를 사용했다. 기본 주행 {basic}개와 정적 장애물 회피 {avoid}개이며, 모두 새 동역학 기반 교사로 생성한 직접 시뮬레이션 녹화다. 미러 증강 전 13,216개, 증강 후 {training['samples_after_mirror']:,}개 제어 표본이다. 실패한 교사 주행 14개와 실패한 교사 제어 단계는 정상 정답으로 사용하지 않고 원본을 보관한다.

장애물은 1~3개를 배치하며 위치, 좌우 변위, 방향을 무작위로 바꿨다. 일부는 급코너 출구 0.35~2.5m 구간에 집중한다. 교사와 모델 모두 라이다로 관측한 장애물을 사용한다. 숨겨진 장애물 실제 좌표는 장면 생성과 통과 방향 정답 표시에만 사용하고 추론이나 경로 제어에 전달하지 않는다.

검증은 92000~92015의 16개 배치다. 검증 회귀 손실에는 완주한 교사 15개를 사용하지만, 폐루프 평가는 실패한 교사 배치도 포함한 16개 전체에서 수행한다. 최초 시험 93000~93023은 실패 원인을 확인하고 제어기를 개선한 뒤 개발 자료로 구분했다. 최종 독립 시험은 별도로 생성한 94000~94023이다. 이 24개는 학습과 최종 모델 선택에 사용하지 않았다. 그 뒤 시험한 추가 여유 공간 보정은 검증 완주율을 높이지 못하고 랩타임이 나빠져 적용하지 않았다. 현재 제어기 해시는 최종 독립 시험의 코드와 동일하다.

교사는 전통적인 IPOPT 기반 MPCC를 그대로 실행한 것이 아니다. 윤곽 오차, 진행 방향 오차, 진행량 비용을 사용하는 샘플링 방식의 MPC/MPCC 계열 구현이다. CEM으로 후보를 탐색하고 프로젝트의 PID·타이어 동역학·RK4로 앞의 2초를 예측한다. `sampled_mpcc.py`에 실제 구현이 있다. 기존 V4 가중치나 이전 실험의 녹화 데이터는 학습에 사용하지 않았다.

## 모델과 입력·출력

새 시간 이력 CNN/GRU 모델은 {training['parameters']:,}개 파라미터를 갖는다. 기존 TinyLidarNet을 그대로 파인튜닝한 모델이 아니다. 라이다 4프레임의 거리·전경 입력 `(4, 541, 2)`와 96개 보조 입력을 받는다. 보조 입력은 레이스라인·지도 폭, 속도·조향 명령·실제 조향각·회전 속도·추정 미끄러짐·가속도·제어 간격, 관측한 장애물과 그 방향이다. 지도와 시뮬레이션 위치 정보를 사용하므로 순수 라이다만 사용하는 모델과 동일한 조건이 아니다.

기본·좌회피·우회피 3개 분기에 대해 각각 다음 6개 시점의 조향·속도 명령을 예측하고, 분기 확률 3개를 출력한다. 총 출력은 39개다. 좌회피와 우회피 명령을 확률 평균하지 않고 한 분기를 선택한다.

현재 조향 보정은 출력 ×0.72rad와 물리 단위를 맞춘 기본 조향의 합이다. 미래 5개 조향은 출력 ×0.36rad, 속도는 출력 ×4.5m/s다. 속도 출력은 음수가 되지 않는다. 실제 조향 명령은 ±0.36rad, 제어 간 변화는 0.16rad로 제한한다. 정상 목표 평균 가속도는 최대 5m/s², 감속도는 최대 3.5m/s²이며 가속도 변화도 제한한다. 긴급 제동은 승차감 제한보다 충돌 방지가 우선이다. EMA는 사용하지 않는다. 세부 단위는 `input_output_contract.json`에 있다.

교사가 사용하는 실제 조향각을 모델 입력에도 추가했다. 라이다·위치·실제 조향각·물리 시간은 같은 헤더 시각으로 맞춘다. 장애물 배치나 차량 위치가 바뀌면 관측 기억과 제어 계획을 초기화한다.

## 새 독립 시험 24개

| 항목 | 신경망과 물리 출력 제한 | MPCC 보정 포함 |
|---|---:|---:|
| 완주 | {raw['completions']}/24 | {protected['completions']}/24 |
| 24초 이내 무충돌·무정지 완주 | {raw['fast_goal_successes']}/24 | {protected['fast_goal_successes']}/24 |
| 벽 충돌 | {raw['wall_collisions']} | {protected['wall_collisions']} |
| 장애물 충돌 | {raw['obstacle_collisions']} | {protected['obstacle_collisions']} |
| 정지 또는 시간 초과 | {raw['stops_or_timeouts']} | {protected['stops_or_timeouts']} |
| 완주한 경우 평균 랩타임 | {raw['successful_mean_lap_sim_s']:.2f}초 | {protected['successful_mean_lap_sim_s']:.2f}초 |
| 평균 레이스라인 오차 | {raw['mean_abs_raceline_error_m']:.3f}m | {protected['mean_abs_raceline_error_m']:.3f}m |
| 평균 직선 속도 | {raw['mean_straight_speed_mps']:.2f}m/s | {protected['mean_straight_speed_mps']:.2f}m/s |
| 제어 시간 95백분위의 배치 평균 | {raw['mean_control_latency_p95_ms']:.1f}ms | {protected['mean_control_latency_p95_ms']:.1f}ms |

완주한 경우의 랩타임 평균에는 실패 배치가 빠져 있다. 모델 자체 성능은 16/24이며, 21/24는 MPCC 보정이 포함된 결과다. 두 결과를 혼동하면 안 된다. 이전 버전의 시험 배치와 개수가 달라 성공률을 단순한 동일 조건 개선율로 비교하지 않는다.

추가로 검증 배치 4개를 각각 연속 3바퀴씩 평가했다. 총 12바퀴에서 충돌·정지·위치 재설정이 없었고 평균 랩타임은 {cr['successful_mean_lap_sim_s']:.2f}초였다. 이는 검증 배치 일부이며 독립 시험 전체의 성공을 뜻하지 않는다.

## 실제 RViz 주행과 시뮬레이션 시간

결과를 저장한 실제 세션은 물리 시간 {runtime['physics_duration_s']:.1f}초, {runtime['full_continuous_laps']}바퀴다. 최고 속도 {runtime['max_speed_mps']:.2f}m/s, 평균 직선 속도 {runtime['straight_mean_mps']:.2f}m/s, 급코너 평균 속도 {runtime['tight_corner_mean_mps']:.2f}m/s였다. 장애물이 없는 구간의 평균 레이스라인 오차는 {runtime['clear_road_mean_abs_error_m']:.3f}m였다. 안전 보정은 제어 단계의 약 {interventions:.1f}%에서 작동했으므로 현재 실행은 신경망 단독 제어가 아니라 혼합 제어다.

차량 물리식, 충돌 판정, 0.01초 물리 간격과 맵은 변경하지 않았다. 다만 최종 화면은 제어 명령 하나당 물리 단계 10개를 진행하는 동기화 방식이며 단조 시계로 실제 시간 속도를 목표로 진행한다. 제어 계산 중 오래된 명령으로 차량이 계속 진행하던 비동기 실행과는 다르다. 계산이 오래 걸리면 시뮬레이션이 늦어질 수 있다. 랩타임은 물리 시간이며 실차 지연 성능을 검증한 결과가 아니다.

초기 비동기 ROS 시험에서는 충돌과 정지가 재현됐다. 이를 숨기지 않고 개발 로그에 남겼다. 센서 시각 정렬, 실제 조향각 입력, 제어 단계 동기화 이후의 최종 세션만 위 장시간 결과에 포함했다. 출발 시 위치 설정 1회는 별도로 표시했고, 출발 이후 위치 재설정은 0회였다.

최종 모델의 전체 라이다 ROS 원본 녹화는 마지막 실행 승인 단계가 중단되어 생성되지 않았다. 대신 실제 제어 디버그 로그를 저장한 `final_runtime_controls.jsonl`과 요약을 보존했다. 이를 라이다 원본 ROS 녹화라고 표시하지 않는다. 교사의 별도 실제 ROS 원본은 `ros_teacher_synced.zip`에 있다. 이 녹화는 물리 시간 {ros['physics_duration_s']:.1f}초, 경로 순환 {ros['full_path_traversals']}회, 충돌 이벤트 {ros['collision_events']}회, 위치 재설정 {ros['pose_resets']}회이며 새 모델 학습 데이터로 사용했다고 주장하지 않는다.

## 동역학·단위 검증

예측기에 사용한 PID/RK4 10단계와 실제 프로젝트 시뮬레이터 10단계의 상태 최대 오차는 {physics['pid_rk4_simulator_10step_max_state_error']:.1f}였다. 목표 평균 감속도 -2m/s²를 요청한 검사에서 이전 변환의 실측 평균은 {physics['old_actual_mean_deceleration_mps2']:.3f}m/s², 수정 변환은 {physics['new_actual_mean_deceleration_mps2']:.3f}m/s²였다. 주기 변환은 연속 근사이므로 실제 이산 PID와 완전히 같은 수치라고 주장하지 않고 측정 결과를 제시한다.

## 남은 문제

독립 시험의 3개 배치에서 충분한 전진 경로를 만들지 못해 안전 정지했다. 이를 위치 재설정이나 임의 최소 속도로 숨기지 않았다. 모델 단독으로는 벽 충돌 2개와 장애물 충돌 6개가 남아 있다. 급코너의 좁은 통로에서 계획의 전진 가능성을 더 일찍 판단하고, 교사가 완주하지 못한 배치에서도 안정적인 교사 정답을 만드는 추가 개선이 필요하다. 모든 정지·충돌 문제가 해결됐다는 보고는 하지 않는다.

## 파일과 실행

- `TLN_DYNAMIC_V5_MOE96.keras`: 현재 실행 중인 모델.
- `random_initialization.keras`: 같은 새 학습 실행의 무작위 초기값.
- `fresh_recordings.zip`: 이번 작업에서 수집한 144개 원본 에피소드와 분리 정보. 최초 시험 24개는 개발용, 최종 24개는 별도 시험용이다.
- `source_and_results.zip`: 선택된 제어기와 소스, 지도·차량 설정, 평가와 실제 제어 로그.
- `ros_teacher_synced.zip`: 교사의 실제 ROS 원본 녹화.
- `actual_speed_profile.png`, `heldout_results.png`: 실제 속도와 독립 시험 그림.
- `SHA256SUMS`: 로컬 파일 검증값.

RViz 실행은 `ofc/ofc/inference_node.py`의 기존 모델 교체가 아니라 실험용 ROS 노드 `tln_fresh_v5`를 사용한다. 현재 모델 경로는 `experiments/tln_fresh/revision_dynamic_v5/models_moe96/best_regression.keras`다. 같은 맵의 기존 ROS 환경을 먼저 불러와야 한다. 시뮬레이터는 `timed_bridge.py`의 `synchronous_control:=true`, `paced_control:=true` 설정으로, 제어기는 `live.py --mode student --model models_moe96/best_regression.keras`로 실행한다. 여러 제어기를 동시에 `/drive`에 발행하지 않도록 한다.

Drive 저장 위치: `ADP/TLN_FRESH_hangar_1905_v0/dynamic_v5_20261028/`.
'''
assert not any('\u3400'<=c<='\u9fff' for c in report)
(out/'TLN_DYNAMIC_V5_RESULT_KO.md').write_text(report)
shutil.copy2(root/'models_moe96/best_regression.keras',out/'TLN_DYNAMIC_V5_MOE96.keras');shutil.copy2(root/'models_moe96/random_initialization.keras',out/'random_initialization.keras')
for n in ['training_moe96.json','dataset_manifest_moe96.json','final_runtime_snapshot.json','physics_verification.json','ros_teacher_audit.json','selection_final.json','evaluation_best_regression_test_raw_protected_1lap_final_holdout_summary.json','evaluation_best_regression_val_protected_1lap_rawpoints_summary.json','evaluation_best_regression_val_protected_3lap_final_continuous_summary.json']:shutil.copy2(results/n,out/n)
contract={'model_sha256':test['model_sha256'],'input_shapes':[[4,541,2],[96]],'features':{'0:16':'8 local route XY points /5m','16:32':'8 connected left/right map widths /3m','32':'speed /4.5mps','33':'previous steering command /0.36rad','34:66':'4x8 perceived-object descriptors; no GT','66':'yaw rate /5radps','67':'previous speed command /4.5mps','68:76':'route XY at 6,7,8,9m /5m','76:84':'future connected map widths /3m','84':'estimated slip /0.25rad','85':'observed mean acceleration /6mps2','86:94':'4x[cos4theta,sin4theta] fitted object orientation relative to body','94':'measured steering /0.4rad','95':'control interval /0.1s'},'output_shape':[39],'outputs':'3x12 action values then 3 side probabilities: clear,left,right','per_branch':'6 pairs: first steering residual x0.72rad + nominal, future steering x0.36rad; all speed x4.5mps','pretrained_weights':False,'ema':False,'physical_dt_s':.01,'control_dt_s':.1};(out/'input_output_contract.json').write_text(json.dumps(contract,indent=2))
versions={}
for n in ['numpy','tensorflow','scipy','numba','casadi','gymnasium']:
 try:versions[n]=importlib.metadata.version(n)
 except importlib.metadata.PackageNotFoundError:versions[n]='system package'
(out/'environment.json').write_text(json.dumps(versions,indent=2))
archives={'fresh_recordings.zip':list((root/'data').glob('*'))+list((root/'prepared').glob('*'))+list((root/'prepared_moe').glob('*'))+[results/'dataset_manifest_moe96.json',results/'feature_preparation.json',results/'moe_labels.json'],'ros_teacher_synced.zip':list((root/'ros_teacher_synced').rglob('*')),'source_and_results.zip':list(root.glob('*.py'))+list(root.parent.glob('*.py'))+list(results.glob('*'))+list((root/'source_snapshots').rglob('*.py'))}
for name,paths in archives.items():
 with zipfile.ZipFile(out/name,'w',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
  for p in paths:
   if p.is_file() and p.name!='live_v5.jsonl':z.write(p,p.relative_to(workspace))
  if name=='source_and_results.zip':
   for p in [workspace/'stack_master/config/SIM/sim_params.yaml',*list((workspace/'stack_master/maps/hangar_1905_v0').glob('*'))]:
    if p.is_file():z.write(p,p.relative_to(workspace))
 print('PACKAGED',name,(out/name).stat().st_size,flush=True)
with (out/'SHA256SUMS').open('w') as f:
 for p in sorted(out.iterdir()):
  if p.is_file() and p.name!='SHA256SUMS':f.write(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.name+'\n')
print('FINAL FILES',len(list(out.iterdir())),flush=True)
