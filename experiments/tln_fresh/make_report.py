"""Build the Korean report and standalone figures from measured results only."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/tln_fresh_mpl')
from pathlib import Path
import json,csv,hashlib,ast
from collections import Counter
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from world import World,MAP,ROOT
root=Path(__file__).parent;out=root/'deliverables';out.mkdir(exist_ok=True);results=root/'results';w=World()
summary=json.loads((results/'evaluation_strict_final_test_summary.json').read_text());rows=json.loads((results/'evaluation_strict_final_test.json').read_text());training=json.loads((results/'training_v2.json').read_text());selection=json.loads((results/'final_selection.json').read_text());failures=json.loads((results/'failure_analysis.json').read_text())
model=results/'final_model.keras';assert hashlib.sha256(model.read_bytes()).hexdigest()==summary['model_sha256']
columns=['controller','scenario_mode','seed','success','collision','sim_time_s','progress_m','mean_abs_raceline_error_m','max_abs_raceline_error_m','controller_ms_p95','path_interventions']
with (out/'evaluation_trials.csv').open('w') as f:
 writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader();writer.writerows({k:r[k] for k in columns} for r in rows)
lines=['| 방식 | 시험 | 무충돌 완주 | 벽 충돌 | 장애물 충돌 | 정지·미완주 | 성공 시 평균 랩타임 | 평균 경로 오차 |','|---|---:|---:|---:|---:|---:|---:|---:|']
for name,title in [('neural','새 신경망 단독'),('protected','새 신경망 + 보호 제어')]:
 r=summary['summary'][name];lines.append(f"| {title} | {r['trials']} | {r['successes']} | {r['wall_collisions']} | {r['obstacle_collisions']} | {r['timeouts']} | {r['clean_lap_mean_s']:.3f} s | {r['mean_abs_raceline_error_m']:.3f} m |")
group=['| 방식 | 기본 주행 완주 | 회피 배치 완주 | MPCC 배치 완주 |','|---|---:|---:|---:|']
for name,title in [('neural','새 신경망 단독'),('protected','보호 제어 적용')]:
 counts=[]
 for mode in ['basic','avoidance','mpcc']:
  rr=[r for r in rows if r['controller']==name and r['scenario_mode']==mode];counts.append(f"{sum(r['success'] for r in rr)}/{len(rr)}")
 group.append('| '+title+' | '+' | '.join(counts)+' |')
log=json.loads((root/'data/ros_fast/import_summary.json').read_text());best_epoch=int(np.argmin(training['history']['val_loss']))+1
report=f'''# 새 TLN 학습·고속 녹화·최종 평가 결과

**기존 TLN 가중치 없이 새 모델을 만들고 학습·평가했다. 다만 모든 장애물 배치에서 충돌 없이 빠르게 완주한다는 목표는 아직 달성하지 못했다.**

## 1. 최종 결과

{chr(10).join(lines)}

{chr(10).join(group)}

시험은 학습에 사용하지 않은 배치 시드 3000~3007, 맵 `hangar_1905_v0`에서 수행했다. 기본 주행 2회는 같은 시작 조건의 반복이고, 나머지 6회는 서로 다른 장애물 배치다. 모든 시험은 pose 리셋 없이 진행했으며, 경로 길이 **{w.length:.4f} m 이상**을 이동하고 충돌이 없어야 완주다. 정지한 시험을 성공으로 세지 않았다.

위 랩타임은 오프라인 물리 시뮬레이터의 고정 시간 간격 기준이다. 성공한 회차만의 평균이므로 완주 횟수와 함께 읽어야 한다. 실제 경과 시간을 사용하는 ROS 평가기의 랩타임과 직접 섞어 비교하지 않는다. 경로 오차에는 의도적인 장애물 회피 구간도 포함된다.

## 2. 기존 모델을 사용했는가

사용하지 않았다. 최종 모델의 Conv1D·Dense 층은 새로 초기화했다. 경로 보정과 조향 보정 출력은 0으로 시작하도록 초기화했으며 기존 모델의 가중치를 복사하지 않았다. `train_v2.py`에는 `load_model`, `load_weights`, `set_weights` 호출이 없다.

- 초기화 난수 시드: 93271.
- 전체 학습 실행: {training['epochs']}회. 그중 검증 손실이 가장 낮은 {best_epoch}회 체크포인트 선택.
- 파라미터 수: {training['parameters']:,}개.
- 최종 모델 SHA256: `{summary['model_sha256']}`.
- 무작위 초기 가중치 배열 SHA256: `{training['initial_weights_sha256']}`.

이번에 새로 만든 모델에 수정 데이터로 18회 추가 학습하는 실험도 했지만, 검증 완주율이 초기 새 모델의 5/8에서 4/8로 낮아 최종 모델로 선택하지 않았다. 추가 학습 실험에서도 예전 TLN은 사용하지 않았다. 최종 선택에는 시험 배치 결과를 사용하지 않았다.

## 3. 직접 수집한 데이터

F1TENTH 시뮬레이터를 직접 구동해 기본 주행·장애물 회피·MPCC 주행을 새로 녹화했다. 원시 LiDAR, 상태, 제어 명령, 실제 배치, 궤적과 충돌 판정을 저장했다. 기본 수집 후보 32회 중 완주한 31회만 학습에 사용했다. 검증 회귀 오차는 전문가가 완주한 6회에서 계산했고, 최종 폐루프 시험에는 전문가가 정지했던 배치도 포함했다.

직선 속도를 1.3 m/s에서 **2.2 m/s로 높인 ROS 기본 주행**도 직접 녹화했다. 이 bag에서 충돌·리셋이 없는 한 바퀴 상당 구간 2개, **{sum(e['samples'] for e in log['full_lap_episodes'])}개 샘플**을 시간 정렬해 학습에 추가했다. GT odometry 속도 상태의 평균은 약 1.96~1.98 m/s, 최대 약 2.2 m/s였다. 초기 전환 구간의 벽 충돌은 평가 원본에 남겨 두었고 이 학습 구간에는 포함하지 않았다.

최종 학습은 새 회차 {len(training['train_episodes'])}개와 좌우 반전 증강 후 {training['training_samples_after_mirror']:,}개 샘플을 사용했다. 최종 데이터 목록과 파일 해시는 `training_v2.json`에 있다. 이후 추가로 모은 수정 데이터는 별도 실험 자료이며 최종 선택 모델에는 사용하지 않았다.

실시간 MPCC bag은 충돌·복구가 있는 진단 자료다. 학습용 정상 데이터와 동일하게 취급하지 않는다. 메시지가 0개였던 첫 ROS 녹화는 사용하지 않았다.

## 4. 새 모델과 제어 구조

```mermaid
flowchart LR
 A[LiDAR 3프레임] --> C[새 CNN]
 B[지도·지역 경로·속도·이전 조향] --> C
 D[LiDAR와 지도 기반 전경·좌우 여유] --> C
 C --> E[경로 보정·목표 속도·조향 보정]
 E --> F[레이스라인 기준 조향과 결합]
 F --> G[선택적 경로 검사·제동 보호]
 G --> H[Ackermann 명령]
```

입력은 거리 3프레임과 전경 마스크 3프레임을 합친 `(541, 6)`, 그리고 42차원 경로·상태·장애물 후보 정보다. 전경은 LiDAR 반사점과 지도·현재 위치로 계산하며, 학생 모델에 시뮬레이터 장애물 정답 좌표를 입력하지 않는다. 다만 지도와 위치를 사용하므로 원래의 LiDAR 단독 모델과 다른 지도 보조 TLN 계열 모델이다.

MPCC는 가상 진행률, contour/lag 오차, 차량 동역학 근사, 경계·장애물 제약으로 전문가 행동을 계산한다. 학생보다 정밀한 장애물 정보를 사용하는 정보 차이는 남아 있다. 현재 시험은 GT 위치·속도를 사용했으므로 위치 추정 오차가 있는 조건의 성능은 확인하지 않았다.

목표 속도는 m/s, 조향은 rad로 맞췄다. 조향은 기준 경로 조향에 학습 보정을 더해 ±0.36 rad로 제한한다. 속도는 0~2.8 m/s 범위와 곡률 제한을 사용한다. 이전의 3 m/s 최저속도와 EMA는 사용하지 않는다. 보호 제어는 별도 경로·제동 보정이며 그 성능을 신경망 단독의 성능이라고 보고하지 않는다.

## 5. 확인한 개선과 남은 실패

- 새 신경망은 최종 기본 주행 2회 모두 충돌 없이 완주했고, 최종 전체 시험에서 벽 충돌은 없었다.
- 신경망 단독은 장애물 충돌 3회가 남았다. 보정된 조향이 기본 추종과 달라지는 거리와 마지막 횡방향 여유를 실패 분석 파일에 기록했다.
- 보호 적용 시험은 충돌 0회였지만 4회는 정지해 완주하지 못했다. 제동으로 충돌을 막는 것과 회피를 완료하는 것은 다른 결과다.
- 보호 경로가 한정된 탐색 안에서 나오지 않았다고 해서 물리적으로 모든 경로가 불가능하다고 단정할 수 없다.
- 관측 장애물의 중심·경계 추정 오차, 좁은 통과 공간, 학습 상태와 실제 폐루프 상태의 차이가 남은 원인 후보다. 개별 기여도를 확정한 결과는 아니다.

추론·명령 계산 시간의 시험별 95백분위 평균은 신경망 단독 {summary['summary']['neural']['controller_p95_ms_mean']:.2f} ms, 보호 적용 {summary['summary']['protected']['controller_p95_ms_mean']:.2f} ms다. 이는 입력 특징 생성과 센서·통신·작동기 지연을 포함한 전체 지연이 아니다.

## 6. 실시간 주행과 시간 기준

화면에서 실행한 고속 기본 주행은 신경망이 아닌 기본 레이스라인 제어기로 수집했다. 최종 새 모델의 실시간 ROS 장애물 완주 성능은 아직 확정하지 않았다.

이전 실시간 실험에는 중복으로 남은 장애물 발행기·위치 추정 노드와 센서 발행 중단 문제가 있었다. 해당 세션과 정상 오프라인 시험을 구분했다. 현재 물리 계산 주기와 실제 경과 시간도 다를 수 있어, ROS 기본 주행 랩타임과 오프라인 시간은 별도로 해석해야 한다.

## 7. 제공 파일

- `final_model.keras`: 최종 새 모델.
- `BASIC_FAST_REPORT_KO.md`, `basic_fast_recording.zip`: 고속 기본 주행 보고서와 원본 ROS bag.
- `new_dataset.zip`: 새 학습·검증·시험 기록과 파생 입력.
- `source_and_results.zip`: 코드, 맵·차량 설정 사본, 학습 이력, 모든 최종 시험 궤적, 실패 분석과 재현 설명.
- `evaluation_trials.csv`: 회차별 최종 수치.
- `evaluation_overview.png`, `test_trajectories.png`: 최종 결과 그림.

현재 결과는 연구용 실험 결과다. **모든 장애물 배치를 무충돌로 빠르게 완주하는 모델로 승인할 근거는 아직 부족하다.**
'''
(out/'FINAL_REPORT_KO.md').write_text(report)
# Standalone plots for presentation.
fig,ax=plt.subplots(figsize=(7,4));names=['neural','protected'];x=np.arange(2);bottom=np.zeros(2)
for key,label,color in [('successes','Completed','#2a9d8f'),('obstacle_collisions','Obstacle collision','#e76f51'),('wall_collisions','Wall collision','#b23a48'),('timeouts','Stopped / timeout','#d9a441')]:
 vals=[summary['summary'][n][key] for n in names];ax.bar(x,vals,bottom=bottom,label=label,color=color);bottom+=vals
ax.set_xticks(x,['New neural model','With protection']);ax.set_ylabel('Trials');ax.set_ylim(0,8);ax.legend(fontsize=8);fig.tight_layout();fig.savefig(out/'evaluation_overview.png',dpi=180);plt.close(fig)
scenarios=sorted({r['seed'] for r in rows});fig,axes=plt.subplots(2,4,figsize=(12,10));s=np.linspace(0,w.length,1000);line=w.xy(s)
for ax,seed in zip(axes.ravel(),scenarios):
 ax.plot(line[:,0],line[:,1],'k--',lw=.7,label='Raceline');scenario=next(r for r in rows if r['seed']==seed);r=json.loads((root/'data'/f"test_{scenario['scenario_mode']}_{seed}.json").read_text())
 for b in r['obstacles']:ax.add_patch(plt.Circle((b['x'],b['y']),.21,color='#555555',alpha=.7))
 for mode,color in [('neural','#3178b7'),('protected','#d38024')]:
  rr=next(r for r in rows if r['seed']==seed and r['controller']==mode);t=np.array([x['state'] for x in rr['trace']]);ax.plot(t[:,0],t[:,1],color=color,lw=.9,label=mode)
  if not rr['success']:ax.scatter(t[-1,0],t[-1,1],color=color,marker='x',s=24)
 ax.set_title(f"{seed} / {scenario['scenario_mode']}",fontsize=9);ax.set_aspect('equal');ax.tick_params(labelsize=6)
axes[0,0].legend(fontsize=6);fig.tight_layout();fig.savefig(out/'test_trajectories.png',dpi=150);plt.close(fig)
# Verify user-facing documents contain no Japanese or CJK ideographs.
for p in [out/'FINAL_REPORT_KO.md',root/'README_KO.md',out/'BASIC_FAST_REPORT_KO.md']:
 assert not any('\u4e00'<=c<='\u9fff' or '\u3040'<=c<='\u30ff' for c in p.read_text()),p
print('최종 보고서와 그림 생성 완료')
