"""Package measured outcomes, including failures; no success claim from resets."""
from pathlib import Path
import json,zipfile,hashlib,shutil,importlib.metadata
r=Path(__file__).resolve().parent;out=r/'deliverables';out.mkdir(exist_ok=True)
a=json.loads((r/'results/basic_ros_audit.json').read_text());test=json.loads((r/'results/evaluation_candidate_epoch_20_selected_raw_protected_test_1lap_summary.json').read_text());continuous=json.loads((r/'results/evaluation_candidate_epoch_20_continuous_protected_val_3lap_summary.json').read_text());train=json.loads((r/'results/training_mixture.json').read_text());raw=test['summary']['raw'];guard=test['summary']['protected']
report=f'''# 새 TLN 학습 및 직선 고속 데이터 녹화 결과

## 현재 결론

직선 고속, 코너 진입 전 감속 데이터를 새로 녹화하고 기존 가중치를 사용하지 않은 새 모델을 학습했다. 학습에 쓰지 않은 시험 배치 19개에서 안전 제어 포함 {guard['completions']}개를 완주했다. 벽 충돌 {guard['wall_collisions']}회와 정지 또는 시간 초과 {guard['stops_or_timeouts']}회가 남아 있다. 모든 장애물에 안정적인 최종 모델이라는 결론은 내릴 수 없다.

## 직선 고속 및 코너 감속 녹화

| 항목 | 실제 ROS 녹화 결과 |
|---|---:|
| 맵 | hangar_1905_v0 |
| 녹화 물리 시간 | {a['physics_time_span_s']:.1f}초 |
| 완전한 경로 순환 | 3회 |
| 충돌 / 위치 재설정 | 0회 / 0회 |
| 직선 평균 실측 속도 | {a['straight_curvature_below_0p08_mean_mps']:.3f}m/s |
| 급코너 평균 실측 속도 | {a['tight_corner_curvature_above_0p55_mean_mps']:.3f}m/s |
| 전체 평균 / 최고 실측 속도 | {a['mean_measured_speed_mps']:.3f} / {a['max_measured_speed_mps']:.3f}m/s |

직선은 곡률 0.08/m 미만, 급코너는 0.55/m 초과로 구분했다. 6m 앞 경로 곡률과 제동거리로 감속한다. 횡가속 제한 1.9m/s², 계획 제동 감속도 2.8m/s², 기본 목표 최고 속도 3.0m/s다. `ros_speed_profile.png`는 실제 ROS 속도와 명령을 표시한다. 이 녹화는 장애물 없는 기본 주행이다.

## 새 학습 데이터

성공한 새 주행 49개를 학습에 사용했다. 새 기본 주행 8개, 실제 ROS 기본 주행 3개, 무작위 장애물 회피 31개, MPCC 주행 7개다. 미러 증강 전 12,883개 제어 표본이다. 학습·검증은 에피소드 단위로 분리했다. 학습과 검증의 실패 교사 기록 13개는 성공 정답에서 제외하고 원본과 실패 원인을 보관했다.

장애물은 1~3개이며 레이스라인 전 구간의 무작위 위치, 좌우 변위, 회전을 사용했다. 일부 배치는 급코너 출구 0.5~2.6m 지점을 더 자주 뽑았다. 회피 모드도 장애물 관측 이후 MPCC 교사를 사용하고, 관측 전에는 새 기본 주행 제어기를 사용한다. 모든 회피 데이터를 순수 MPCC만으로 수집했다고 해석하면 안 된다.

기존 TLN 가중치와 과거 실험 데이터는 새 학습에 사용하지 않았다. `random_initialization.keras`는 이 학습 실행에서 새로 생성한 초기값이다. 최종 후보는 새 모델의 20번째 학습 단계 파일이며, 검증 결과로 선택하고 시험 결과로 다시 고르지 않았다. 검증 오차가 가장 작은 다른 모델은 실제 완주율이 낮아 선택하지 않았다.

## 입력과 출력 및 실제 제어

- 입력: 라이다 거리와 전경 표시 4프레임, 모양 `(4, 541, 2)`.
- 보조 입력: 지도 기반 레이스라인과 연결된 통로 폭, 속도, 이전 조향·속도, 회전 속도, 관측 장애물 기억과 9m 경로 미리 보기, 총 84개 값.
- 네트워크: 시간 축에 공유한 CNN과 GRU, 경로 및 상태 인코더. 기존 TinyLidarNet을 그대로 파인튜닝한 구조가 아니다.
- 출력: 조향 보정 1개, 부호 있는 속도 1개, 좌회피 경로 5개, 우회피 경로 5개, 좌·우·기본 주행 확률 3개.
- 출력 단위 복원: 조향 보정 ×0.72rad, 속도 ×3.2m/s, 경로 변위 ×1.2m.
- 후처리: 선택한 한쪽 경로를 추종하며 관측 가능한 회전 속도와 속도로 차체 미끄러짐과 조향 관계를 보정한다. 양쪽 경로를 확률 평균하지 않는다.
- 조향 제한 ±0.36rad, 제어 간 조향 변화 제한 0.16rad, 속도 증가 제한 제어 간 0.25m/s, 제어 간격 물리 시간 0.1초. EMA는 사용하지 않는다.
- 별도 안전 제어: 지도·라이다·관측 장애물에 대해 프로젝트의 PID/RK4 동역학으로 후보 명령과 제동 경로를 검사한다. 막힐 때 경로 수정, 감속 및 제한된 후진 복구를 사용할 수 있다.

교사와 학생 모두 지도 및 시뮬레이션 위치 정보를 사용한다. 장애물 실제 좌표는 장면 생성과 정답 표시에만 사용하며 모델 입력이나 회피 제어에 직접 전달하지 않는다. 물체가 가려진 뒤에도 정적 장애물 관측을 기억하도록 바꿨다. 지도와 위치 입력이 없는 순수 라이다 모델과 성능을 직접 비교하면 안 된다.

## 시험 결과: 학습에 쓰지 않은 19개 배치

| 항목 | 안전 제어 제외 | 안전 제어 포함 |
|---|---:|---:|
| 완주 | {raw['completions']}/19 | {guard['completions']}/19 |
| 충돌·정지 없이 32초 이내 완주 | {raw['fast_goal_successes']}/19 | {guard['fast_goal_successes']}/19 |
| 벽 충돌 | {raw['wall_collisions']} | {guard['wall_collisions']} |
| 장애물 충돌 | {raw['obstacle_collisions']} | {guard['obstacle_collisions']} |
| 정지 또는 시간 초과 | {raw['stops_or_timeouts']} | {guard['stops_or_timeouts']} |
| 완주한 경우 평균 랩타임 | {raw['successful_mean_lap_sim_s']:.2f}초 | {guard['successful_mean_lap_sim_s']:.2f}초 |
| 장애물 없는 구간 평균 레이스라인 오차 | {raw['clear_road_mean_abs_error_m']:.3f}m | {guard['clear_road_mean_abs_error_m']:.3f}m |
| 제어 전체 실행 시간 95백분위의 배치 평균 | {raw['total_control_p95_ms_mean']:.1f}ms | {guard['total_control_p95_ms_mean']:.1f}ms |

안전 제어 제외 결과에도 출력 단위 복원, 경로 추종, 속도·조향 제한은 들어간다. 안전 제어 포함 결과를 네트워크 자체의 성능으로 표시하면 안 된다. 랩타임은 고정 물리 시간이며 ROS 전송 지연은 포함하지 않는다. 완주한 경우의 평균이므로 실패를 제외한 선택 편향이 있다.

## 연속 주행

기본 주행 2개와 장애물 배치 2개에서 각각 3바퀴를 평가했다. 총 12바퀴에서 충돌·정지·위치 재설정 없이 완주했다. 평균 랩타임은 {continuous['summary']['protected']['successful_mean_lap_sim_s']:.2f}초였다. 이 4개는 검증 배치 일부이며, 시험 19개보다 좁은 조건이다. 전체 일반화 성공의 증거로 확대 해석하면 안 된다.

## 실제 확인한 원인과 남은 추정

실제 확인: 이전 교사의 장애물 기억과 학생의 짧은 기억 길이가 달랐다. 한쪽으로만 회피해야 하는 정답을 조향 회귀 하나로 학습하면 대응이 어렵다. 안전 제어의 불가능한 경로 판정과 후진 복구 상태에서 정지가 발생했다. 장애물 추정 상자가 차량 주변 빈 공간과 겹치는 사례가 있었다. 센서 콜백 지연에 따른 정지와 물리 시간·벽시계 시간 차이도 별도 문제였다.

실제 확인: 시뮬레이터 조향 PID는 각도 오차가 있으면 매 0.01초에 최대 조향 속도로 움직여 0.032rad 간격으로 각도가 변한다. 목표 각도가 그 사이에 있으면 반복 움직일 수 있다. 조향 명령을 이 간격에 맞추는 실험은 전체 검증 완주율을 개선하지 못해 선택하지 않았다. 여유 공간에 따른 추가 경로 보정도 충돌이 남아 선택하지 않았다.

추정: 급코너의 가려진 물체, 관측 물체 중심·크기 오차, 짧은 경로 추종 목표와 실제 차체 운동의 차이가 남은 실패에 기여한다. 각 요인의 독립적인 영향은 아직 분리 실험으로 확정하지 않았다.

## 남은 한계

- 시험에서 벽 충돌 1개와 정지 또는 시간 초과 3개가 남아 있다. 요청한 무충돌·무정지를 달성했다고 보고하지 않는다.
- 안전 제어를 제외한 모델 자체의 시험 완주율은 8/19다. 추가 학습과 교사 품질 개선이 필요하다.
- 실제 ROS 실행에서는 위치 및 회전 속도로 미끄러짐을 추정하지만, 오프라인 시험은 시뮬레이터 상태를 사용한다. 둘의 완전한 동등성은 검증되지 않았다.
- 정적 장면의 관측된 빈 공간을 기억한다. 장애물 배치나 차량 위치가 바뀌면 기억을 초기화해야 한다. 동적 장애물 검증은 하지 않았다.
- 실제 차량과 다른 맵에 대한 성능은 확인하지 않았다.

## 파일

`fresh_recorded_data.zip`: 새 원본 데이터, 분리 목록, 실패 기록. `ros_fast_basic.zip`: 직선 고속·코너 감속 ROS 원본. `tln_fresh_v4_candidate.keras`: 선택한 후보 모델. `random_initialization.keras`: 같은 실행의 무작위 초기값. `source_and_results.zip`: 소스·평가·학습 결과·지도 설정. `ros_speed_profile.png`: 실제 녹화 속도 그림. `SHA256SUMS`: 파일 검증값.

Google Drive 저장 위치: `ADP/TLN_FRESH_hangar_1905_v0/random_v4_20261028/`.
'''
live=json.loads((r/'results/candidate_three_ros_audit.json').read_text())
report+='\n## RViz 실제 실행 녹화 검증\n\n새 후보와 안전 제어를 적용하고 직선, 코너 출구, 복귀 구간에 0.3m 정적 장애물 3개를 배치했다. 닫힌 ROS 녹화의 물리 시간은 '+str(round(live['physics_time_span_s'],2))+'초이며, 완전한 경로 순환 2회, 충돌 메시지 0회, 위치 재설정 0회였다. 전체 평균 실측 속도 '+str(round(live['mean_measured_speed_mps'],3))+'m/s, 직선 평균 '+str(round(live['straight_curvature_below_0p08_mean_mps'],3))+'m/s, 급코너 평균 '+str(round(live['tight_corner_curvature_above_0p55_mean_mps'],3))+'m/s였다. 이 배치의 성공을 무작위 시험 전체의 성공으로 확대하지 않는다. 원본은 ros_candidate_three.zip, 요약은 candidate_three_ros_audit.json이다.\n'
report+='\n## 남은 벽 충돌의 명령 기록\n\n시험 83003에서 경로 탐색이 불가능하다고 판정한 뒤에도 안전 제어가 0.25m/s 전진 명령과 정지를 반복하며 벽 쪽으로 조금씩 진행했다. 마지막에는 벽 충돌이 발생했다. 이는 정지에서 전진을 복원하는 조건과 경로 가능 조건을 함께 제한해야 함을 보여준다. 같은 시험을 수정에 사용한 추가 진단은 새 독립 시험으로 계산하지 않는다.\n'
snapshot=json.loads((r/'results/live_validation_snapshot.json').read_text())
report+='\n## 추가 실행 관찰'+'\n\n결과 생성 시점의 RViz 관찰은 '+str(snapshot['layout_full_laps'])+'바퀴였다. 장면 활성화 뒤 충돌 '+str(len(snapshot['collisions_after_scene_activation']))+'회, 위치 재설정 '+str(len(snapshot['pose_resets_after_scene_activation']))+'회였다. 후진 복구 명령은 '+str(snapshot['reverse_command_steps'])+'개 제어 단계에서 발생했다. 후진을 정지로 표시하던 감시 코드의 부호 비교 오류를 확인하고 절댓값 비교로 수정했다. 후진 복구를 끊김 없는 전진 주행으로 계산하지 않는다.\n'
report+='\n## 벽 충돌 수정 진단\n\ncontroller_no_creep_experimental.py는 경로 탐색 실패 상태에서 정지 차량의 전진 복원을 금지한다. 이미 관찰한 83003을 재실행한 결과 벽 충돌은 없어졌지만 정지 실패로 바뀌었다. 따라서 완성된 해결책으로 선택하거나 실제 화면에 적용하지 않았다. 선택 모델과 실제 화면은 검증한 controller.py를 사용한다. 이 수정 진단은 시험 성적에 합산하지 않는다.\n'
(out/'TLN_V4_RESULT_KO.md').write_text(report)
shutil.copy2(r/'results/live_validation_snapshot.json',out/'live_validation_snapshot.json')
shutil.copy2(r/'results/candidate_three_ros_audit.json',out/'candidate_three_ros_audit.json')
for name in ['training_mixture.json','selection.json','basic_ros_audit.json','evaluation_candidate_epoch_20_selected_raw_protected_test_1lap_summary.json','evaluation_candidate_epoch_20_continuous_protected_val_3lap_summary.json']:shutil.copy2(r/'results'/name,out/name)
shutil.copy2(r/'models_mixture/random_initialization.keras',out/'random_initialization.keras')
packages={}
for name in ['tensorflow','numpy','scipy','numba','casadi','rclpy','gymnasium']:
 try:packages[name]=importlib.metadata.version(name)
 except importlib.metadata.PackageNotFoundError:packages[name]='ROS system package / see workspace'
(out/'environment.json').write_text(json.dumps(packages,indent=2))
with zipfile.ZipFile(out/'source_and_results.zip','w',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
 paths=list(r.glob('*.py'))+list(r.parent.glob('*.py'))+list((r/'results').glob('*'))+list((r/'source_snapshots').rglob('*.py'))
 for p in paths:
  if p.is_file():z.write(p,p.relative_to(r.parent.parent.parent))
 workspace=r.parents[2]
 for p in [workspace/'stack_master/config/SIM/sim_params.yaml',*list((workspace/'stack_master/maps/hangar_1905_v0').glob('*'))]:
  if p.is_file():z.write(p,p.relative_to(workspace))
with (out/'SHA256SUMS').open('w') as f:
 for p in sorted(out.iterdir()):
  if p.is_file() and p.name!='SHA256SUMS':f.write(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.name+'\n')
print('REPORT PACKAGED',len(list(out.iterdir())),flush=True)
