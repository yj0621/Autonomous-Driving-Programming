# 급코너 출구 데이터와 새 모델 재현

맵은 `hangar_1905_v0`다. 작업 경로는 `/home/misys/forza_ws/race_stack`다.

## 데이터

`record_corners.py --split train|val|test`는 직접 물리 시뮬레이터를 실행해 LiDAR 원본, 상태, 입력 특징, 전문가 명령과 배치를 기록한다. 코너 출구 회차는 11 m 구간 시험이며 전체 랩이 아니다. `mpcc`와 `basic` 회차는 55.2902 m 전체 랩이다. 실패한 회차도 보존하지만 학습에서는 제외한다.

`record_corner_dagger.py --split train --limit 24`는 이번 작업에서 처음부터 만든 학생 모델을 행동 혼합에 사용한다. 네 번 중 한 번은 전문가 65%, 학생 35%를 섞고 조향에 작은 잡음을 넣는다. 저장된 학습 정답과 실제 실행 명령은 각각 `commands`, `executed_commands`로 구분한다. 이 과정에서 기존 예전 TLN 모델은 사용하지 않았다.

`prepare_corners.py`는 원시 기록으로부터 온라인과 같은 LiDAR/지도 기반 기억 특징을 재계산한다. 장애물 정답 좌표는 학생 입력을 만드는 데 사용하지 않는다. 학습에 사용한 정확한 목록은 각 모델의 `training_corner.json`에 있다. 데이터 묶음 전체를 사용했다고 가정하지 않는다.

## 학습

```bash
python3 experiments/tln_fresh/prepare_corners.py
python3 experiments/tln_fresh/train_corner.py \
  --dataset-manifest corner_revision/results/dataset_manifest.json \
  --output-dir corner_revision/reproduction --epochs 70
```

학습 스크립트는 무작위 초기화부터 시작한다. 과거 TLN이나 이번 작업의 중간 모델 가중치를 불러와 학습을 이어가지 않는다. DAgger 녹화기의 모델 불러오기는 데이터 수집 행동을 위한 것이며 초기 가중치로 사용하는 것이 아니다.

## 화면 주행과 녹화 분리

기본 속도 확인은 도메인 0의 실시간 bridge와 기본 레이스라인 제어기로 수행했다. 이후 사용자 요청으로 최종 학생 모델과 경로 보호로 전환했다. 현재 시연에는 정적 장애물 3개가 있고 벽 충돌이 확인됐다. 아래 기본주행 명령은 속도 확인 재현용이다. 학생 시연에는 같은 터미널에서 기본 제어기를 중복 실행하지 않고 다음 명령을 대신 사용한다.

```bash
python3 experiments/tln_fresh/live.py --mode student --revision corner \
  --model /home/misys/forza_ws/race_stack/experiments/tln_fresh/corner_revision/results/final_model_corner.keras
```

장애물 시연은 다른 터미널에서 `python3 experiments/tln_fresh/show_static_three.py`로 실행한다. 기존 장애물 발행기를 중복 실행하지 않는다.

```bash
source /opt/ros/humble/setup.bash
source /home/misys/forza_ws/race_stack/install/setup.bash
python3 experiments/tln_fresh/timed_bridge.py --ros-args -r __node:=bridge \
  --params-file experiments/tln_fresh/corner_revision/bridge_params.yaml \
  -p synchronous_control:=false
```

별도 터미널에서 다음을 실행한다. 같은 도메인에 기존 bridge나 다른 `/drive` 발행기가 있으면 중복 실행하지 않는다.

```bash
python3 experiments/tln_fresh/live.py --mode basic --revision corner --cruise 2.2
```

학습용 ROS 녹화는 별도 도메인 42에서 동기 모드로 실행한다. 전문가 계산이 끝날 때까지 물리 계산이 기다리므로 실제 경과 시간 기준으로 느리게 보인다. 동기 녹화의 물리 시간과 화면 주행의 실제 시간은 직접 비교하지 않는다.

```bash
export ROS_DOMAIN_ID=42
python3 experiments/tln_fresh/timed_bridge.py --ros-args -r __node:=bridge \
  --params-file experiments/tln_fresh/corner_revision/bridge_params.yaml \
  -p synchronous_control:=true
```

다른 터미널에서도 같은 도메인 42를 지정한 뒤 `record_ros_corners.py`를 실행한다. 기존 녹화를 덮어쓰지 않으므로 재수집 시에는 코드의 출력 경로를 새 경로로 지정한다. ROS bag은 진단 자료로 보존하고 자동으로 정상 학습 데이터에 합치지 않는다.

물리 계산, 차량 설정과 충돌 규칙은 기존 bridge에서 상속한다. `timed_bridge.py`는 실행 순서를 직렬화하고 실제 물리 시간 및 조향각을 `/sim/physics_state`로 발행한다. 동기 옵션은 명령당 물리 계산 10회로 제어 간격을 맞춘다. 일반 실시간 옵션에서는 계산을 기다리지 않는다.

## 단위와 한계

학생 입력은 `(541,6)` 거리/전경 3프레임과 42차원 지도·상태·장애물 후보 특징이다. 장애물 관측은 현재 위치로 보정해 최대 30개 제어 단계 동안 유지한다. 출력은 보조 경로 오프셋 3개, 속도 비율, 조향 보정이다. 조향 보정은 0.72 rad, 속도는 2.8 m/s로 복원하며 실제 조향은 ±0.36 rad로 제한한다. EMA와 양의 최저 주행 속도는 사용하지 않는다.

보호 제어는 별도 기하 경로 계획과 제동을 포함한다. 보호 제어의 성능을 신경망 단독의 성능으로 보고하지 않는다. 현재는 단일 지도 및 GT 위치 기반 시뮬레이션 결과이며 실제 차량 성능을 확인한 결과가 아니다.
