# Berlin MPCC 고속 TLN 학습 결과

## 지도와 수집 설정

`berlin-map.zip`의 맵 파일을 `stack_master/maps/berlin/`에 풀었다. 해상도는 0.04 m/pixel이며 `berlin.yaml`, `berlin.png`, `global_waypoints.json`, `speed_scaling.yaml`을 포함한다. 기존 실험의 기본 지도는 유지했고 `TLN_MAP_DIR` 환경 변수로 Berlin을 선택한다.

MPCC 크루즈 목표와 직선 속도 상한을 4.5 m/s로 설정했다. 급한 곡선에서는 횡가속도 2.1 m/s² 제한과 연속 제동 여유를 적용해 속도를 낮춘다. 차량의 최대 수집 속도는 4.498 m/s였다. 각 회차마다 시작 위치와 난수 시드를 바꾸고, 장애물 회차에는 1~3개의 정적 장애물을 새 위치에 배치했다. 장애물은 한 회차 중에는 고정되며 회차가 바뀔 때 다시 배치된다.

| 구분 | 회차 | 무장애물 | 정적 장애물 | MPCC 완주 |
|---|---:|---:|---:|---:|
| 훈련 | 36 | 12 | 24 | 36 |
| 검증 | 10 | 4 | 6 | 10 |
| 미사용 시험 | 12 | 4 | 8 | 11 |

시험 중 전문가가 벽에 충돌한 회차 1개도 보존했다. 학습에는 성공한 훈련·검증 회차만 사용했다. 전체 원시 LiDAR, 상태, 명령, 장애물 배치와 솔버 로그는 `data/`와 `results/collection_*.json`에 있다.

## 모델

`models_moe96/best_regression.keras`는 4프레임 LiDAR와 경로·장애물 특징을 받는 TLN MoE96 모델이다. 무작위 초기화로 49 epoch 학습했으며 기존 모델 가중치를 가져오지 않았다. 좌우 반전 포함 훈련 샘플은 11,250개다. 검증 첫 조향 오차는 0.0199 rad, 목표 속도 오차는 0.0808 m/s였다. 저장된 초기화·학습 정보는 `results/training_moe96.json`과 `results/dataset_manifest_moe96.json`에 기록했다.

## 시험 결과

새로운 12개 시험 시드를 위치 초기화 없이 평가했다. 기존 모델은 같은 Berlin 장면에서 비교용으로만 사용했다.

| 주행 방식 | 완주 | 충돌 | 완주 주행 평균 시간 | 직선 평균 속도 | 제어 지연 p95 |
|---|---:|---:|---:|---:|---:|
| 이전 TLN 단독 | 6/12 | 벽 2, 장애물 4 | 17.80 s | 3.15 m/s | 11.7 ms |
| 새 TLN 단독 | 7/12 | 벽 2, 장애물 3 | 15.57 s | 3.42 m/s | 13.0 ms |
| 새 TLN + MPCC 보호 | 12/12 | 없음 | 15.64 s | 3.75 m/s | 40.8 ms |

완주 회차의 평균 물리 시간은 새 TLN 단독에서 이전 모델보다 약 12.5% 짧았다. 새 TLN 단독은 아직 장애물 시험 5개에서 실패했다. 보호 모드는 장애물·경계 충돌 예상 시 MPCC로 보정했고 총 344 제어 스텝에 개입했다. 따라서 12/12 결과는 TLN 단독 성능이 아니라 TLN과 MPCC 보호기를 함께 쓴 성능이다. 제어 지연은 오프라인 시뮬레이터 측정이며 ROS 통신·실차 작동기 지연은 포함하지 않는다.

세부 기록은 `results/evaluation_best_regression_test_raw_protected_1lap_final_summary.json` 및 `results/evaluation_prior_model_reference_test_raw_1lap_prior_summary.json`에서 확인할 수 있다.

## 재현 명령

저장소 루트에서 실행한다.

```bash
export TLN_MAP_DIR="$PWD/stack_master/maps/berlin"
python3 experiments/tln_fresh/berlin_mpcc_fast/collect.py --split train --offset 0 --count 36
python3 experiments/tln_fresh/berlin_mpcc_fast/collect.py --split val --offset 0 --count 10
python3 experiments/tln_fresh/berlin_mpcc_fast/collect.py --split test --offset 0 --count 12
python3 experiments/tln_fresh/berlin_mpcc_fast/prepare.py
python3 experiments/tln_fresh/berlin_mpcc_fast/prepare_moe.py
python3 experiments/tln_fresh/berlin_mpcc_fast/train_moe96.py --epochs 100
python3 experiments/tln_fresh/berlin_mpcc_fast/evaluate.py --model models_moe96/best_regression.keras --split test --modes raw,protected
```

이 결과는 F1TENTH 동역학 시뮬레이터와 지도 위치를 사용하는 실험이다. 실차에서 같은 속도·회피 성능이 보장되는 것은 아니며 별도 저속 검증이 필요하다.
