# 실제 RViz2 실행 녹화 · 18초 편집본

결과: `forzaeth_rviz2_18s.mp4` — 1600×900, 30 fps, H.264, 18초, 무음.

실제 ROS2 Humble의 설치된 ForzaETH/F1TENTH 스택을 실행하고 Xvfb 디스플레이 `:88`의 RViz2 화면을 ffmpeg x11grab으로 녹화했다. 생성형 영상이나 대리 시뮬레이션 그림은 사용하지 않았다. 편집은 원본의 실시간 구간 선택, 한국어 자막, 동일 프레임 확대 창으로 구성된다. 세 구간 모두 재생 속도는 1배다. 원본 녹화와 실행 로그, 프레임별 측정값을 같은 폴더에 보관했다.

## 편집 구간과 확인된 문제

| 최종 영상 | 원본 | 원본 시작 | 조건 및 관측 |
|---|---|---|---|
| 0–6초 | `pp_verified_raw.mp4` | 3.3초 | 실제 C++ PP controller. t_clip_min=1.5, t_clip_max=6.0, m_l1=1.0, q_l1=0.8, global_limit=0.65. 긴 lookahead 조건에서 코너 안쪽 벽 충돌. |
| 6–12초 | `local_verified_raw.mp4` | 6.0초 | 실제 spline_planner와 state_machine. evasion_dist=1.4 m, global_limit=0.25. TRACKBOUNDS 제약으로 회피 경로 취소, 출력 0점, TRAILING 감속·정체. 충돌 장면이 아니다. |
| 12–18초 | `tln_verified_raw.mp4` | 3.3초 | 실제 TLN_aug.keras 추론과 기본 후처리(v_min=3.0). 경로 이탈·벽 충돌. 충돌 후 실제 차량 속도는 0이나 속도 명령은 약 4.0 m/s로 남는다. |

이는 지정한 시험 조건에서 새로 관측한 결과다. 기존 튜닝 이력의 녹화이거나 각 알고리즘의 일반적 성능을 입증하는 결과로 해석하지 않는다. TLN 벽 충돌의 원인을 최저속도 제한으로 확정하지 않는다.

## 실행 조건과 증거

- ROS_DOMAIN_ID=87로 기존 세션과 분리. 맵 `hyhall12_orig.yaml`을 사용하는 실제 gym_bridge. RViz2에서 맵, LiDAR, 차량, 경로를 렌더링했다.
- ground truth pose/odom을 실제 frenet_odom_republisher로 변환했다. PF 오차를 포함한 시험이 아니다.
- 실제 global_trajectory_publisher, speed_scaler, state_machine, spline_planner, controller_manager_cpp(mode=PP), ofc/tln_inference를 실행했다.
- `recording_harness.py`는 선택한 실제 제어기의 명령을 `/drive`로 그대로 전달하고, `/initialpose`로 리셋하며, 물리 시뮬레이터에 0.30 m 정적 장애물을 넣었다. 리셋 직후 0.3초는 정지 명령을 전달한다.
- local planner를 인식 오차와 분리해 시험하기 위해 장애물의 알려진 위치를 `/perception/obstacles`에도 입력했다. 실제 perception 검출 성능을 시험한 영상이 아니다.
- TLN은 `/scan`만으로 직접 구동한다. 화면의 local waypoints와 state는 TLN 제어에 사용하지 않는다.
- `/sim/collision`, `/car_state/odom_GT`, 실제 제어 명령과 spline 출력을 `telemetry.jsonl`에 기록했다. `/sim/collision`의 반복 발행은 접촉 상태 변화 시 한 번만 충돌 이벤트로 기록한다. 초기 0.35초의 리셋 전 잔여 샘플은 충돌 집계와 궤적에서 제외했다.
- PP/TLN의 wall 충돌 이벤트와 planner의 경계 거부 로그는 `verified_summary.json`, `pp.log`, `stack3.log`, `tln.log`에 남아 있다. 각 녹화의 `*_capture.json`에 명령·시각을, `edit_manifest.json`에 편집 구간을 저장했다.
- SIM l1 설정에 누락된 `trailing_to_gbtrack_speed_scale` 때문에 PP가 처음 시작되지 않아, 설치본에만 0.3을 임시 추가했다. 촬영 후 설치본과 소스 설정 모두 원래 파일과 바이트 단위로 동일하게 복구했다. 해당 값은 GB_TRACK 충돌 시험의 동작 조건을 바꾸는 튜닝 항목이 아니다.
- 작업용 도메인 87의 실행 프로세스는 촬영 후 종료했다. 기존 도메인의 scorer 등은 유지했다.

`edit_video.py`로 원본에서 동일 편집본을 재생성할 수 있다. `capture_case.py`, `set_params.py`, `recording_harness.py`는 사용한 녹화·시험 도구다. 한국어 폰트는 인접 `../tuning_video/NotoSansKR.ttf`의 Noto Sans KR(SIL OFL)을 사용한다.
