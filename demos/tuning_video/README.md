# ForzaETH 튜닝 문제 설명 영상

`forzaeth_tuning_18s.mp4`: 18초, 1280×720, 30 fps, H.264, 무음.

- 0–6초: PurePursuit 수식과 운동학 자전거 모델을 실행. 짧은 lookahead와 조향 지연 조건에서 진동을 시각화.
- 6–12초: 좌우 회피 후보를 주기적으로 전환하는 대리 local planner. 실제 ForzaETH spline planner를 실행한 결과가 아님.
- 12–18초: 조향 0의 예시 예측에 최저속도 3 m/s 제한을 적용. 실제 TinyLidarNet 모델 추론 결과가 아님.

**실제 ROS/ForzaETH 실행 녹화가 아닌 설명용 시뮬레이션이다.** 원본 실행 영상은 작업 폴더에서 발견하지 못했다. 따라서 이 파일만으로 실제 스택 튜닝 중 발견한 문제의 실행 증거 요구를 충족한다고 볼 수 없다. 실제 실행 증거가 필요하면 해당 실행 녹화로 각 구간을 교체해야 한다. TLN 속도 제한의 근거는 저장소 `TLN_ANALYSIS.md`이며, 실제 충돌 원인이나 제어기 성능 비교를 입증하지 않는다. 조향 기록과 속도는 이 경량 시뮬레이션에서 생성한 수치다. 충돌 시 차량 위치를 고정하며 명령 속도와 조향 후처리는 계속 표시한다.

재생성: `python3 make_video.py` (numpy, Pillow, imageio-ffmpeg 필요).
`telemetry.json`에 세 장면의 프레임별 상태를 저장한다.
폰트: Google Fonts의 Noto Sans KR (SIL Open Font License).
