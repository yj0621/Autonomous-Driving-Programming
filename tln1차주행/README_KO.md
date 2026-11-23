# TLN 1차 주행 재현

ROS 2 Humble 및 이 저장소의 F1TENTH simulator 설치가 된 환경용 최소 실행 묶음입니다. 맵은 Hangar 1905입니다. 모델, 전처리/제어 코드, 시뮬레이터 브리지, 무작위 정적 장애물 노드 및 RViz 설정을 포함합니다.

이 폴더를 기존 `race_stack` 저장소 루트에 덮어써서 파일을 배치한 뒤 실행합니다. Python 경로를 고정한 launch 파일에서 `R`을 본인 저장소 경로로 수정하세요.

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID=86
ros2 launch /tmp/tln_random_demo.launch.py
```

장애물 3개가 트랙 안쪽에 무작위 배치되며 20초마다 재배치됩니다. 학생 모델은 지정된 `models_moe96/best_regression.keras`를 읽습니다. 맵, 시뮬레이터 파라미터, race_manager 유틸리티 파일도 함께 필요합니다. Python 의존성은 저장소 환경(NumPy, SciPy, PyYAML, Pillow, Numba, TensorFlow, pyglet, f110_gym)을 사용합니다.
