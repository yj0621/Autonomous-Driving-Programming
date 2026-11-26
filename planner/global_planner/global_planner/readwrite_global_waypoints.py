'''
Shared functions to read and write map information (global waypoints)
'''
import os
import json
from rosidl_runtime_py.convert import message_to_ordereddict
from rosidl_runtime_py.set_message import set_message_fields

from visualization_msgs.msg import MarkerArray
from f110_msgs.msg import WpntArray
from std_msgs.msg import String, Float32
from typing import Tuple, List, Dict


def write_global_waypoints(map_dir: str,
                           map_info_str: str,
                           est_lap_time: Float32,
                           centerline_markers: MarkerArray,
                           centerline_waypoints: WpntArray,
                           global_traj_markers_iqp: MarkerArray,
                           global_traj_wpnts_iqp: WpntArray,
                           global_traj_markers_sp: MarkerArray,
                           global_traj_wpnts_sp: WpntArray,
                           trackbounds_markers: MarkerArray
                           ) -> None:
    '''
    Writes map information to a JSON file in the specified `map_dir` directory.
    '''
    path = os.path.join(map_dir, 'global_waypoints.json')
    print(f"[INFO] WRITE_GLOBAL_WAYPOINTS: Writing global waypoints to {path}")

    d: Dict[str, Dict] = {}
    d['map_info_str'] = {'data': map_info_str}
    d['est_lap_time'] = {'data': est_lap_time.data}
    d['centerline_markers'] = message_to_ordereddict(centerline_markers)
    d['centerline_waypoints'] = message_to_ordereddict(centerline_waypoints)
    d['global_traj_markers_iqp'] = message_to_ordereddict(global_traj_markers_iqp)
    d['global_traj_wpnts_iqp'] = message_to_ordereddict(global_traj_wpnts_iqp)
    d['global_traj_markers_sp'] = message_to_ordereddict(global_traj_markers_sp)
    d['global_traj_wpnts_sp'] = message_to_ordereddict(global_traj_wpnts_sp)
    d['trackbounds_markers'] = message_to_ordereddict(trackbounds_markers)

    with open(path, 'w') as f:
        json.dump(d, f)


def read_global_waypoints(map_dir: str) -> Tuple[
    String, Float32, MarkerArray, WpntArray, MarkerArray, WpntArray, MarkerArray, WpntArray, MarkerArray
]:
    '''
    Reads map information from a JSON file with path specified `map_dir`.
    '''
    path = os.path.join(map_dir, 'global_waypoints.json')

    print(f"[INFO] READ_GLOBAL_WAYPOINTS: Reading global waypoints from {path}")
    with open(path, 'r') as f:
        d: Dict[str, List] = json.load(f)

    map_info_str = String()
    set_message_fields(map_info_str, d['map_info_str'])

    est_lap_time = Float32()
    set_message_fields(est_lap_time, d['est_lap_time'])

    centerline_markers = MarkerArray()
    set_message_fields(centerline_markers, d['centerline_markers'])

    centerline_waypoints = WpntArray()
    set_message_fields(centerline_waypoints, d['centerline_waypoints'])

    global_traj_markers_iqp = MarkerArray()
    set_message_fields(global_traj_markers_iqp, d['global_traj_markers_iqp'])

    global_traj_wpnts_iqp = WpntArray()
    set_message_fields(global_traj_wpnts_iqp, d['global_traj_wpnts_iqp'])

    global_traj_markers_sp = MarkerArray()
    set_message_fields(global_traj_markers_sp, d['global_traj_markers_sp'])

    global_traj_wpnts_sp = WpntArray()
    set_message_fields(global_traj_wpnts_sp, d['global_traj_wpnts_sp'])

    trackbounds_markers = MarkerArray()
    set_message_fields(trackbounds_markers, d['trackbounds_markers'])

    return map_info_str, est_lap_time, \
        centerline_markers, centerline_waypoints, \
        global_traj_markers_iqp, global_traj_wpnts_iqp, \
        global_traj_markers_sp, global_traj_wpnts_sp, trackbounds_markers


def write_global_waypoints_edited(map_dir: str,
                                  map_info_str: str,
                                  edited_traj_markers: MarkerArray,
                                  edited_traj_wpnts: WpntArray,
                                  filename: str = 'global_waypoints_edited.json') -> None:
    '''
    Save the edited raceline into a JSON file, but keep all other fields from the
    existing global_waypoints.json for compatibility.

    - Reads the original global_waypoints.json if it exists
    - Replaces only "map_info_str", "global_traj_markers_iqp", "global_traj_wpnts_iqp"
    - Writes out to <map_dir>/<filename>
    '''
    orig_path = os.path.join(map_dir, 'global_waypoints_edited.json')
    path = os.path.join(map_dir, filename)

    d: Dict[str, Dict] = {}

    # 기존 JSON 있으면 로드해서 유지
    if os.path.exists(orig_path):
        with open(orig_path, 'r') as f:
            d = json.load(f)
        print(f"[INFO] WRITE_GLOBAL_WAYPOINTS_EDITED: Loaded existing fields from {orig_path}")
    else:
        print(f"[WARN] WRITE_GLOBAL_WAYPOINTS_EDITED: No existing {orig_path}, creating minimal JSON.")

    # 수정한 필드 덮어쓰기
    d['map_info_str'] = {'data': map_info_str}
    d['global_traj_markers_iqp'] = message_to_ordereddict(edited_traj_markers)
    d['global_traj_wpnts_iqp'] = message_to_ordereddict(edited_traj_wpnts)

    # 새 파일로 저장
    with open(path, 'w') as f:
        json.dump(d, f)

    print(f"[INFO] WRITE_GLOBAL_WAYPOINTS_EDITED: Saved edited waypoints to {path}")
