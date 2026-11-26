#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math
import time
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import os
import json
import configparser
import copy

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from geometry_msgs.msg import PoseStamped, PointStamped, Quaternion, Point, PoseWithCovarianceStamped
from nav_msgs.msg import Path
from visualization_msgs.msg import Marker, MarkerArray
from visualization_msgs.msg import InteractiveMarker, InteractiveMarkerControl, InteractiveMarkerFeedback
from interactive_markers.interactive_marker_server import InteractiveMarkerServer

# f110_msgs
from f110_msgs.msg import WpntArray, Wpnt

# JSON 저장
from rosidl_runtime_py.convert import message_to_ordereddict

# TPH
import trajectory_planning_helpers as tph


# ---------- 유틸 ----------
def yaw_of_quat(q: Quaternion) -> float:
    siny_cosp = 2.0 * (q.w*q.z + q.x*q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y*q.y + q.z*q.z)
    return math.atan2(siny_cosp, cosy_cosp)

def quat_from_yaw(yaw: float) -> Quaternion:
    q = Quaternion()
    q.z = math.sin(yaw/2.0)
    q.w = math.cos(yaw/2.0)
    return q

def wrap_index(i: int, n: int, closed: bool) -> int:
    return (i % n) if closed else max(0, min(n-1, i))

def dist2(ax: float, ay: float, bx: float, by: float) -> float:
    return (ax-bx)**2 + (ay-by)**2

def uniform_resample(xy: np.ndarray, ds: float, closed: bool) -> np.ndarray:
    if len(xy) < 2:
        return xy.copy()
    diffs = np.diff(xy, axis=0)
    seg = np.linalg.norm(diffs, axis=1)
    s = np.insert(np.cumsum(seg), 0, 0.0)
    total = s[-1]
    if total < 1e-9:
        return xy.copy()
    new_s = np.arange(0.0, total, max(ds, 1e-6))
    if len(new_s) == 0 or abs(total - new_s[-1]) > 1e-9:
        new_s = np.append(new_s, total)
    x_new = np.interp(new_s, s, xy[:,0])
    y_new = np.interp(new_s, s, xy[:,1])
    out = np.column_stack([x_new, y_new])
    return out

# ---------- 데이터 ----------
@dataclass
class CompWp:
    x: float
    y: float
    psi: float
    s: float = 0.0
    vx: float = 0.0
    ax: float = 0.0
    d_right: float = 0.0
    d_left: float = 0.0
    kappa: float = 0.0


# ---------- 노드 ----------
class RacelineEditorWpnt(Node):
    def __init__(self):
        super().__init__('raceline_editor_wpnt')

        # 입력/출력/기본
        self.declare_parameter('input_topic', '/global_waypoints')
        self.declare_parameter('output_topic', '/global_waypoints_edited')
        self.declare_parameter('path_topic', '/raceline_edited_path')
        self.declare_parameter('clicked_topic', '/clicked_point')
        self.declare_parameter('goal_topic', '/goal_pose')
        self.declare_parameter('frame_id', 'map')

        # 라인 편집 범위
        self.declare_parameter('left_neighbors', 5)
        self.declare_parameter('right_neighbors', 5)
        self.declare_parameter('falloff_sigma', 2.0)

        # 리샘플 간격(해상도): 급커브 보존 위해 0.05 m 기본
        self.declare_parameter('spline_ds', 0.05)

        self.declare_parameter('closed_loop', True)

        # 동결 옵션
        self.declare_parameter('freeze_after_first_edit', True)
        self._frozen = False

        # 저장 옵션
        self.declare_parameter('map_dir', '')
        self.declare_parameter('source_json_name', 'global_waypoints.json')
        self.declare_parameter('edited_json_name', 'global_waypoints_edited.json')
        self.declare_parameter('save_on_every_edit', True)

        # 동역학/프로파일
        self.declare_parameter('ini_path', '/home/misys/forza_ws/race_stack/stack_master/config/global_planner/racecar_f110.ini')
        self.declare_parameter('recompute_kappa', True)
        self.declare_parameter('recompute_speed', True)

        # >>> 곡률/헤딩 미리보기/되돌아보기 거리 (m)
        self.declare_parameter('stepsize_psi_preview_m', 0.4)
        self.declare_parameter('stepsize_psi_review_m', 0.4)
        self.declare_parameter('stepsize_curv_preview_m', 0.4)
        self.declare_parameter('stepsize_curv_review_m', 0.4)

        # Undo
        self.declare_parameter('undo_history_size', 50)
        self.declare_parameter('undo_trigger_topic', '/initialpose')

        # 속도 표시(막대/텍스트) — 선택 인접만
        self.declare_parameter('speed_neighbors', 10)
        self.declare_parameter('speed_bar_width', 0.06)
        self.declare_parameter('speed_bar_scale', 0.15)
        self.declare_parameter('speed_bar_min_height', 0.02)
        self.declare_parameter('show_speed_text', True)
        self.declare_parameter('speed_text_size', 0.05)
        self.declare_parameter('speed_text_height', 0.10)


        # 편집 종료 신호(별도 Publish Point 토픽)
        self.declare_parameter('edit_done_point_topic', '/edit_done_point')

        qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                         history=HistoryPolicy.KEEP_LAST,
                         depth=20)

        # 통신
        self._input_topic = self.get_parameter('input_topic').value
        self._qos = qos
        self.sub_wp = self.create_subscription(WpntArray, self._input_topic, self.on_wpnts, self._qos)
        self.sub_click = self.create_subscription(PointStamped, self.get_parameter('clicked_topic').value, self.on_clicked, qos)
        self.sub_goal  = self.create_subscription(PoseStamped,  self.get_parameter('goal_topic').value,   self.on_goal, qos)
        self.sub_undo_trigger = self.create_subscription(
            PoseWithCovarianceStamped,
            self.get_parameter('undo_trigger_topic').value,
            self.on_initialpose,
            qos
        )
        # 편집 종료 Publish Point 구독 (RViz Publish Point QoS = BEST_EFFORT에 맞춤)
        qos_pp = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        edit_done_topic = self.get_parameter('edit_done_point_topic').value
        self.sub_edit_done = self.create_subscription(
            PointStamped, edit_done_topic, self.on_edit_done_point, qos_pp
        )

        self.pub_wp  = self.create_publisher(WpntArray, self.get_parameter('output_topic').value, qos)
        self.pub_path= self.create_publisher(Path,      self.get_parameter('path_topic').value,   qos)
        self.pub_mk  = self.create_publisher(MarkerArray, '/raceline_editor_wpnt/markers', qos)


        # Interactive Marker 서버 (Humble 호환)
        self._im_server = InteractiveMarkerServer(self, "raceline_speed_editor")

        # 상태
        self.frame_id = self.get_parameter('frame_id').value
        self.pts: List[CompWp] = []
        self.sel_idx: int = -1

        self._history: List[List[CompWp]] = []
        self._dyn_loaded = False
        self._dyn = {}
        self._est_lap_time = None

        self._last_bar_ids: set[int] = set()
        self._last_text_ids: set[int] = set()

        # 최초 /global_waypoints 수신 후 한 번만 /global_waypoints_edited 내보내기
        self._did_initial_publish = False

        self.get_logger().info('RacelineEditorWpnt ready.')

    # ---------- Undo ----------
    def on_initialpose(self, msg: PoseWithCovarianceStamped):
        if self._restore_snapshot():
            self.get_logger().info('[undo] Triggered.')
            self._publish_current()
            self._refresh_im_markers()
        else:
            self.get_logger().warn('[undo] No history to undo.')

    def _push_snapshot(self):
        hist_size = int(self.get_parameter('undo_history_size').value)
        snap_pts = copy.deepcopy(self.pts)
        self._history.append((snap_pts, int(self.sel_idx)))
        if len(self._history) > max(1, hist_size):
            self._history.pop(0)

    def _restore_snapshot(self):
        if not self._history:
            return False
        snap_pts, snap_sel = self._history.pop()
        self.pts = snap_pts
        self.sel_idx = int(snap_sel)
        return True

    # ---------- Interactive Marker (속도 편집) ----------
    def _refresh_im_markers(self):
        self._im_server.clear()

        if 0 <= self.sel_idx < len(self.pts):
            p = self.pts[self.sel_idx]
            bar_k   = float(self.get_parameter('speed_bar_scale').value)
            bar_min = float(self.get_parameter('speed_bar_min_height').value)

            im = InteractiveMarker()
            im.header.frame_id = self.frame_id
            im.name = f"speed_editor_{self.sel_idx}"
            im.description = f"vx={p.vx:.1f}"
            im.pose.position.x = p.x
            im.pose.position.y = p.y
            im.pose.position.z = 0.0  # 바닥 기준

            pillar = Marker()
            pillar.type = Marker.CUBE
            pillar.scale.x = 0.1
            pillar.scale.y = 0.1
            pillar.scale.z = max(bar_min, float(p.vx) * bar_k)
            pillar.color.a = 0.95
            pillar.color.r, pillar.color.g, pillar.color.b = 0.1, 0.8, 1.0
            pillar.pose.position.z = pillar.scale.z * 0.5

            txt = Marker()
            txt.type = Marker.TEXT_VIEW_FACING
            txt.text = f"{p.vx:.1f}"
            txt.scale.z = 0.12
            txt.color.a = 1.0
            txt.color.r = 1.0; txt.color.g = 1.0; txt.color.b = 1.0
            txt.pose.position.z = pillar.scale.z + 0.08

            ctrl = InteractiveMarkerControl()
            ctrl.name = "move_z"
            ctrl.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
            ctrl.orientation.w = 1.0
            ctrl.orientation.x = 0.0
            ctrl.orientation.y = 1.0
            ctrl.orientation.z = 0.0
            ctrl.always_visible = True
            ctrl.markers.append(pillar)
            ctrl.markers.append(txt)
            im.controls.append(ctrl)

            self._im_server.insert(im)
            self._im_server.setCallback(im.name, self._on_im_feedback)

        self._im_server.applyChanges()

    def _on_im_feedback(self, feedback: InteractiveMarkerFeedback):
        if feedback.event_type != InteractiveMarkerFeedback.MOUSE_UP:
            return
        if not (0 <= self.sel_idx < len(self.pts)):
            return

        bar_k   = float(self.get_parameter('speed_bar_scale').value)
        bar_min = float(self.get_parameter('speed_bar_min_height').value)

        base = self.sel_idx
        cur_vx = float(self.pts[base].vx)
        h0 = cur_vx * bar_k
        dz = float(feedback.pose.position.z)
        new_h = max(0.0, h0 + dz)
        new_vx = max(0.0, new_h / max(1e-6, bar_k))

        self._push_snapshot()

        self.pts[base].vx = new_vx

        L = int(self.get_parameter('left_neighbors').value)
        R = int(self.get_parameter('right_neighbors').value)
        sigma = float(self.get_parameter('falloff_sigma').value)
        closed = bool(self.get_parameter('closed_loop').value)

        for off in range(1, L+1):
            j = wrap_index(base - off, len(self.pts), closed)
            w = math.exp(- (off / max(1e-6, sigma))**2)
            self.pts[j].vx += (new_vx - self.pts[j].vx) * w
        for off in range(1, R+1):
            j = wrap_index(base + off, len(self.pts), closed)
            w = math.exp(- (off / max(1e-6, sigma))**2)
            self.pts[j].vx += (new_vx - self.pts[j].vx) * w

        # 속도 편집으로도 입력 동결
        if (not self._frozen) and bool(self.get_parameter('freeze_after_first_edit').value):
            self._frozen = True
            self.get_logger().info('RacelineEditor: input frozen (speed edit).')

        self._publish_current()
        self._maybe_save_json()

        self._refresh_im_markers()
        # 편집 중에는 GUI 트리거 호출하지 않음

    def _maybe_save_json(self):
        if not bool(self.get_parameter('save_on_every_edit').value):
            return

        map_dir = str(self.get_parameter('map_dir').value)
        if not map_dir:
            self.get_logger().warn("map_dir is empty. JSON save skipped.")
            return

        source_name = str(self.get_parameter('source_json_name').value)
        edited_name = str(self.get_parameter('edited_json_name').value)

        source_path = os.path.join(map_dir, source_name)
        edited_path = os.path.join(map_dir, edited_name)
        temp_path = edited_path + ".tmp"

        try:
            if not os.path.isfile(source_path):
                raise FileNotFoundError(
                    f"Original waypoint JSON not found: {source_path}"
                )

            # 원본 JSON 전체를 읽어서 centerline, shortest path, est_lap_time 등
            # 기존 메타데이터와 부가 경로 정보를 그대로 유지한다.
            with open(source_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 편집기에서 실제로 변경한 IQP raceline 항목만 교체한다.
            data['global_traj_markers_iqp'] = message_to_ordereddict(
                self.build_line_strip_markers()
            )
            data['global_traj_wpnts_iqp'] = message_to_ordereddict(
                self.build_wpnt_array()
            )

            # 경로를 다시 계산한 경우에만 추정 랩타임을 최신 값으로 갱신한다.
            # 원본 구조가 std_msgs/Float32 직렬화 형태라면 {"data": 값}을 유지한다.
            if self._est_lap_time is not None and 'est_lap_time' in data:
                if isinstance(data['est_lap_time'], dict):
                    data['est_lap_time']['data'] = float(self._est_lap_time)
                else:
                    data['est_lap_time'] = float(self._est_lap_time)

            # 저장 도중 파일이 깨지는 것을 막기 위해 임시 파일 작성 후 교체한다.
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            os.replace(temp_path, edited_path)
            self.get_logger().info(
                f"Saved edited JSON while preserving original metadata → {edited_path}"
            )

        except Exception as e:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass
            self.get_logger().error(f"Failed to save edited JSON: {e}")

    # ---------- 동역학 ----------
    def load_dyn_from_ini(self) -> bool:
        ini_path = self.get_parameter('ini_path').value
        parser = configparser.ConfigParser()
        if not parser.read(ini_path):
            self.get_logger().warn(f"ini not found or empty: {ini_path}")
            return False
        try:
            ggv_file = json.loads(parser.get('GENERAL_OPTIONS', 'ggv_file'))
            ax_file  = json.loads(parser.get('GENERAL_OPTIONS', 'ax_max_machines_file'))
            vel_calc_opts = json.loads(parser.get('GENERAL_OPTIONS', 'vel_calc_opts'))
            veh_params    = json.loads(parser.get('GENERAL_OPTIONS', 'veh_params'))

            base_dir = os.path.dirname(ini_path)
            veh_dyn_dir = os.path.join(base_dir, 'veh_dyn_info')
            ggv_path = os.path.join(veh_dyn_dir, ggv_file)
            ax_path  = os.path.join(veh_dyn_dir, ax_file)

            ggv, ax_max = tph.import_veh_dyn_info.import_veh_dyn_info(ggv_path, ax_path)

            self._dyn = {
                "ggv": ggv,
                "ax_max": ax_max,
                "v_max": float(veh_params.get("v_max", 15.0)),
                "mass": float(veh_params.get("mass", 3.5)),
                "dragcoeff": float(veh_params.get("dragcoeff", 0.0136)),
                "curvlim": float(veh_params.get("curvlim", 1.0)),
                "dyn_model_exp": float(vel_calc_opts.get("dyn_model_exp", 1.0)),
                "filt_window": vel_calc_opts.get("vel_profile_conv_filt_window", None)
            }
            self._dyn_loaded = True
            self.get_logger().info(f"Loaded dyn from ini: {ini_path}")
            return True
        except Exception as e:
            self.get_logger().error(f"Failed to load ini: {e}")
            return False

    # ---------- 곡률/속도 재계산 ----------
    def recompute_profiles(self, xy_calc: np.ndarray, closed: bool, ds: float, v_start: float | None):
        N = len(xy_calc)
        if N < 2:
            z = np.zeros(N)
            return z, z, z, z, z, 0.0

        # 균일 리샘플이라 seg 길이는 거의 ds
        diffs = np.diff(xy_calc, axis=0)
        el_lengths_seg = np.linalg.norm(diffs, axis=1)
        s_arr = np.insert(np.cumsum(el_lengths_seg), 0, 0.0)

        # TPH 요구 길이 맞춤: closed면 N개, open이면 원래는 N-1이지만 기존 호환 유지 위해 1개 붙임
        if closed:
            last_len = el_lengths_seg[0] if len(el_lengths_seg) > 0 else 0.0
        else:
            last_len = el_lengths_seg[-1] if len(el_lengths_seg) > 0 else 0.0
        el_lengths_for_tph = np.concatenate([el_lengths_seg, [last_len]])

        # >>> 작은 미리보기/되돌아보기 거리(m) 적용 <<<
        psi_prev = float(self.get_parameter('stepsize_psi_preview_m').value)
        psi_rev  = float(self.get_parameter('stepsize_psi_review_m').value)
        curv_prev= float(self.get_parameter('stepsize_curv_preview_m').value)
        curv_rev = float(self.get_parameter('stepsize_curv_review_m').value)

        psi_pts, kappa_pts = tph.calc_head_curv_num.calc_head_curv_num(
            path=xy_calc,
            el_lengths=el_lengths_for_tph,
            is_closed=closed,
            stepsize_psi_preview=psi_prev,
            stepsize_psi_review=psi_rev,
            stepsize_curv_preview=curv_prev,
            stepsize_curv_review=curv_rev,
            calc_curv=True
        )

        # TPH의 psi는 북(=y+) 기준, 기존 관례에 맞춰 +pi/2
        psi_pts = psi_pts + np.pi / 2.0

        if not self._dyn_loaded:
            vx_pts = np.zeros(N); ax_pts = np.zeros(N); lap_time = 0.0
            return s_arr, psi_pts, kappa_pts, vx_pts, ax_pts, lap_time

        # 속도 프로파일 입력용 곡률: 세그먼트 수에 맞춤
        kappa_for_vel = kappa_pts[:-1]

        kwargs = dict(
            ggv=self._dyn["ggv"],
            ax_max_machines=self._dyn["ax_max"],
            v_max=self._dyn["v_max"],
            kappa=kappa_for_vel,
            el_lengths=el_lengths_seg,
            closed=closed,
            filt_window=self._dyn["filt_window"],
            dyn_model_exp=self._dyn["dyn_model_exp"],
            drag_coeff=self._dyn["dragcoeff"],
            m_veh=self._dyn["mass"]
        )
        if not closed:
            kwargs["v_start"] = float(v_start if v_start is not None else 0.0)

        vx_seg = tph.calc_vel_profile.calc_vel_profile(**kwargs)
        vx_pts = np.append(vx_seg, vx_seg[0] if closed else vx_seg[-1])

        ax_seg = tph.calc_ax_profile.calc_ax_profile(
            vx_profile=vx_pts,
            el_lengths=el_lengths_seg,
            eq_length_output=False
        )
        ax_pts = np.append(ax_seg, ax_seg[-1] if len(ax_seg) else 0.0)

        t_prof = tph.calc_t_profile.calc_t_profile(
            vx_profile=vx_seg,
            ax_profile=ax_seg,
            el_lengths=el_lengths_seg
        )
        lap_time = float(t_prof[-1]) if len(t_prof) else 0.0

        return s_arr, psi_pts, kappa_pts, vx_pts, ax_pts, lap_time

    # ---------- 콜백 ----------
    def on_wpnts(self, msg: WpntArray):
        if self._frozen:
            return
        self.pts = []
        for w in msg.wpnts:
            x = float(getattr(w, 'x_m', getattr(w, 'x', 0.0)))
            y = float(getattr(w, 'y_m', getattr(w, 'y', 0.0)))
            psi = float(getattr(w, 'psi_rad', getattr(w, 'theta', 0.0)))
            s = float(getattr(w, 's_m', 0.0))
            vx = float(getattr(w, 'vx_mps', 0.0))
            ax = float(getattr(w, 'ax_mps2', 0.0))
            dR = float(getattr(w, 'd_right', 0.0))
            dL = float(getattr(w, 'd_left', 0.0))
            kap = float(getattr(w, 'kappa_radpm', 0.0))
            self.pts.append(CompWp(x,y,psi,s,vx,ax,dR,dL,kap))
        self.publish_markers()
        self._refresh_im_markers()

        # 최초 한 번만 편집본도 즉시 퍼블리시 → 슬라이서 autostart_on_data 트리거
        if (not self._did_initial_publish) and self.pts:
            self._publish_current()
            self._did_initial_publish = True
            self.get_logger().info("[init] Published initial /global_waypoints_edited")

    def on_clicked(self, msg: PointStamped):
        if not self.pts:
            return
        px, py = msg.point.x, msg.point.y
        idx = min(range(len(self.pts)), key=lambda i: dist2(self.pts[i].x, self.pts[i].y, px, py))
        self.sel_idx = idx
        self.get_logger().info(f'Selected index = {idx}')
        self.publish_markers()
        self._refresh_im_markers()

    def on_goal(self, msg: PoseStamped):
        if self.sel_idx < 0 or not self.pts:
            return
        self._push_snapshot()

        gx = msg.pose.position.x
        gy = msg.pose.position.y
        gyaw = yaw_of_quat(msg.pose.orientation)

        L = int(self.get_parameter('left_neighbors').value)
        R = int(self.get_parameter('right_neighbors').value)
        sigma = float(self.get_parameter('falloff_sigma').value)
        closed = bool(self.get_parameter('closed_loop').value)

        n = len(self.pts)
        sel = self.pts[self.sel_idx]
        dx = gx - sel.x
        dy = gy - sel.y

        self.pts[self.sel_idx].x = gx
        self.pts[self.sel_idx].y = gy
        self.pts[self.sel_idx].psi = gyaw

        def weight(k):
            return math.exp(- (k / max(1e-6, sigma))**2)

        for off in range(1, L+1):
            j = wrap_index(self.sel_idx - off, n, closed)
            w = weight(off)
            self.pts[j].x += dx * w
            self.pts[j].y += dy * w
        for off in range(1, R+1):
            j = wrap_index(self.sel_idx + off, n, closed)
            w = weight(off)
            self.pts[j].x += dx * w
            self.pts[j].y += dy * w

        self.rebuild_with_spline_and_publish()
        self._refresh_im_markers()

        if (not self._frozen) and bool(self.get_parameter('freeze_after_first_edit').value):
            self._frozen = True
            self.get_logger().info('RacelineEditor: input frozen.')

    # 편집 종료 Publish Point 콜백
    def on_edit_done_point(self, msg: PointStamped):
        # 사용자가 RViz Publish Point(토픽=edit_done_point_topic)로 “편집 끝”을 알림
        self._maybe_fire_trigger("edit done")

    # ---------- 리빌드 ----------
    def rebuild_with_spline_and_publish(self):
        if not self.pts:
            return
        closed = bool(self.get_parameter('closed_loop').value)
        ds = float(self.get_parameter('spline_ds').value)
        recompute_kappa = bool(self.get_parameter('recompute_kappa').value)
        recompute_speed = bool(self.get_parameter('recompute_speed').value)

        if recompute_speed and not self._dyn_loaded:
            if not self.load_dyn_from_ini():
                self.get_logger().warn("Failed to load ini/GGV. Will keep old vx/ax values.")

        prev_xy = np.array([[p.x, p.y] for p in self.pts], dtype=float)
        prev_vx = np.array([p.vx for p in self.pts], dtype=float)

        # ★ Catmull-Rom 제거: 원본 → 균일 리샘플만
        xy_calc = uniform_resample(prev_xy, ds, closed)

        if recompute_kappa or recompute_speed:
            v_start = None
            if not closed and len(prev_xy) > 0:
                idx0 = int(np.argmin(((prev_xy - xy_calc[0])**2).sum(axis=1)))
                v_start = float(prev_vx[idx0]) if len(prev_vx) else 0.0

            s_arr, psi_arr, kap_arr, vx_arr, ax_arr, lap_time = self.recompute_profiles(
                xy_calc=xy_calc, closed=closed, ds=ds, v_start=v_start
            )
        else:
            seg = np.linalg.norm(np.diff(xy_calc, axis=0), axis=1) if len(xy_calc) > 1 else np.array([])
            s_arr = np.insert(np.cumsum(seg), 0, 0.0) if len(seg) else np.zeros(len(xy_calc))
            psi_arr = np.zeros(len(xy_calc)); kap_arr = np.zeros(len(xy_calc))
            vx_arr = np.zeros(len(xy_calc));  ax_arr = np.zeros(len(xy_calc))
            lap_time = 0.0

        src_xy = np.array([[p.x, p.y] for p in self.pts])
        src_dR = np.array([p.d_right for p in self.pts]); src_dL = np.array([p.d_left for p in self.pts])

        new_pts: List[CompWp] = []
        for i in range(len(xy_calc)):
            x, y = float(xy_calc[i,0]), float(xy_calc[i,1])
            j = int(np.argmin(np.sum((src_xy - np.array([x,y]))**2, axis=1)))
            dR = float(src_dR[j]); dL = float(src_dL[j])
            new_pts.append(CompWp(
                x=x, y=y, psi=float(psi_arr[i]), s=float(s_arr[i]),
                vx=float(vx_arr[i]) if i < len(vx_arr) else 0.0,
                ax=float(ax_arr[i]) if i < len(ax_arr) else 0.0,
                d_right=dR, d_left=dL, kappa=float(kap_arr[i])
            ))
        self.pts = new_pts
        self._est_lap_time = float(lap_time)

        self.get_logger().info(f'Rebuilt raceline: N={len(self.pts)}, lap_time≈{lap_time:.2f}s')
        self._publish_current()
        self._maybe_save_json()

    # ---------- 메시지 빌드 ----------
    def build_wpnt_array(self) -> WpntArray:
        out = WpntArray()
        out.header.frame_id = self.frame_id
        out.header.stamp = self.get_clock().now().to_msg()
        for p in self.pts:
            w = Wpnt()
            w.s_m = float(p.s); w.x_m = float(p.x); w.y_m = float(p.y)
            w.psi_rad = float(p.psi); w.vx_mps = float(p.vx); w.ax_mps2 = float(p.ax)
            w.d_right = float(p.d_right); w.d_left = float(p.d_left); w.kappa_radpm = float(p.kappa)
            out.wpnts.append(w)
        return out

    # ---------- 마커 퍼블리시 ----------
    def publish_markers(self):
        if not self.pts:
            return

        arr = MarkerArray()

        m = Marker()
        m.header.frame_id = self.frame_id
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = 'raceline_wpnt'
        m.id = 0
        m.type = Marker.POINTS
        m.action = Marker.ADD
        m.scale.x = 0.05
        m.scale.y = 0.05
        m.color.a = 1.0
        m.color.r = 0.8; m.color.g = 0.8; m.color.b = 0.8
        for p in self.pts:
            q = Point(); q.x, q.y, q.z = p.x, p.y, 0.0
            m.points.append(q)
        arr.markers.append(m)

        current_bar_ids = set()
        current_text_ids = set()

        if 0 <= self.sel_idx < len(self.pts):
            ms = Marker()
            ms.header = m.header
            ms.ns = 'raceline_wpnt'
            ms.id = 1
            ms.type = Marker.SPHERE
            ms.action = Marker.ADD
            ms.scale.x = ms.scale.y = ms.scale.z = 0.15
            ms.color.a = 1.0; ms.color.r = 1.0; ms.color.g = 0.2; ms.color.b = 0.2
            ms.pose.position.x = self.pts[self.sel_idx].x
            ms.pose.position.y = self.pts[self.sel_idx].y
            arr.markers.append(ms)

            nbh = int(self.get_parameter('speed_neighbors').value)
            bar_w = float(self.get_parameter('speed_bar_width').value)
            bar_k = float(self.get_parameter('speed_bar_scale').value)
            bar_min_h = float(self.get_parameter('speed_bar_min_height').value)
            show_text = bool(self.get_parameter('show_speed_text').value)
            text_sz = float(self.get_parameter('speed_text_size').value)
            text_h  = float(self.get_parameter('speed_text_height').value)
            closed = bool(self.get_parameter('closed_loop').value)

            N = len(self.pts)
            rel = 0
            for j_idx in range(self.sel_idx - nbh, self.sel_idx + nbh + 1):
                j = wrap_index(j_idx, N, closed)
                p = self.pts[j]
                if j == self.sel_idx:
                    rel += 1
                    continue

                bar_id = 2000 + (rel + nbh)
                rel += 1
                h = max(bar_min_h, float(p.vx) * bar_k)
                b = Marker()
                b.header = m.header
                b.ns = 'raceline_speed_bar'
                b.id = bar_id
                b.type = Marker.CUBE
                b.action = Marker.ADD
                b.scale.x = bar_w
                b.scale.y = bar_w
                b.scale.z = h
                b.pose.position.x = p.x
                b.pose.position.y = p.y
                b.pose.position.z = h * 0.5
                b.color.a = 0.9
                t = min(1.0, max(0.0, p.vx / 12.0))
                b.color.r = 0.1 + 0.9*t
                b.color.g = 0.9 - 0.5*t
                b.color.b = 0.9 - 0.8*t
                arr.markers.append(b)
                current_bar_ids.add(bar_id)

                if show_text:
                    txt_id = 3000 + (j_idx - (self.sel_idx - nbh))
                    tmk = Marker()
                    tmk.header = m.header
                    tmk.ns = 'raceline_speed_text'
                    tmk.id = txt_id
                    tmk.type = Marker.TEXT_VIEW_FACING
                    tmk.action = Marker.ADD
                    tmk.pose.position.x = p.x
                    tmk.pose.position.y = p.y
                    tmk.pose.position.z = text_h
                    tmk.scale.z = text_sz
                    tmk.color.a = 1.0
                    tmk.color.r, tmk.color.g, tmk.color.b = 0.05, 0.9, 0.9
                    tmk.text = f"{p.vx:.1f}"
                    arr.markers.append(tmk)
                    current_text_ids.add(txt_id)

        for old_id in (self._last_bar_ids - current_bar_ids):
            dm = Marker()
            dm.header.frame_id = self.frame_id
            dm.header.stamp = self.get_clock().now().to_msg()
            dm.ns = 'raceline_speed_bar'
            dm.id = old_id
            dm.action = Marker.DELETE
            arr.markers.append(dm)

        for old_id in (self._last_text_ids - current_text_ids):
            dm = Marker()
            dm.header.frame_id = self.frame_id
            dm.header.stamp = self.get_clock().now().to_msg()
            dm.ns = 'raceline_speed_text'
            dm.id = old_id
            dm.action = Marker.DELETE
            arr.markers.append(dm)

        self._last_bar_ids = current_bar_ids
        self._last_text_ids = current_text_ids

        self.pub_mk.publish(arr)

    # JSON용 라인스트립 (시각화만)
    def build_line_strip_markers(self) -> MarkerArray:
        arr = MarkerArray()
        if not self.pts:
            return arr
        m = Marker()
        m.header.frame_id = self.frame_id
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = 'raceline_wpnt_json'
        m.id = 100
        m.type = Marker.LINE_STRIP
        m.action = Marker.ADD
        m.scale.x = 0.05
        m.color.a = 1.0
        m.color.r, m.color.g, m.color.b = 0.1, 0.9, 0.1
        for p in self.pts:
            q = Point(); q.x, q.y, q.z = p.x, p.y, 0.0
            m.points.append(q)
        arr.markers.append(m)
        return arr

    def _publish_current(self):
        if not self.pts:
            return
        self.pub_wp.publish(self.build_wpnt_array())

        path = Path()
        path.header.frame_id = self.frame_id
        path.header.stamp = self.get_clock().now().to_msg()
        for i in range(len(self.pts)):
            ps = PoseStamped()
            ps.header = path.header
            ps.pose.position.x = self.pts[i].x
            ps.pose.position.y = self.pts[i].y
            ps.pose.orientation = quat_from_yaw(self.pts[i].psi)
            path.poses.append(ps)
        self.pub_path.publish(path)

        self.publish_markers()

    # 내부 스냅샷 유틸
    def _restore_snapshot(self):
        if not hasattr(self, "_history") or not self._history:
            return False
        snap_pts, snap_sel = self._history.pop()
        self.pts = snap_pts
        self.sel_idx = int(snap_sel)
        return True


def main(args=None):
    rclpy.init(args=args)
    node = RacelineEditorWpnt()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
