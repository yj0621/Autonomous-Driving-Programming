#!/usr/bin/env python3
"""Prints a map's obstacle candidate pool so it can be eyeballed.

The spawner builds the same pool at startup from the same inputs, so nothing
reads this back and there is no file to keep in sync. Write one out only to
hand-edit the list and feed it back with candidates_file.

    ros2 run race_manager dump_candidates --map hyhall12
    ros2 run race_manager dump_candidates --map hyhall12 --out /tmp/pool.yaml
"""

import argparse
import collections
import hashlib
import math
import os
import sys

import yaml
from ament_index_python.packages import get_package_prefix, get_package_share_directory

from race_manager.placement import generate_candidates
from race_manager.track import Track

GEN_PARAMS = ('obstacle_size', 'wall_margin', 'car_width', 'squeeze_clearance',
              'candidate_spacing', 'candidate_d_step', 'candidate_thetas',
              'exclusion_ahead', 'exclusion_behind', 'min_severity')


def maps_root():
    return os.path.normpath(
        os.path.join(get_package_prefix('stack_master'), '..', '..', 'stack_master', 'maps'))


def spawner_params():
    cfg = os.path.join(get_package_share_directory('race_manager'),
                       'config', 'race_manager_params.yaml')
    with open(cfg) as f:
        data = yaml.safe_load(f)
    return data['obstacle_spawner']['ros__parameters']


def md5(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--map', required=True)
    ap.add_argument('--list', action='store_true', help='print every candidate, one per line')
    args = ap.parse_args()

    map_dir = os.path.join(maps_root(), args.map)
    wpnts = os.path.join(map_dir, 'global_waypoints.json')
    if not os.path.exists(wpnts):
        sys.exit(f'no raceline for map "{args.map}": {wpnts}')

    p = spawner_params()
    track = Track(wpnts)
    cands = generate_candidates(
        track,
        size=p['obstacle_size'],
        wall_margin=p['wall_margin'],
        car_width=p['car_width'],
        squeeze_clearance=p['squeeze_clearance'],
        spacing=p['candidate_spacing'],
        d_step=p['candidate_d_step'],
        thetas=list(p['candidate_thetas']),
        exclusion_ahead=p['exclusion_ahead'],
        exclusion_behind=p['exclusion_behind'],
        min_severity=p['min_severity'],
    )
    if not cands:
        sys.exit('generated 0 candidates — lower min_severity, squeeze_clearance, '
                 'or the exclusion_ahead/exclusion_behind bands')

    by_side = collections.Counter(c.side for c in cands)
    sev = sorted(c.severity for c in cands)
    vis = sorted(c.visibility for c in cands)
    pct = lambda v, q: v[min(len(v) - 1, int(q * len(v)))]

    print(f'map {args.map}  track {track.length:.3f} m  waypoints md5 {md5(wpnts)}')
    print('  ' + ', '.join(f'{k}={p[k]}' for k in GEN_PARAMS))
    print(f'\n{len(cands)} candidates — '
          f'either {by_side["either"]}, LEFT {by_side["LEFT"]}, RIGHT {by_side["RIGHT"]}')
    print(f'  severity   min {sev[0]:.2f}  p50 {pct(sev, .5):.2f}  max {sev[-1]:.2f} m')
    print(f'  visibility min {vis[0]:.1f}  p50 {pct(vis, .5):.1f}  max {vis[-1]:.1f} m')
    print(f'  chicane material (|d| >= 0.10, either): '
          f'{sum(1 for c in cands if c.side == "either" and abs(c.d) >= 0.10)}')
    print(f'  hard forced (severity >= 0.35): '
          f'{sum(1 for c in cands if c.side != "either" and c.severity >= 0.35)}')

    if args.list:
        print()
        for c in cands:
            print(f'  {c.id:<12} s={c.s:6.2f} d={c.d:+.2f} theta={math.degrees(c.theta):+7.1f}  '
                  f'{c.side:<6} sev {c.severity:.3f}  vis {c.visibility:5.2f}')


if __name__ == '__main__':
    main()
