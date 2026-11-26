# race_manager

A measurement harness for static obstacle avoidance. One launch is one run, and you
**end it with Ctrl-C.**

## Running

With the simulator already up:

```bash
ros2 launch race_manager race_manager_launch.xml \
  map_name:=hyhall12 seed:=20260913 num_obstacles:=5
```

| Argument | Default | Meaning |
|---|---|---|
| `map_name` | `hyhall12` | Folder name under `stack_master/maps/` |
| `seed` | `-1` | **>= 0 is reproducible** (for grading), `-1` randomises every run (practice) |
| `num_obstacles` | `4` | How many to draw from the candidate pool each lap |
| `log_dir` | `race_result` | Where the JSON accumulates |
| `plot` | `true` | Live report window while driving. `false` gives JSON only |

There is no fixed lap count. Run as long as you like, then **Ctrl-C**: a summary is
printed to the terminal and `race_result/race_result_<YYYYmmdd_HHMMSS>.json` is left behind.

```
--- seed 20260913  ·  hyhall12  ·  3 laps ---
  best lap    16.95 s
  collisions  4   (obstacle 1, wall 3)

  lap      time     hits   obstacle   wall
    1     18.83 s      4          1      3
    2     17.20 s      0          0      0
    3 *   16.95 s      0          0      0

  best lap is the fastest clean lap driven line to line  (1 of 3 left out)
```

`*` marks the best lap, and **only a lap driven clean and timed line to line can be
one.** A lap with a collision is slower for a reason that is already recorded, and a
lap timed from where the car was put back on track (`partial`) covers less than a lap,
so it is short rather than fast. Both are still listed and still count towards
`laps_completed`. Reporting:

```bash
ros2 run race_manager report                  # the most recent one in race_result/
ros2 run race_manager report race_result/race_result_20260913_185432.json
ros2 run race_manager report --watch          # keeps updating while you drive
ros2 run race_manager report --no-show        # PNG only, no window
```

The PNG is saved next to the JSON and a window opens.

### Watching while you drive — `--watch`

**The launch file opens this window for you.** Laps and collisions appear as they happen.
If you do not want the window, turn it off with `plot:=false`; the JSON still accumulates.
To open one yourself from a terminal:

```bash
ros2 run race_manager report --watch        # every 1 s
ros2 run race_manager report --watch 0.5    # pick the interval
```

The scorer rewrites the JSON **on every lap and every collision** (swapped in atomically
with `os.replace`, so reading it mid-run is safe). `--watch` polls that mtime and
**redraws a single window.**

```
watching every 1 s — close the window to stop
following race_result/race_result_20260914_160000.json
16:08:50  lap 1  best 18.83 s  obstacle 0  wall 0
16:08:52  lap 2  best 17.20 s  obstacle 1  wall 0
16:08:55  lap 3  best 16.95 s  obstacle 1  wall 1
```

| | Behaviour |
|---|---|
| No log path given | **Follows the newest log.** When a new run starts, the window switches to it |
| No log yet | Waits for one to appear (not an error) |
| Window closed | Loop ends (if the launch file opened it, only that node stops; measurement continues) |
| No display | Fails immediately with a clear message. Use `plot:=false` |
| Title | Seed · map · start time |

`--watch` and `--no-show` cannot be combined.

## The obstacle candidate pool

**The candidate pool is not kept in a file.** `generate_candidates()` is a pure function:
the same `global_waypoints.json` and the same generation parameters give the same list in
the same order. What has to be frozen is the **inputs**, not the result.

```
reproducibility contract =  waypoints md5  +  generation parameters  +  seed
```

All three are logged on one line when the spawner starts. That line alone is enough to
reconstruct a layout, so there is no extra file to keep in the map directory.

### What one candidate carries

```
s042d+01t1   s=30.23  d=+0.10  LEFT   sev 0.47  vis 4.8
└─┬─┘└─┬┘└┬┘
  │    │  └ theta index (which of candidate_thetas)
  │    └─── d index     (d = index × candidate_d_step)
  └──────── s index     (which station of the candidate_spacing grid)
```

| Label | Meaning |
|---|---|
| `side` | Which side the car may pass on: `either` / `LEFT` / `RIGHT` |
| `severity` | Metres the car must leave the raceline by to get past |
| `visibility` | Metres of raceline it stays in line of sight for (occlusion by the track boundary) |

The id *is* the grid coordinate, so an id in the log tells you where and how it was placed.

### Generation rules

Every cell of the `candidate_spacing` × `candidate_d_step` grid is tried with each rotation
in `candidate_thetas`, and only those passing all of the below are kept.

| Condition | Meaning |
|---|---|
| `exclusion_ahead <= s <= length - exclusion_behind` | Outside the start line bands (below) |
| Footprint fits between the walls, less `wall_margin` | Does not merge into a wall cluster |
| `car_width + 2×squeeze_clearance` left on at least one side | Physically passable |
| `severity >= min_severity` | Default `0.0`, so nothing is dropped here |

If only one side is open the candidate is `LEFT` / `RIGHT`, if both are open it is `either`.
`squeeze_clearance` is the threshold for that call — set it high and gaps the car could
actually take count as "blocked", classifying non-forced placements as forced.

#### `min_severity` cuts position, not difficulty

`severity` is measured against the raceline, so `severity >= x` is really the geometric
constraint `|d| <= 0.31 - x`. Even `0.05` confines every obstacle to a ±0.3 m ribbon along
the raceline on a track 2–3 m wide, which throws away most of the grid. Leave it at `0.0`
and let `recipes` carry the difficulty instead.

#### The start line bands are asymmetric

The layout is re-drawn the instant the car crosses the start line, so at that moment the
car sits at `s ≈ 0`:

| Band | Default | Why |
|---|---|---|
| `[0, exclusion_ahead)` | `5.0 m` | Straight ahead of the car. This is the reaction distance for an obstacle appearing in its face |
| `(length - exclusion_behind, length]` | `1.0 m` | Already behind the car. It will not be met for another full lap, so this only has to clear the parked car |

Reserving the same width behind as ahead leaves the run up to the finish line permanently
empty, which rewards flooring it through that section every lap.

### Per-lap composition — `recipes`

Drawing at random gives some laps four `either` obstacles that never ask for a direction and
other laps nothing but forced ones. A recipe is **a table of how many of each kind to draw**,
and `recipes` is a list of those tables — the seed picks one per lap.

```yaml
recipes:
  - 'spread:1.0 chicane:1:0.10 forced:1:0.20'   # chicane pair + one forced + one free
  - 'spread:1.0 forced:2:0.20'                  # two forced sides + two free
  - 'spread:0.5 forced:1:0.30 either:1:0.20'    # one hard forced + one two-way + two free
  - 'chicane:2:0.10'                            # no spread: the deliberately clustered lap
```

One line is one recipe; its slots are separated by whitespace. A slot is
`kind:count[:min_severity[:max_visibility]]`.

| kind | Meaning |
|---|---|
| `chicane` | A pair that pushes the car opposite ways, within `chicane_span`. **Uses two obstacles** |
| `LEFT` / `RIGHT` / `either` | The side the car may pass on |
| `forced` | One side only, `LEFT` or `RIGHT`, no preference |
| `any` | No condition |

Whatever is left after the recipe is filled with no condition at all. A slot that cannot be
filled is relaxed step by step (`min_severity` / `max_visibility` dropped, then falling back
to `any`) and **what gave is recorded as a warning.** A narrow map with no forced positions
shows up here.

A slot is also skipped, with the same warning, when there is **no room left in
`num_obstacles`** — a `chicane` needs two free places, everything else needs one. Without
that check the surplus would be truncated off the end of the list and a chicane could lose
its partner silently. Recipe 3 above asks for exactly 4, so lowering `num_obstacles` below 4
makes it warn on every lap it is drawn; drop it from the pool if you run with fewer.

A malformed slot (`:0.20`, no kind) raises at startup rather than being skipped, and an
empty `recipes` list is rejected outright.

#### `spread` — how far apart one lap's obstacles sit

Uniform random positions clump: that is Poisson clumping, not a bug, and it is why roughly
one lap in five used to put all four obstacles into a short stretch of track. `spread:<f>`
turns the draw into dart throwing — the reference Poisson-disk sampler — by requiring a
minimum arc gap between separate placements:

```
gap = f × (arc length the candidate pool covers) / (independent placements + 1)
```

`f` runs 0 to 1, and being a fraction rather than a distance keeps it meaningful on a 23 m
map and a 55 m one alike. A `chicane` pair counts as **one** placement and both halves are
put down in the same step, so they stay within `chicane_span` however high `f` goes; the gap
only separates a pair from everything else.

Leaving `spread` off a recipe is what allows a clustered lap, so **the share of recipes
without it is the clustered-lap rate** — one in four in the pool above, on every map.
Measured over seven maps, recipes carrying `spread` never once put every obstacle inside a
quarter of the track, and added no relaxation warnings.

### The passability check

`draw()` only compares centre distances, so on its own it would let several obstacles
**jointly** block the track. Every draw therefore computes the corridor left at every `s` the
obstacles reach and discards the combination if it falls below
`car_width + 2×squeeze_clearance`. That check is what makes it safe to set
`min_obstacle_gap` as low as 0.5 m.

### Inspecting the pool

```bash
ros2 run race_manager dump_candidates --map hyhall12            # summary only
ros2 run race_manager dump_candidates --map hyhall12 --list     # print them all
```

It prints the **same list** the spawner builds at startup, in the same order. Output only —
nothing reads it back.

## Topics

| Topic | Direction | Contents |
|---|---|---|
| `/sim/obstacles` | spawner → bridge | This lap's obstacles (`PoseArray`, latched) |
| `/sim/obstacles/markers` | spawner → RViz | Orange cubes with number labels |
| `/sim/collision` | bridge → scorer | `none` / `obstacle` / `wall` |
| `/sim/lap_count` | scorer → spawner | Lap boundary; triggers the re-draw |

Recovering a crashed car is the operator's job, via `/initialpose` (RViz's `2D Pose
Estimate`). The particle filter subscribes to the same topic, so localisation is re-seeded
at the same time.

## Seeing the obstacles in RViz

Obstacles go out **all at once** on two topics. Independently of what the car sees through
`/scan`, the operator sees the whole layout.

| Display | Topic | Shows |
|---|---|---|
| MarkerArray | `/sim/obstacles/markers` | 0.3 m orange cubes, numbered from 1 |
| PoseArray | `/sim/obstacles` | Position and orientation only, lightweight |

Both are latched (TRANSIENT_LOCAL), so starting RViz late still shows the current layout
immediately. The same ids are re-sent every second so nothing flickers, and the previous
markers are cleared only when a lap boundary triggers a re-draw.

## What is recorded

**Nothing is scored.** Lap times, collision counts (obstacle and wall kept apart), collision
positions and operator interventions (teleports) are recorded; no conversion is applied.
These JSON keys are all of it.

```
started_at  map  waypoints_file  seed
best_lap  laps_completed  total_time
laps: [{lap, time, obstacle, wall, total, partial}, ...]
collisions: {obstacle, wall, total}  collision_events  teleports
```

`best_lap` is the one derived value: the fastest lap with `total == 0` and
`partial == false`. `partial` marks a lap whose clock started at a recovery rather
than at the line.

The terminal summary puts **best lap / total collisions / counts per kind** at the top and,
below that, **lap number · lap time · collision count · breakdown** per lap. The PNG is laid
out like a race timing screen.

| Zone | Contents |
|---|---|
| Header | Map · seed · start time |
| Readout | `BEST LAP` / `LAPS` / `HITS` / `OBSTACLE` / `WALL` |
| Timing table | `LAP · TIME · GAP · HITS` per lap. **GAP is the delta to the best lap**, signed, or `PART` for a lap timed from a recovery (its time is greyed, since it is not a lap's worth of driving). Collisions are **one glyph per hit**: an orange circle is an obstacle, a blue X is a wall. The best lap's row is lightened and `BEST` appears in the GAP column |
| TRACK | The raceline and collision positions, using the same orange-circle / blue-X legend as the timing table |

Up to 10 laps the timing table is one column, from 11 it is two. From 25 laps only the most
recent 24 are shown so the type size can stay put, and a best lap that has scrolled off is
pinned separately above the table.

There are only two colours — **obstacle orange `#d95926` / wall blue `#3987e5`** — and
everything else is greyscale. The obstacle colour matches the orange cubes in RViz. The
motorsport conventions, purple for session best and green for personal best, are **not
used**: purple sits at ΔE 1.9 from blue under protanopia and green at ΔE 1.4 from orange
under deuteranopia, so they are indistinguishable with colour vision deficiency. The best
lap is marked by the row background and a `BEST` tag instead of by colour.

## Known behaviour

- **A collision stops the car for good.** That is what the simulator's `check_ttc` does.
  `/sim/collision` latches in that state, so **the next collision is not counted until the
  car is recovered.**
- When the operator moves the car with `/initialpose`, `env.reset()` fires, the signal
  returns to `none` and counting resumes. That move is detected as a jump larger than
  `teleport_m` (default 1.5 m), which **voids the lap in progress** and is recorded under
  `teleports` — this is what stops shortcutting. The next lap to be credited is then
  timed from where the car was put down, so it covers less than a lap: it is recorded
  with `partial: true` and cannot be the best lap.
- Crossing the finish line is not enough for a lap. More than half the track has to have been
  driven (judged by accumulated change in `s`).
- Lap counting runs off ground truth (`/car_state/odom_GT`). Basing it on the particle filter
  would make lap detection depend on localisation quality, which is unfair.
