# Pure Pursuit Demo

Run the interactive demo from the repository root:

```bash
python3 demos/pure_pursuit_demo.py
```

Move the `Ld [m]` slider to see how lookahead distance changes behavior:

- Smaller `Ld`: sharper, more reactive steering and more oscillation.
- Larger `Ld`: smoother motion, but wider cornering and slower recovery.

The vehicle model includes a small steering delay so the short-lookahead
oscillation is intentionally easy to see.

The demo is standalone Python and uses only `matplotlib` for visualization.
