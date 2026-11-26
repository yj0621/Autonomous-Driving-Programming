#!/usr/bin/env python3
"""Interactive Pure Pursuit demo using only Python and matplotlib."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Polygon
from matplotlib.widgets import Button, Slider


def wrap_to_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


@dataclass
class ReferencePath:
    x: list[float]
    y: list[float]
    s: list[float]
    length: float
    theta: list[float]
    radius: list[float]
    y_scale: float

    @classmethod
    def make_track(cls, samples: int = 900) -> "ReferencePath":
        x_values: list[float] = []
        y_values: list[float] = []
        theta_values: list[float] = []
        radius_values: list[float] = []
        y_scale = 0.72

        for i in range(samples):
            theta = 2.0 * math.pi * i / samples
            radius = 7.0 + 1.25 * math.sin(3.0 * theta + 0.45) + 0.65 * math.sin(5.0 * theta - 1.2)
            theta_values.append(theta)
            radius_values.append(radius)
            x_values.append(radius * math.cos(theta))
            y_values.append(y_scale * radius * math.sin(theta))

        s_values = [0.0]
        total = 0.0
        for i in range(1, samples):
            total += math.hypot(x_values[i] - x_values[i - 1], y_values[i] - y_values[i - 1])
            s_values.append(total)
        total += math.hypot(x_values[0] - x_values[-1], y_values[0] - y_values[-1])

        return cls(x_values, y_values, s_values, total, theta_values, radius_values, y_scale)

    def boundaries(self, half_width: float = 0.9) -> tuple[list[float], list[float], list[float], list[float]]:
        left_x: list[float] = []
        left_y: list[float] = []
        right_x: list[float] = []
        right_y: list[float] = []

        for theta, radius in zip(self.theta, self.radius):
            outer_r = radius + half_width
            inner_r = max(0.5, radius - half_width)
            left_x.append(outer_r * math.cos(theta))
            left_y.append(self.y_scale * outer_r * math.sin(theta))
            right_x.append(inner_r * math.cos(theta))
            right_y.append(self.y_scale * inner_r * math.sin(theta))

        return left_x, left_y, right_x, right_y

    def nearest_index(self, px: float, py: float) -> int:
        best_idx = 0
        best_dist = float("inf")
        for i, (x_i, y_i) in enumerate(zip(self.x, self.y)):
            dist = (x_i - px) ** 2 + (y_i - py) ** 2
            if dist < best_dist:
                best_idx = i
                best_dist = dist
        return best_idx

    def target_index(self, nearest_idx: int, lookahead: float) -> int:
        target_s = (self.s[nearest_idx] + lookahead) % self.length
        if target_s >= self.s[nearest_idx]:
            for i in range(nearest_idx, len(self.s)):
                if self.s[i] >= target_s:
                    return i
        for i, s_i in enumerate(self.s):
            if s_i >= target_s:
                return i
        return 0


@dataclass
class Vehicle:
    x: float
    y: float
    yaw: float
    speed: float = 4.1
    wheelbase: float = 0.50
    max_steer: float = math.radians(42.0)
    steering_angle: float = 0.0
    steering_delay: float = 0.16
    history_x: list[float] = field(default_factory=list)
    history_y: list[float] = field(default_factory=list)

    def step(self, path: ReferencePath, lookahead: float, dt: float) -> tuple[int, int, float, float]:
        nearest_idx = path.nearest_index(self.x, self.y)
        target_idx = path.target_index(nearest_idx, lookahead)
        target_x = path.x[target_idx]
        target_y = path.y[target_idx]
        cross_track_error = math.hypot(path.x[nearest_idx] - self.x, path.y[nearest_idx] - self.y)

        target_angle = math.atan2(target_y - self.y, target_x - self.x)
        alpha = wrap_to_pi(target_angle - self.yaw)
        steering_cmd = math.atan2(2.0 * self.wheelbase * math.sin(alpha), lookahead)
        steering_cmd = max(-self.max_steer, min(self.max_steer, steering_cmd))

        response = min(1.0, dt / self.steering_delay)
        self.steering_angle += (steering_cmd - self.steering_angle) * response

        self.x += self.speed * math.cos(self.yaw) * dt
        self.y += self.speed * math.sin(self.yaw) * dt
        self.yaw = wrap_to_pi(
            self.yaw + self.speed / self.wheelbase * math.tan(self.steering_angle) * dt
        )

        self.history_x.append(self.x)
        self.history_y.append(self.y)
        if len(self.history_x) > 900:
            self.history_x.pop(0)
            self.history_y.pop(0)

        return nearest_idx, target_idx, steering_cmd, cross_track_error

    def reset(self, path: ReferencePath) -> None:
        self.x = path.x[0] - 0.6
        self.y = path.y[0] - 2.6
        self.yaw = math.radians(20.0)
        self.steering_angle = 0.0
        self.history_x = [self.x]
        self.history_y = [self.y]


class PurePursuitDemo:
    def __init__(self) -> None:
        self.path = ReferencePath.make_track()
        self.vehicle = Vehicle(0.0, 0.0, 0.0)
        self.vehicle.reset(self.path)

        self.dt = 0.04
        self.lookahead = 2.1
        self.paused = False
        self.last_target_idx = 0
        self.last_steering_cmd = 0.0
        self.last_cross_track_error = 0.0

        self.fig = plt.figure(figsize=(11.5, 7.0))
        self.ax = self.fig.add_axes([0.06, 0.08, 0.74, 0.86])
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.set_title("Pure Pursuit lookahead demo")
        self.ax.set_xlabel("x [m]")
        self.ax.set_ylabel("y [m]")
        self.ax.grid(True, alpha=0.25)

        left_x, left_y, right_x, right_y = self.path.boundaries()
        self.ax.plot(left_x + [left_x[0]], left_y + [left_y[0]], color="0.45", lw=1.2)
        self.ax.plot(right_x + [right_x[0]], right_y + [right_y[0]], color="0.45", lw=1.2)
        self.ax.plot(self.path.x + [self.path.x[0]], self.path.y + [self.path.y[0]], "k--", lw=1.5)
        self.ax.plot([left_x[0], right_x[0]], [left_y[0], right_y[0]], color="#c0392b", lw=2.0)
        (self.trace_line,) = self.ax.plot([], [], color="#2b7bba", lw=2.0)
        (self.lookahead_line,) = self.ax.plot([], [], color="#ff7f0e", lw=1.7)
        (self.target_point,) = self.ax.plot([], [], "o", color="#ff7f0e", ms=8)

        self.lookahead_circle = Circle((0.0, 0.0), self.lookahead, fill=False, color="#ff7f0e", alpha=0.35)
        self.ax.add_patch(self.lookahead_circle)

        self.car_patch = Polygon(self.car_polygon(), closed=True, fc="#3ca370", ec="black", lw=1.0)
        self.ax.add_patch(self.car_patch)

        self.info_text = self.ax.text(
            0.02,
            0.98,
            "",
            transform=self.ax.transAxes,
            va="top",
            ha="left",
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )

        self.ax.set_xlim(min(self.path.x) - 3.5, max(self.path.x) + 3.5)
        self.ax.set_ylim(min(self.path.y) - 3.5, max(self.path.y) + 3.5)

        self.slider = self.make_lookahead_slider()
        self.slider.on_changed(self.on_lookahead_changed)

        reset_ax = self.fig.add_axes([0.84, 0.08, 0.11, 0.045])
        pause_ax = self.fig.add_axes([0.84, 0.14, 0.11, 0.045])
        self.reset_button = Button(reset_ax, "Reset")
        self.pause_button = Button(pause_ax, "Pause")
        self.reset_button.on_clicked(self.on_reset)
        self.pause_button.on_clicked(self.on_pause)

        self.animation = FuncAnimation(self.fig, self.update, interval=25, blit=False)

    def make_lookahead_slider(self) -> Slider:
        slider_ax = self.fig.add_axes([0.88, 0.25, 0.035, 0.58])
        try:
            return Slider(
                slider_ax,
                "Ld [m]",
                0.15,
                7.0,
                valinit=self.lookahead,
                valstep=0.05,
                orientation="vertical",
            )
        except TypeError:
            slider_ax.remove()
            slider_ax = self.fig.add_axes([0.18, 0.02, 0.55, 0.03])
            return Slider(slider_ax, "Ld [m]", 0.15, 7.0, valinit=self.lookahead, valstep=0.05)

    def car_polygon(self) -> list[tuple[float, float]]:
        length = 0.85
        width = 0.42
        corners = [
            (length / 2.0, width / 2.0),
            (length / 2.0, -width / 2.0),
            (-length / 2.0, -width / 2.0),
            (-length / 2.0, width / 2.0),
        ]
        cos_yaw = math.cos(self.vehicle.yaw)
        sin_yaw = math.sin(self.vehicle.yaw)
        return [
            (
                self.vehicle.x + cx * cos_yaw - cy * sin_yaw,
                self.vehicle.y + cx * sin_yaw + cy * cos_yaw,
            )
            for cx, cy in corners
        ]

    def on_lookahead_changed(self, value: float) -> None:
        self.lookahead = float(value)
        self.lookahead_circle.set_radius(self.lookahead)

    def on_reset(self, _event) -> None:
        self.vehicle.reset(self.path)

    def on_pause(self, _event) -> None:
        self.paused = not self.paused
        self.pause_button.label.set_text("Resume" if self.paused else "Pause")

    def update(self, _frame):
        if not self.paused:
            for _ in range(2):
                _nearest_idx, self.last_target_idx, self.last_steering_cmd, self.last_cross_track_error = self.vehicle.step(
                    self.path, self.lookahead, self.dt
                )

        target_x = self.path.x[self.last_target_idx]
        target_y = self.path.y[self.last_target_idx]

        self.trace_line.set_data(self.vehicle.history_x, self.vehicle.history_y)
        self.lookahead_line.set_data([self.vehicle.x, target_x], [self.vehicle.y, target_y])
        self.target_point.set_data([target_x], [target_y])
        self.lookahead_circle.center = (self.vehicle.x, self.vehicle.y)
        self.car_patch.set_xy(self.car_polygon())
        self.info_text.set_text(
            f"Ld = {self.lookahead:.2f} m\n"
            f"steer cmd = {math.degrees(self.last_steering_cmd):.1f} deg\n"
            f"actual steer = {math.degrees(self.vehicle.steering_angle):.1f} deg\n"
            f"cross-track = {self.last_cross_track_error:.2f} m\n"
            f"speed = {self.vehicle.speed:.1f} m/s"
        )

        return (
            self.trace_line,
            self.lookahead_line,
            self.target_point,
            self.lookahead_circle,
            self.car_patch,
            self.info_text,
        )

    def show(self) -> None:
        plt.show()


def main() -> None:
    PurePursuitDemo().show()


if __name__ == "__main__":
    main()
