#!/usr/bin/env python3
"""
Renderable 2-DOF balance-board simulation using MuJoCo.

We assume you have identified a small-angle linear model:

    theta = [theta_x, theta_y]^T    (roll, pitch; radians)
    omega = [omega_x, omega_y]^T    (rad/s)

Dynamics:
    theta_dot = omega
    omega_dot = -K @ theta - D @ omega

where K and D are 2x2 matrices from your system ID script.

This script:
  * integrates that model with RK4,
  * converts [theta_x, theta_y] to a quaternion,
  * sets a free root joint in a MuJoCo model to that pose,
  * and uses mujoco.viewer to visualize the wobbling board.

You can pass K and D on the command line (row-major), or edit the defaults below.
Optional roll/pitch Coulomb-like friction coefficients (from analyze.py) can also
be provided to match the identified model.
Hold Shift and left-drag to stage an impulse arrow; release the mouse button to
inject that angular-velocity impulse (all configurable via CLI flags).
"""

import argparse
import os
import sys
import time
from typing import Optional

import glfw
import mujoco
import numpy as np


# ============================================================
# Linear 2-DOF dynamics
# ============================================================

def dynamics(x, K, D, friction=None, friction_eps=0.05):
    """
    Continuous-time dynamics for the 2-DOF model.

    x = [theta_x, theta_y, omega_x, omega_y]
    K, D: 2x2 numpy arrays
    friction: optional length-2 array of Coulomb-like coefficients applied to roll/pitch rates
    friction_eps: smoothing rate [rad/s] for the tanh regularization
    """
    theta = x[:2]
    omega = x[2:]
    theta_dot = omega
    coulomb = np.zeros(2)
    if friction is not None:
        fr = np.asarray(friction, dtype=float)
        if fr.size != 2:
            raise ValueError("friction must have exactly two entries (roll, pitch)")
        eps = max(float(friction_eps), 1e-6)
        coulomb = fr * np.tanh(omega / eps)
    omega_dot = -K @ theta - D @ omega - coulomb
    return np.concatenate([theta_dot, omega_dot])


def rk4_step(x, dt, K, D, friction=None, friction_eps=0.05):
    """One RK4 integration step."""
    k1 = dynamics(x, K, D, friction, friction_eps)
    k2 = dynamics(x + 0.5 * dt * k1, K, D, friction, friction_eps)
    k3 = dynamics(x + 0.5 * dt * k2, K, D, friction, friction_eps)
    k4 = dynamics(x + dt * k3, K, D, friction, friction_eps)
    return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


# ============================================================
# Initial disturbances
# ============================================================

def make_initial_conditions(disturbance, amp_deg, vel_deg_s):
    """
    Create initial [theta, omega] from disturbance spec.

    disturbance: one of
        'x_angle', 'y_angle', 'diag_angle',
        'x_vel', 'y_vel', 'diag_vel'

    amp_deg: angle amplitude (deg) for *_angle disturbances
    vel_deg_s: angular velocity amplitude (deg/s) for *_vel disturbances
    """
    amp = np.deg2rad(amp_deg)
    vel = np.deg2rad(vel_deg_s)

    theta0 = np.zeros(2)
    omega0 = np.zeros(2)

    if disturbance == "x_angle":
        theta0 = np.array([amp, 0.0])
    elif disturbance == "y_angle":
        theta0 = np.array([0.0, amp])
    elif disturbance == "diag_angle":
        theta0 = np.array([amp, amp])
    elif disturbance == "x_vel":
        omega0 = np.array([vel, 0.0])
    elif disturbance == "y_vel":
        omega0 = np.array([0.0, vel])
    elif disturbance == "diag_vel":
        omega0 = np.array([vel, vel])
    else:
        raise ValueError(f"Unknown disturbance '{disturbance}'")

    return theta0, omega0


# ============================================================
# MuJoCo model construction
# ============================================================

def build_model_xml(board_length=0.5969, board_width=0.40005, board_thickness=0.0254):
    """
    Build a minimal MJCF model with:
      * a plane
      * a board as a free body (6-DOF root joint)
    We will override its orientation each frame from our 2-DOF model.
    """
    # Height so the board sits just above the plane
    height = board_thickness * 0.5 + 0.01

    xml = f"""
    <mujoco model="balance_board">
      <compiler angle="radian" coordinate="local"/>
      <option gravity="0 0 -9.81" timestep="0.01"/>
      <default>
        <geom rgba="0.8 0.6 0.2 1" friction="1 0.005 0.0001"/>
      </default>

      <worldbody>
        <!-- Floor -->
        <geom name="floor" type="plane" size="2 2 0.1"
              rgba="0.7 0.7 0.7 1"/>

        <!-- Board with a free joint (root body) -->
        <body name="board" pos="0 0 {height}">
          <joint name="board_free" type="free"/>
          <geom type="box"
                size="{board_length/2.0} {board_width/2.0} {board_thickness/2.0}"/>
        </body>
      </worldbody>
    </mujoco>
    """
    return xml, height


# ============================================================
# Input handling helpers
# ============================================================


class MouseImpulseTracker:
    """Turns Shift+left-drag gestures into angular velocity impulses."""

    _MOD_FLAGS = {
        "none": 0,
        "shift": glfw.MOD_SHIFT,
        "ctrl": glfw.MOD_CONTROL,
        "alt": glfw.MOD_ALT,
        "super": glfw.MOD_SUPER,
    }

    _KEY_MAP = {
        "shift": (glfw.KEY_LEFT_SHIFT, glfw.KEY_RIGHT_SHIFT),
        "ctrl": (glfw.KEY_LEFT_CONTROL, glfw.KEY_RIGHT_CONTROL),
        "alt": (glfw.KEY_LEFT_ALT, glfw.KEY_RIGHT_ALT),
        "super": (glfw.KEY_LEFT_SUPER, glfw.KEY_RIGHT_SUPER),
    }

    def __init__(self, modifier: str, gain: float):
        if modifier not in self._MOD_FLAGS:
            raise ValueError(f"Unknown modifier '{modifier}'.")
        self.modifier = modifier
        self.gain = gain
        self._dragging = False
        self._drag_start: Optional[np.ndarray] = None
        self._drag_current: Optional[np.ndarray] = None
        self._pending = np.zeros(2, dtype=float)
        self._hover_point_world: Optional[np.ndarray] = None
        self._drag_start_world: Optional[np.ndarray] = None
        self._drag_current_world: Optional[np.ndarray] = None

    def update_hover_point(self, point: Optional[np.ndarray]) -> None:
        if point is None:
            self._hover_point_world = None
        else:
            self._hover_point_world = np.array(point, dtype=float)
            if self._dragging:
                self._drag_current_world = np.array(
                    self._hover_point_world, dtype=float
                )

    def handle_button(
        self,
        window,
        button,
        action,
        mods,
        fallback_point: Optional[np.ndarray],
    ) -> bool:
        if button != glfw.MOUSE_BUTTON_LEFT:
            return False

        if action == glfw.PRESS:
            if self._modifier_active(mods, window):
                self._dragging = True
                pos = np.array(glfw.get_cursor_pos(window), dtype=float)
                self._drag_start = pos.copy()
                self._drag_current = pos.copy()
                world_point = self._hover_point_world
                if world_point is None and fallback_point is not None:
                    world_point = np.array(fallback_point, dtype=float)
                if world_point is not None:
                    self._drag_start_world = world_point.copy()
                    self._drag_current_world = world_point.copy()
                return True
            self._dragging = False
            self._drag_start = None
            self._drag_current = None
            self._drag_start_world = None
            self._drag_current_world = None
        elif action == glfw.RELEASE and self._dragging:
            if self._drag_start is not None and self._drag_current is not None:
                delta = self._drag_current - self._drag_start
                self._pending += np.array([-delta[1], delta[0]]) * self.gain
            self._dragging = False
            self._drag_start = None
            self._drag_current = None
            self._drag_start_world = None
            self._drag_current_world = None
            return True

        return self._dragging

    def handle_move(self, window, xpos: float, ypos: float) -> bool:
        if not self._dragging:
            return False
        if not self._modifier_active(None, window):
            self._dragging = False
            self._drag_start = None
            self._drag_current = None
            self._drag_start_world = None
            self._drag_current_world = None
            return False

        if self._drag_start is None:
            self._drag_start = np.array([xpos, ypos], dtype=float)
            self._drag_current = np.array([xpos, ypos], dtype=float)
            return True

        if self._drag_current is None:
            self._drag_current = np.array([xpos, ypos], dtype=float)
            return True

        self._drag_current = np.array([xpos, ypos], dtype=float)
        if self._hover_point_world is not None:
            self._drag_current_world = np.array(
                self._hover_point_world, dtype=float
            )
        return True

    def consume_impulse(self) -> np.ndarray:
        impulse = self._pending.copy()
        self._pending[:] = 0.0
        return impulse

    def arrow_points(self) -> Optional[tuple[np.ndarray, np.ndarray]]:
        if not self._dragging or self._drag_start is None or self._drag_current is None:
            return None
        return self._drag_start.copy(), self._drag_current.copy()

    def world_arrow(self) -> Optional[tuple[np.ndarray, np.ndarray]]:
        if (
            not self._dragging
            or self._drag_start_world is None
            or self._drag_current_world is None
        ):
            return None
        return self._drag_start_world.copy(), self._drag_current_world.copy()

    def _modifier_active(self, mods: Optional[int], window) -> bool:
        if self.modifier == "none":
            return True

        flag = self._MOD_FLAGS[self.modifier]
        if mods is not None and (mods & flag):
            return True

        key_codes = self._KEY_MAP.get(self.modifier, ())
        return any(glfw.get_key(window, key) == glfw.PRESS for key in key_codes)

    def label(self) -> str:
        mapping = {
            "none": "click-drag",
            "shift": "Shift + drag",
            "ctrl": "Ctrl + drag",
            "alt": "Alt/Option + drag",
            "super": "Cmd/Win + drag",
        }
        return mapping.get(self.modifier, "drag")


def draw_instruction_overlay(
    context,
    viewport,
    tracker_label: str,
    impulse_gain: float,
    show_menu_hint: bool,
    impulses_enabled: bool,
    last_impulse: Optional[np.ndarray] = None,
    feedback_timer: float = 0.0,
):
    left = [
        "Controls:",
        "  Mouse left: orbit",
        "  Mouse right: pan",
        "  Mouse middle/scroll: zoom",
        "",
    ]
    if impulses_enabled:
        left += [
            "Impulse workflow:",
            f"  Hold {tracker_label} then drag to stage",
            "  Arrow shows impulse direction",
            "  Release to apply",
            f"  Gain: {impulse_gain:.2e} rad/s per pixel",
        ]
    else:
        left += ["Impulse workflow disabled (--no-mouse-impulses)."]

    if impulses_enabled:
        left += [" "]
        if last_impulse is not None and feedback_timer > 0.0:
            left += [
                "Last impulse:",
                f"  roll:  {last_impulse[0]:+.3f} rad/s",
                f"  pitch: {last_impulse[1]:+.3f} rad/s",
            ]
            left += [" "]
    left += [
        "Keyboard:",
        "  Esc: quit",
    ]
    if show_menu_hint:
        left += ["", "MuJoCo side menus are hidden in this custom viewer."]

    left_text = "\n".join(left)
    mujoco.mjr_overlay(
        mujoco.mjtFontScale.mjFONTSCALE_100,
        mujoco.mjtGridPos.mjGRID_TOPLEFT,
        viewport,
        left_text,
        "",
        context,
    )


def pick_board_point(
    window,
    xpos: float,
    ypos: float,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    opt: mujoco.MjvOption,
    scene: mujoco.MjvScene,
    board_body_id: int,
):
    width, height = glfw.get_window_size(window)
    if width <= 0 or height <= 0:
        return None

    relx = xpos / width
    rely = 1.0 - ypos / height
    aspect = width / height

    sel_point = np.zeros(3, dtype=np.float64)
    geomid = np.array([-1], dtype=np.int32)
    flexid = np.array([-1], dtype=np.int32)
    skinid = np.array([-1], dtype=np.int32)

    try:
        body_id = mujoco.mjv_select(
            model,
            data,
            opt,
            aspect,
            relx,
            rely,
            scene,
            sel_point,
            geomid,
            flexid,
            skinid,
        )
    except mujoco.Error:
        return None

    if body_id == board_body_id:
        return sel_point.copy()

    geom_id = int(geomid[0]) if geomid.size else -1
    if geom_id >= 0:
        geom_body = int(model.geom_bodyid[geom_id])
        if geom_body == board_body_id:
            return sel_point.copy()
    return None


def add_drag_indicator(
    scene: mujoco.MjvScene,
    tracker: Optional[MouseImpulseTracker],
    board_normal: np.ndarray,
) -> None:
    if tracker is None:
        return
    arrow = tracker.world_arrow()
    if arrow is None:
        return

    if scene.ngeom + 2 >= scene.maxgeom:
        return

    base, current = arrow
    base = np.asarray(base, dtype=float)
    current = np.asarray(current, dtype=float)
    board_normal = np.asarray(board_normal, dtype=float)
    board_normal = board_normal / max(np.linalg.norm(board_normal), 1e-6)

    raw_direction = current - base
    tangent = raw_direction - np.dot(raw_direction, board_normal) * board_normal
    length = np.linalg.norm(tangent)
    if length < 1e-5:
        return
    tangent /= length
    projected_tip = base + tangent * length
    tip = projected_tip + 0.03 * board_normal

    identity_mat = np.eye(3, dtype=float).reshape(9)

    sphere_geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        sphere_geom,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([0.03, 0.0, 0.0], dtype=np.float64),
        base.astype(np.float64),
        identity_mat,
        np.array([0.2, 0.7, 1.0, 0.9], dtype=np.float32),
    )
    scene.ngeom += 1

    arrow_geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        arrow_geom,
        mujoco.mjtGeom.mjGEOM_ARROW,
        np.zeros(3, dtype=np.float64),
        base.astype(np.float64),
        identity_mat,
        np.array([1.0, 0.4, 0.1, 0.9], dtype=np.float32),
    )
    mujoco.mjv_connector(
        arrow_geom,
        mujoco.mjtGeom.mjGEOM_ARROW,
        0.01,
        base.astype(np.float64),
        tip.astype(np.float64),
    )
    scene.ngeom += 1


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Renderable 2-DOF balance-board sim (MuJoCo)."
    )

    parser.add_argument(
        "--K", nargs=4, type=float,
        #default=[255.2761952, 0, 0, 148.67869445],
        #default=[293.84572338, -27.71568464, -27.71568464, 153.9643804],
        #default=[294.20442462, -27.00260791, -27.00260791, 152.35473913],
        default=[151.85662431, 0, 0, 99.12569694],
        help="Stiffness matrix entries K11 K12 K21 K22 (row-major)."
    )
    parser.add_argument(
        "--D", nargs=4, type=float,
        #default=[0, 0, 0, 2 * 0.11226245],
        #default=[0.52338527, 0.77013944, 0.77013944, 1.13322783],
        #default=[1.23746175, -0.36677313, -0.36677313, 1.16627366],
        default=[0.58446909, 0, 0, 0.84147819],
        help="Damping matrix entries D11 D12 D21 D22 (row-major)."
    )

    parser.add_argument(
        "--disturbance", type=str, default="x_angle",
        choices=["x_angle", "y_angle", "diag_angle",
                 "x_vel", "y_vel", "diag_vel"],
        help="Initial disturbance type."
    )
    parser.add_argument(
        "--amp_deg", type=float, default=5.0,
        help="Angle amplitude in degrees for *_angle disturbances."
    )
    parser.add_argument(
        "--vel_deg_s", type=float, default=30.0,
        help="Angular velocity amplitude in deg/s for *_vel disturbances."
    )

    parser.add_argument(
        "--t_final", type=float, default=15.0,
        help="Simulation duration [s]."
    )
    parser.add_argument(
        "--dt", type=float, default=0.01,
        help="Integration time step [s]."
    )
    parser.add_argument(
        "--real_time", action="store_true",
        help="Sleep to approximately match real-time."
    )
    parser.add_argument(
        "--friction", nargs=2, type=float, metavar=("ROLL", "PITCH"),
        default=[0.0, 0.0],
        help="Coulomb-like friction coefficients for roll/pitch (analysis output)."
    )
    parser.add_argument(
        "--friction_eps", type=float, default=0.05,
        help="Rate scale [rad/s] for smooth friction tanh regularization."
    )
    parser.add_argument(
        "--mouse_impulses",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable Shift+left-drag screen impulses (use --no-mouse-impulses to disable)."
    )
    parser.add_argument(
        "--impulse_gain", type=float, default=1e-3,
        help="Angular velocity impulse gain [rad/s per pixel]."
    )
    parser.add_argument(
        "--impulse_modifier",
        type=str,
        default="shift",
        choices=["shift", "ctrl", "alt", "super", "none"],
        help="Modifier key required while left-dragging to inject impulses."
    )
    parser.add_argument(
        "--allow_mjpython",
        action="store_true",
        help="Bypass macOS mjpython guard (custom GLFW window must be on the main thread)."
    )

    args = parser.parse_args()

    if (
        sys.platform == "darwin"
        and "mjpython" in os.path.basename(sys.executable)
        and not args.allow_mjpython
    ):
        raise SystemExit(
            "The custom GLFW viewer must run on the macOS main thread. "
            "When using the MuJoCo-provided 'mjpython' shim, user code runs on a "
            "worker thread and GLFW window creation fails. Please re-run using "
            "'python sim_mujoco.py ...' (or pass --allow_mjpython to bypass this "
            "guard if you know the risks)."
        )

    # ----- K and D -----
    if args.K is None or args.D is None:
        # Replace these with your ID results if you want defaults:
        print("No K/D provided; using placeholder defaults.")
        print("Pass --K and --D on the command line to use your identified matrices.")
        K = np.array([[5.0, 0.0],
                      [0.0, 4.0]])
        D = np.array([[0.8, 0.0],
                      [0.0, 0.7]])
    else:
        K = np.array(args.K, dtype=float).reshape(2, 2)
        D = np.array(args.D, dtype=float).reshape(2, 2)

    print("K =\n", K)
    print("D =\n", D)
    friction = np.array(args.friction, dtype=float)
    print("Friction coeffs (roll, pitch) =", friction)

    # Basic sanity checks
    if K.shape != (2, 2) or D.shape != (2, 2):
        raise ValueError("K and D must be 2x2 matrices.")

    if friction.shape != (2,):
        raise ValueError("Friction must have exactly two coefficients (roll, pitch).")

    # Warn if K or D are obviously non-symmetric
    if not np.allclose(K, K.T, atol=1e-6):
        print("[Warning] K is not symmetric; check that you passed it row-major as 4 entries.")
    if not np.allclose(D, D.T, atol=1e-6):
        print("[Warning] D is not symmetric; this may indicate a parameter ordering issue.")

    # ----- initial conditions -----
    theta0, omega0 = make_initial_conditions(
        args.disturbance, args.amp_deg, args.vel_deg_s
    )
    x = np.concatenate([theta0, omega0])

    print("Initial theta [deg] =", np.rad2deg(theta0))
    print("Initial omega [deg/s] =", np.rad2deg(omega0))

    # ----- MuJoCo model & data -----
    xml, board_height = build_model_xml()
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    try:
        board_body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "board"
        )
    except mujoco.Error as exc:
        raise RuntimeError("Board body 'board' not found in MJCF model.") from exc

    # Init pose at t=0
    quat = np.zeros(4, dtype=np.float64)
    euler = np.array([theta0[0], theta0[1], 0.0], dtype=np.float64)  # [rx, ry, rz]
    board_center_world = np.array([0.0, 0.0, board_height], dtype=float)
    board_normal_world = np.array([0.0, 0.0, 1.0], dtype=float)

    def sync_state_to_sim(state: np.ndarray, sim_time: float) -> None:
        theta_x, theta_y = state[0], state[1]
        euler[:] = [theta_x, theta_y, 0.0]
        mujoco.mju_euler2Quat(quat, euler, "xyz")
        data.qpos[0:3] = np.array([0.0, 0.0, board_height])
        data.qpos[3:7] = quat
        data.qvel[:] = 0.0
        data.qacc[:] = 0.0
        data.time = sim_time
        mujoco.mj_forward(model, data)
        board_center_world[:] = data.xpos[board_body_id]
        xmat = data.xmat[board_body_id].reshape(3, 3)
        board_normal_world[:] = xmat[:, 2]

    sync_state_to_sim(x, 0.0)

    # ----- custom GLFW viewer -----
    if not glfw.init():
        raise RuntimeError("Failed to initialize GLFW.")

    window = glfw.create_window(1280, 720, "Balance Board", None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("Failed to create GLFW window.")

    glfw.make_context_current(window)
    glfw.swap_interval(1)

    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.lookat = np.array([0.0, 0.0, board_height])
    cam.distance = 1.5
    cam.azimuth = 90.0
    cam.elevation = -20.0

    opt = mujoco.MjvOption()
    mujoco.mjv_defaultOption(opt)
    scene = mujoco.MjvScene(model, 2000)
    context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)

    mujoco.mjv_updateScene(
        model,
        data,
        opt,
        None,
        cam,
        mujoco.mjtCatBit.mjCAT_ALL,
        scene,
    )

    def framebuffer_size_callback(window, width, height):  # noqa: ARG001
        mujoco.mjr_resizeContext(model, context)

    glfw.set_framebuffer_size_callback(window, framebuffer_size_callback)

    mouse_buttons = {
        glfw.MOUSE_BUTTON_LEFT: False,
        glfw.MOUSE_BUTTON_RIGHT: False,
        glfw.MOUSE_BUTTON_MIDDLE: False,
    }
    last_cursor = np.array(glfw.get_cursor_pos(window))

    impulse_tracker = (
        MouseImpulseTracker(args.impulse_modifier, args.impulse_gain)
        if args.mouse_impulses
        else None
    )
    if impulse_tracker is not None:
        instruction = (
            f"Hold {impulse_tracker.label()} then drag; release to inject impulses"
        )
        print(
            f"Mouse impulses enabled: {impulse_tracker.label()} stages a drag arrow "
            f"(~{args.impulse_gain:.2e} rad/s per pixel) and applies on release."
        )
    else:
        instruction = "Mouse impulses disabled (--no-mouse-impulses)."
    glfw.set_window_title(window, f"Balance Board | {instruction}")

    def mouse_button_callback(window, button, action, mods):  # noqa: ARG001
        consumed = False
        cursor_x, cursor_y = glfw.get_cursor_pos(window)
        hover_point = pick_board_point(
            window,
            cursor_x,
            cursor_y,
            model,
            data,
            opt,
            scene,
            board_body_id,
        )
        if impulse_tracker is not None:
            impulse_tracker.update_hover_point(hover_point)
            fallback = hover_point
            consumed = impulse_tracker.handle_button(
                window,
                button,
                action,
                mods,
                None if fallback is None else np.array(fallback, dtype=float),
            )
        if not consumed and button in mouse_buttons:
            mouse_buttons[button] = action == glfw.PRESS

    glfw.set_mouse_button_callback(window, mouse_button_callback)

    def cursor_pos_callback(window, xpos, ypos):  # noqa: ARG001
        nonlocal last_cursor
        hover_point = pick_board_point(
            window,
            xpos,
            ypos,
            model,
            data,
            opt,
            scene,
            board_body_id,
        )
        if impulse_tracker is not None:
            impulse_tracker.update_hover_point(hover_point)
        handled = impulse_tracker.handle_move(window, xpos, ypos) if impulse_tracker else False
        dx = xpos - last_cursor[0]
        dy = ypos - last_cursor[1]
        last_cursor = np.array([xpos, ypos])

        if handled:
            return

        height = glfw.get_window_size(window)[1]
        if height <= 0:
            return

        action = None
        if mouse_buttons.get(glfw.MOUSE_BUTTON_LEFT, False):
            action = mujoco.mjtMouse.mjMOUSE_ROTATE_H
        elif mouse_buttons.get(glfw.MOUSE_BUTTON_RIGHT, False):
            action = mujoco.mjtMouse.mjMOUSE_MOVE_H
        elif mouse_buttons.get(glfw.MOUSE_BUTTON_MIDDLE, False):
            action = mujoco.mjtMouse.mjMOUSE_ZOOM

        if action is not None:
            mujoco.mjv_moveCamera(model, action, dx / height, dy / height, scene, cam)

    glfw.set_cursor_pos_callback(window, cursor_pos_callback)

    def scroll_callback(window, xoffset, yoffset):  # noqa: ARG001
        mujoco.mjv_moveCamera(
            model,
            mujoco.mjtMouse.mjMOUSE_ZOOM,
            0.0,
            -0.05 * yoffset,
            scene,
            cam,
        )

    glfw.set_scroll_callback(window, scroll_callback)

    def key_callback(window, key, scancode, action, mods):  # noqa: ARG001
        if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
            glfw.set_window_should_close(window, True)

    glfw.set_key_callback(window, key_callback)

    t = 0.0
    last_wall = time.time()
    last_impulse = np.zeros(2, dtype=float)
    impulse_feedback_timer = 0.0

    try:
        while not glfw.window_should_close(window) and t < args.t_final:
            if impulse_feedback_timer > 0.0:
                impulse_feedback_timer = max(0.0, impulse_feedback_timer - args.dt)

            if impulse_tracker is not None:
                impulse = impulse_tracker.consume_impulse()
                if np.any(impulse):
                    x[2:] += impulse
                    last_impulse = impulse.copy()
                    impulse_feedback_timer = 1.5

            x = rk4_step(
                x,
                args.dt,
                K,
                D,
                friction=friction,
                friction_eps=args.friction_eps,
            )
            t += args.dt

            sync_state_to_sim(x, t)
            mujoco.mjv_updateScene(
                model,
                data,
                opt,
                None,
                cam,
                mujoco.mjtCatBit.mjCAT_ALL,
                scene,
            )

            if impulse_tracker is not None:
                add_drag_indicator(
                    scene,
                    impulse_tracker,
                    board_normal_world,
                )

            fb_width, fb_height = glfw.get_framebuffer_size(window)
            win_width, win_height = glfw.get_window_size(window)
            if fb_width == 0 or fb_height == 0 or win_width == 0 or win_height == 0:
                glfw.poll_events()
                continue

            viewport = mujoco.MjrRect(0, 0, fb_width, fb_height)
            mujoco.mjr_render(viewport, scene, context)

            draw_instruction_overlay(
                context,
                viewport,
                impulse_tracker.label() if impulse_tracker else "N/A",
                args.impulse_gain,
                show_menu_hint=True,
                impulses_enabled=impulse_tracker is not None,
                last_impulse=last_impulse if impulse_tracker else None,
                feedback_timer=impulse_feedback_timer,
            )

            glfw.swap_buffers(window)
            glfw.poll_events()

            if args.real_time:
                now = time.time()
                sleep = args.dt - (now - last_wall)
                if sleep > 0:
                    time.sleep(sleep)
                last_wall = time.time()
    finally:
        mujoco.mjr_freeContext(context)
        mujoco.mjv_freeScene(scene)
        glfw.terminate()


if __name__ == "__main__":
    main()