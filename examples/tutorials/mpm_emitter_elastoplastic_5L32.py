#!/usr/bin/env python3
# mpm_emitter2_elastoplastic_fastshots.py
#
# Concentric-layer MPM print + timed pauses + 2 OFFSCREEN screenshots via the
# already-running pyglet viewer. We add a temporary camera node to the scene
# (under viewer.render_lock), render offscreen, then remove it.
#
# Photos saved to ./Photos as:
#   MPM_Liquid_E<...>_<YYYY-MM-DD_HH-MM-SS>_{above|front}.png
#   MPM_ElastoPlastic_E<...>_...
#   SPH_mu<...>_...

import os
import copy
from datetime import datetime
import types
import numpy as np
import imageio

import genesis as gs
import genesis.utils.particle as pu
from genesis.ext.pyrender.node import Node

# ─────────────────────────────────────────────────────────────────────────────
# 1) Init
# ─────────────────────────────────────────────────────────────────────────────
gs.init()

# ─────────────────────────────────────────────────────────────────────────────
# 2) Scene & solver options
# ─────────────────────────────────────────────────────────────────────────────
scene = gs.Scene(
    sim_options = gs.options.SimOptions(
        dt       = 1e-3,
        substeps = 2,
        gravity  = (0.0, 0.0, -9.81),
    ),
    mpm_options = gs.options.MPMOptions(
        lower_bound   = (-4, -4,  -1.0),
        upper_bound   = ( 4,  4,  1.5),
        particle_size = 0.03,
    ),
    viewer_options = gs.options.ViewerOptions(res=(800, 600)),
    show_viewer    = True,   # viewer runs in background thread automatically
)

# ─────────────────────────────────────────────────────────────────────────────
# 3) Ground
# ─────────────────────────────────────────────────────────────────────────────
ground = scene.add_entity(
    morph=gs.morphs.Plane(),
    material=gs.materials.Rigid(
        needs_coup=True,
        coup_friction=0.9,
        coup_softness=0.001,
        coup_restitution=0.0
    ),
    surface=gs.surfaces.Default(color=(0.5, 0.5, 0.5)),
)

# ─────────────────────────────────────────────────────────────────────────────
# 4) “Carrier” spheres (samplers only)
# ─────────────────────────────────────────────────────────────────────────────
carrier1 = scene.add_entity(
    morph    = gs.morphs.Sphere(radius=0.3, pos=(3.0, 0.0, 1.0)),
    material = gs.materials.MPM.ElastoPlastic(
        E=2.0e6, nu=0.30, rho=10000.0, sampler='pbs',
        yield_lower=0.55, yield_higher=0.85, use_von_mises=False,
    ),
    surface  = gs.surfaces.Default(vis_mode="particle", color=(0.8, 0.05, 0.05)),
)

carrier2 = scene.add_entity(
    morph    = gs.morphs.Sphere(radius=0.3, pos=(2.0, 0.0, 1.0)),
    material = gs.materials.MPM.ElastoPlastic(
        E=2.0e6, nu=0.30, rho=10000.0, sampler='pbs',
        yield_lower=0.05, yield_higher=0.35, use_von_mises=False,
    ),
    surface  = gs.surfaces.Default(vis_mode="particle", color=(0.8, 0.05, 0.05)),
)

# ─────────────────────────────────────────────────────────────────────────────
# 5) Printed material: MPM Liquid
# ─────────────────────────────────────────────────────────────────────────────
liq1 = gs.materials.MPM.Liquid(viscous=True)
# Optional: liq1.E = 5e5  # set if you want E in filename for MPM Liquid
emitter1 = scene.add_emitter(
    material      = liq1,
    max_particles = 50_000,
    surface       = gs.surfaces.Default(vis_mode="particle", color=(0.8, 0.3, 0.1)),
)
emitter1.set_entity(carrier1)

liq2 = gs.materials.MPM.Liquid(viscous=True)
# Optional: liq2.E = 5e5
emitter2 = scene.add_emitter(
    material      = liq2,
    max_particles = 50_000,
    surface       = gs.surfaces.Default(vis_mode="particle", color=(0.1, 0.4, 0.9)),
)
emitter2.set_entity(carrier2)

# ─────────────────────────────────────────────────────────────────────────────
# 6) Patch emit() → (B,N,3)
# ─────────────────────────────────────────────────────────────────────────────
def emit_fixed(self, droplet_shape="sphere", droplet_size=0.01,
               pos=(0.0,0.0,1.0), direction=(0.0,0.0,-1.0),
               speed=0.4, p_size=None, **kwargs):
    direction = np.asarray(direction, dtype=gs.np_float)
    direction /= (np.linalg.norm(direction) + gs.EPS)

    p_size = self._solver.particle_size if p_size is None else p_size
    pts_local = pu.sphere_to_particles(
        p_size=p_size, radius=droplet_size * 0.5, sampler=self._entity.sampler
    ).astype(gs.np_float, copy=False)

    pts_world = pts_local + np.asarray(pos, dtype=gs.np_float)
    pts_world = np.tile(pts_world[None], (self._sim._B, 1, 1))

    if not self._solver.boundary.is_inside(pts_world):
        gs.raise_exception("Emitted particles are outside the boundary.")

    v_single = (speed * direction).astype(gs.np_float, copy=False)
    vels = np.tile(v_single, (self._sim._B, pts_local.shape[0], 1))

    n   = pts_local.shape[0]
    idx = self._entity.particle_start + self._next_particle
    self._solver._kernel_set_particles_pos(self._sim.cur_substep_local, idx, n, pts_world)
    self._solver._kernel_set_particles_vel(self._sim.cur_substep_local, idx, n, vels)
    self._solver._kernel_set_particles_active(self._sim.cur_substep_local, idx, n, gs.ACTIVE)
    self._next_particle = (self._next_particle + n) % self._entity.n_particles

emitter1.emit = types.MethodType(emit_fixed, emitter1)
emitter2.emit = types.MethodType(emit_fixed, emitter2)

# A bit of drag so material settles
scene.add_force_field(gs.force_fields.Drag(linear=2.0, quadratic=0.0))

# ─────────────────────────────────────────────────────────────────────────────
# 7) Utilities
# ─────────────────────────────────────────────────────────────────────────────
def pause_seconds(seconds: float):
    dt = scene.sim.dt
    for _ in range(int(np.ceil(seconds / dt))):
        scene.step()

def photos_dir():
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        base_dir = os.getcwd()
    p = os.path.join(base_dir, "Photos")
    os.makedirs(p, exist_ok=True)
    return p

def fmt_num(val):
    if val is None: return "na"
    try:
        s = f"{val:.6g}" if 1e-3 <= abs(val) < 1e6 else f"{val:.3g}"
        return s.replace('.', 'p').replace('+', '').replace('-', 'm')
    except Exception:
        return "na"

def extract_E(mat):
    for name in ("E", "young_modulus", "youngs_modulus", "young"):
        if hasattr(mat, name):
            v = getattr(mat, name)
            if v is not None:
                try: return float(v)
                except: pass
    lam = getattr(mat, "lam", None)
    mu  = getattr(mat, "mu",  None)
    if lam is not None and mu is not None and abs(lam + mu) > 1e-12:
        lam = float(lam); mu = float(mu)
        return mu * (3.0*lam + 2.0*mu) / (lam + mu)
    K = getattr(mat, "bulk_modulus", None)
    if K is not None and mu is not None and abs(3.0*K + mu) > 1e-12:
        K = float(K); mu = float(mu)
        return 9.0*K*mu / (3.0*K + mu)
    return None

def extract_mu(mat):
    for name in ("mu", "viscosity"):
        if hasattr(mat, name):
            v = getattr(mat, name)
            if v is not None:
                try: return float(v)
                except: pass
    return None

def material_label_from_emitter(em):
    mat = getattr(em, "material", None) or getattr(em, "_material", None)
    if mat is None:
        return "UnknownMaterial"
    cls = mat.__class__.__name__
    mod = mat.__class__.__module__

    if "ElastoPlastic" in cls:
        E = extract_E(mat)
        return f"MPM_ElastoPlastic_E{fmt_num(E)}"
    if ".MPM" in mod and "Liquid" in cls:
        E = extract_E(mat)  # include E for MPM Liquid as requested
        return f"MPM_Liquid_E{fmt_num(E)}"
    if "SPH" in mod or "SPH" in cls:
        mu = extract_mu(mat)
        return f"SPH_mu{fmt_num(mu)}"
    E = extract_E(mat)
    return f"{cls}_E{fmt_num(E)}" if E is not None else cls

def dated_name(prefix: str, view: str):
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return f"{prefix}_{ts}_{view}.png"

def look_at_pose(eye, target, up=(0.0, 0.0, 1.0)):
    eye = np.asarray(eye, dtype=float)
    target = np.asarray(target, dtype=float)
    up = np.asarray(up, dtype=float)
    f = target - eye
    fn = np.linalg.norm(f)
    f = np.array([0.0, 0.0, -1.0]) if fn < 1e-12 else f / fn
    # Viewer convention: camera looks along -Z
    z = -f
    x = np.cross(up, z); xn = np.linalg.norm(x)
    if xn < 1e-12:
        up = np.array([1.0, 0.0, 0.0]) if abs(z[2]) > 0.9 else np.array([0.0, 0.0, 1.0])
        x = np.cross(up, z); xn = np.linalg.norm(x)
    x = x / (xn + 1e-12)
    y = np.cross(z, x); y = y / (np.linalg.norm(y) + 1e-12)
    T = np.eye(4); T[:3,0]=x; T[:3,1]=y; T[:3,2]=z; T[:3,3]=eye
    return T

def capture_with_running_viewer(scene, eye, target, out_path, viewport=(800, 600)):
    """
    Use the already-running pyglet viewer thread to render OFFSCREEN with
    our own camera node:
      1) create a camera Node at the desired look-at,
      2) ADD it to the viewer's scene under render_lock,
      3) call render_offscreen(camera_node, ...),
      4) REMOVE our temp node under render_lock,
      5) save PNG.
    """
    pv = scene.viewer._pyrender_viewer

    # Ensure viewer is ready
    try: pv.wait_until_initialized()
    except Exception: pass

    # Unthrottle & hide axes/overlays
    try: pv.viewer_flags["refresh_rate"] = 1000.0
    except Exception: pass
    for fn in ("off_world_frame", "off_link_frame", "off_camera_frustum"):
        try: getattr(pv.gs_context, fn)()
        except Exception: pass

    # Build our camera node (copy intrinsics from current camera)
    cam_copy = copy.copy(pv._camera_node.camera)
    cam_node = Node(matrix=look_at_pose(eye, target), camera=cam_copy)

    # 2) ADD node to scene under render_lock (so it's a valid main camera)
    pv.render_lock.acquire()
    try:
        pv.scene.add_node(cam_node)
    finally:
        pv.render_lock.release()

    # 3) Offscreen render (viewer thread will acquire lock internally)
    try:
        res = pv.render_offscreen(cam_node, None)   # returns (color[, depth])
        color = res[0] if isinstance(res, (list, tuple)) else res
    finally:
        # 4) REMOVE our temp node safely
        pv.render_lock.acquire()
        try:
            try: pv.scene.remove_node(cam_node)
            except Exception: pass
        finally:
            pv.render_lock.release()

    if color is None:
        # Fallback to reading the last color buffer
        try:
            color = pv._renderer.read_color_buf()
        except Exception:
            color = None

    if color is None:
        print(f"[WARN] Could not save screenshot: {out_path}")
        return False

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    imageio.imwrite(out_path, color)
    print(f"[INFO] Saved screenshot → {out_path}")
    return True

# ─────────────────────────────────────────────────────────────────────────────
# 8) Build & run — two concentric circles per layer, same linear speed
# ─────────────────────────────────────────────────────────────────────────────
scene.build()
dt = scene.sim.dt

# Try to make the live viewer as un-throttled as possible
try: scene.viewer._pyrender_viewer.viewer_flags["refresh_rate"] = 1000.0
except Exception: pass

# Adjustable parameters (pictures are taken AFTER the loop)
n_layers     = 1            # set 1 or 5 — photos still taken after stack is done
droplet_d    = 0.02
base_radius  = 0.20
r_inner      = base_radius
r_outer      = base_radius + droplet_d*2

path_speed   = 1.0
omega_inner  = path_speed / max(r_inner, 1e-6)
omega_outer  = path_speed / max(r_outer, 1e-6)

T_layer     = (2.0 * np.pi * r_outer) / path_speed
steps_layer = int(np.ceil(T_layer / dt))

settle_between_layers_s = 0.3   # pause between layers
settle_after_all_s      = 1.0   # pause before screenshots

# Print layers
for layer in range(n_layers):
    z_emit = (layer + 1) * (droplet_d + 0.01)
    theta1 = 0.0; theta2 = 0.0

    for _ in range(steps_layer):
        if theta1 < 2.0*np.pi - 1e-9:
            x1 = r_inner * np.cos(theta1)
            y1 = r_inner * np.sin(theta1)
            emitter1.emit(
                droplet_shape = "sphere",
                droplet_size  = droplet_d,
                pos           = (x1, y1, z_emit),
                direction     = (0.0, 0.0, -1.0),
                speed         = 1.0,
                p_size        = 0.03,
            )
            theta1 = min(theta1 + omega_inner * dt, 2.0*np.pi)

        if theta2 < 2.0*np.pi - 1e-9:
            x2 = r_outer * np.cos(theta2)
            y2 = r_outer * np.sin(theta2)
            emitter2.emit(
                droplet_shape = "square",
                droplet_size  = droplet_d,
                pos           = (x2, y2, z_emit),
                direction     = (0.0, 0.0, -1.0),
                speed         = 1.0,
                p_size        = 0.03,
            )
            theta2 = min(theta2 + omega_outer * dt, 2.0*np.pi)

        scene.step()

    if layer < n_layers - 1:
        pause_seconds(settle_between_layers_s)

# After all layers, let the stack rest before shots
pause_seconds(settle_after_all_s)

# Filenames
def material_label_from_emitter(em):
    mat = getattr(em, "material", None) or getattr(em, "_material", None)
    if mat is None:
        return "UnknownMaterial"
    cls = mat.__class__.__name__
    mod = mat.__class__.__module__
    def fmt_num(v):
        if v is None: return "na"
        s = f"{v:.6g}" if 1e-3 <= abs(v) < 1e6 else f"{v:.3g}"
        return s.replace('.', 'p').replace('+', '').replace('-', 'm')
    def extract_E(m):
        for name in ("E","young_modulus","youngs_modulus","young"):
            if hasattr(m,name):
                vv = getattr(m,name)
                if vv is not None:
                    try: return float(vv)
                    except: pass
        lam = getattr(m,"lam",None); mu = getattr(m,"mu",None)
        if lam is not None and mu is not None and abs(lam+mu)>1e-12:
            lam=float(lam); mu=float(mu)
            return mu*(3.0*lam+2.0*mu)/(lam+mu)
        K = getattr(m,"bulk_modulus",None)
        if K is not None and mu is not None and abs(3.0*K+mu)>1e-12:
            K=float(K); mu=float(mu)
            return 9.0*K*mu/(3.0*K+mu)
        return None
    def extract_mu(m):
        for name in ("mu","viscosity"):
            if hasattr(m,name):
                vv = getattr(m,name)
                if vv is not None:
                    try: return float(vv)
                    except: pass
        return None

    if "ElastoPlastic" in cls:
        return f"MPM_ElastoPlastic_E{fmt_num(extract_E(mat))}"
    if ".MPM" in mod and "Liquid" in cls:
        return f"MPM_Liquid_E{fmt_num(extract_E(mat))}"
    if "SPH" in mod or "SPH" in cls:
        return f"SPH_mu{fmt_num(extract_mu(mat))}"
    E = extract_E(mat)
    return f"{cls}_E{fmt_num(E)}" if E is not None else cls

def dated_name(prefix: str, view: str):
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return f"{prefix}_{ts}_{view}.png"

label  = material_label_from_emitter(emitter1) or material_label_from_emitter(emitter2)
outdir = photos_dir()

# Camera 1: from (2, 2, 1) looking at (0, 0, 0.25)
p1 = os.path.join(outdir, dated_name(label, "above"))
capture_with_running_viewer(scene, eye=(2.0, 2.0, 1.0), target=(0.0, 0.0, 0.25), out_path=p1)

# Camera 2: from (2, 2, 0.25) looking at (0, 0, 0.25)
p2 = os.path.join(outdir, dated_name(label, "front"))
capture_with_running_viewer(scene, eye=(2.0, 2.0, 0.25), target=(0.0, 0.0, 0.25), out_path=p2)

# Do NOT call scene.viewer.run(); the viewer is already running in a thread.