#!/usr/bin/env python3
# Only-curing pipeline (no expansion)
# - Same overall sampled capacity as previous 3-carrier setup
# - Emit MPM Liquid (viscous=False) → cure to ElastoPlastic
# - Viewer 1000x700, direction jitter, timing/particle summaries

import time
import types
import math
import numpy as np
import genesis as gs
import genesis.utils.particle as pu

# ======================= 0) Tunables ========================================
METHOD                   = "MPM"
SIMULATION_LABEL         = "(Only curing)"
P_SIZE                   = 0.007              # particle diameter (m)
DT                       = 1e-3               # sim dt (s)
SUBSTEPS                 = 10                 # stability
GRAV                     = (0.0, 0.0, -9.81)
DURATION                 = 4.0                # simulated seconds

# Emitter (sphere-per-step)
DROPLET_SIZE             = 0.015              # sphere diameter (m)
EMIT_SPEED               = 5.0                # initial velocity (m/s)
EMIT_DIR_BASE            = np.array([0.0, 0.0, -1.0], dtype=np.float32)
EMIT_DIR_JITTER_DEG      = 3.0                # direction jitter (deg)

# Curing timing (from emitted → cured)
TAU_TO_P1_S              = 0.15
TAU_TO_P1_JITTER         = 0.14               # set >0 for per-particle jitter

# Kept here only to reproduce previous "capacity math" target
EXPANSION_RATIO          = 10.0               # used ONLY to match prior capacity

# Ground & bounds
LOWER_BOUND              = (-0.6, -0.6, -0.05)
UPPER_BOUND              = ( 0.6,  0.6,  0.65)
GROUND_Z                 = 0.0
COLLISION_MARGIN         = 0.6 * P_SIZE

# Materials / densities
RHO_P0                   = 1400.0
RHO_CURED                = RHO_P0             # keep same rho for apples-to-apples
ELASTO_E                 = 1.0e6
ELASTO_NU                = 0.05

# Drag (mild)
DRAG_LINEAR              = 1.25

# Auto-sizing heuristics
PACKING_EFF              = 0.90               # fraction “filled” at spacing p_size
SAFETY_MARGIN            = 1.30               # headroom for each carrier

# ======================= helpers: emission & sizing ==========================
def estimate_counts_sphere_per_step(dt, duration, sphere_diam, p_size, exp_ratio,
                                    tau_expand_dummy, tau_to_p1,
                                    packing_eff=0.9, margin=1.3):
    """
    Same estimator used before to reproduce the *previous three-carrier capacities*,
    even though we won't expand now. We then merge the two downstream capacities
    (Expanded + P1) into a single cured carrier so total sampled capacity matches.
    """
    steps = int(round(duration / dt))
    r = 0.5 * sphere_diam
    vol_per_step = (4.0/3.0) * math.pi * r**3
    per_particle_vol = (p_size ** 3) / max(packing_eff, 1e-6)
    n_per_step = max(1, int(math.ceil(vol_per_step / per_particle_vol)))
    n_emit_total = n_per_step * steps

    # windows (only TAU_TO_P1 matters here, but keep the same form)
    win_expand_steps = max(1, int(math.ceil(0.15 / dt)))  # dummy, just to match prior math
    win_p1_steps     = max(1, int(math.ceil(tau_to_p1 / dt)))

    # these three match the old 3-stage capacity math
    cap_emit_prev      = int(math.ceil(n_per_step * win_expand_steps * margin))
    cap_expanded_prev  = int(math.ceil(n_per_step * exp_ratio * win_p1_steps * margin))
    cap_p1_prev        = int(math.ceil(n_emit_total * exp_ratio * margin))
    return n_per_step, n_emit_total, cap_emit_prev, cap_expanded_prev, cap_p1_prev

def sphere_radius_for_particles(n_target, p_size, packing_eff=0.9):
    V = (n_target * (p_size ** 3)) / max(packing_eff, 1e-6)
    r_cubed = (3.0 * V) / (4.0 * math.pi)
    r = max((r_cubed ** (1.0/3.0)), 1.5 * p_size)
    return r

def jitter_direction(base_dir: np.ndarray, jitter_deg: float) -> np.ndarray:
    d = base_dir.astype(np.float32)
    d /= (np.linalg.norm(d) + 1e-12)
    if jitter_deg <= 0.0:
        return d
    sigma = math.tan(math.radians(jitter_deg))
    jitter = np.random.normal(0.0, sigma, size=3).astype(np.float32)
    dj = d + jitter
    dj /= (np.linalg.norm(dj) + 1e-12)
    return dj

# ======================= 1) Boot Genesis ====================================
gs.init()
scene = gs.Scene(
    vis_options = gs.options.VisOptions(
        show_world_frame = False,   # <--- hide the world frame
    ),
    sim_options = gs.options.SimOptions(dt=DT, substeps=SUBSTEPS, gravity=GRAV),
    mpm_options = gs.options.MPMOptions(
        lower_bound   = LOWER_BOUND,
        upper_bound   = UPPER_BOUND,
        particle_size = P_SIZE,
    ),
    viewer_options = gs.options.ViewerOptions(res=(1000, 700), max_FPS=None),
    show_viewer    = True,
)

# ======================= 2) Ground ==========================================
_ = scene.add_entity(
    morph=gs.morphs.Plane(),
    material=gs.materials.Rigid(
        needs_coup=True, coup_friction=1000.0, coup_softness=0.001, coup_restitution=0.0
    ),
    surface=gs.surfaces.Default(color=(0.50, 0.50, 0.50)),
)

# ======= 2.5) Auto-size carriers (merge old Expanded+P1 into one cured) =====
n_per_step, n_emit_total, cap_emit_prev, cap_expanded_prev, cap_p1_prev = \
    estimate_counts_sphere_per_step(
        DT, DURATION, DROPLET_SIZE, P_SIZE, EXPANSION_RATIO,
        0.15, TAU_TO_P1_S, PACKING_EFF, SAFETY_MARGIN
    )

cap_emit_new = cap_emit_prev
cap_cured_new = cap_expanded_prev + cap_p1_prev   # merge to keep total capacity same

r_emit  = sphere_radius_for_particles(cap_emit_new,  P_SIZE, PACKING_EFF)
r_cured = sphere_radius_for_particles(cap_cured_new, P_SIZE, PACKING_EFF)
Z_POS = 0.30

# ======================= 3) Carriers (Emit + Cured) =========================
P0_emit = scene.add_entity(
    morph    = gs.morphs.Sphere(radius=r_emit, pos=(0.0, 0.0, Z_POS)),
    material = gs.materials.MPM.Liquid(viscous=True, rho=RHO_P0),  # viscous=False (as asked)
    surface  = gs.surfaces.Default(vis_mode="particle", color=(0.15, 0.65, 1.00)),
)
P1_cured = scene.add_entity(
    morph    = gs.morphs.Sphere(radius=r_cured, pos=(0.0, 0.0, Z_POS)),
    material = gs.materials.MPM.ElastoPlastic(rho=RHO_CURED, E=ELASTO_E, nu=ELASTO_NU),
    surface  = gs.surfaces.Default(vis_mode="particle", color=(0.20, 0.95, 0.25)),
)

# ======================= 4) Emitter (into P0_emit) ==========================
emitter = scene.add_emitter(
    material      = gs.materials.MPM.Liquid(viscous=False, rho=RHO_P0),
    max_particles = P0_emit.n_particles,
    surface       = gs.surfaces.Default(vis_mode="particle", color=(0.15, 0.65, 1.00)),
)
emitter.set_entity(P0_emit)

def _write_block(entity, start_idx, pts_world, vels):
    n   = pts_world.shape[1]
    f   = scene.sim.cur_substep_local
    sol = entity._solver
    sol._kernel_set_particles_pos(f, entity.particle_start + start_idx, n, pts_world)
    sol._kernel_set_particles_vel(f, entity.particle_start + start_idx, n, vels)
    sol._kernel_set_particles_active(f, entity.particle_start + start_idx, n, gs.ACTIVE)

def _push_block(entity, head, pts_world, vels, stamp_fn=None):
    cap = entity.n_particles
    n   = pts_world.shape[1]
    rem = cap - head
    if n <= rem:
        _write_block(entity, head, pts_world, vels)
        if stamp_fn: stamp_fn(head, n)
        head = (head + n) % cap
    else:
        _write_block(entity, head, pts_world[:, :rem, :], vels[:, :rem, :])
        if stamp_fn: stamp_fn(head, rem)
        _write_block(entity, 0,    pts_world[:, rem:, :],  vels[:, rem:, :])
        if stamp_fn: stamp_fn(0,   n - rem)
        head = (n - rem) % cap
    return head

emit_head = 0
p1_head   = 0

birth_emit   = None
tau_to_p1_noise = None

# instrumentation counters (ever-activated)
activated_emit_total = 0
activated_p1_total   = 0

def emit_fixed(self, droplet_size=DROPLET_SIZE,
               pos=(0.0,0.0,1.0), base_direction=EMIT_DIR_BASE,
               speed=EMIT_SPEED, p_size=None, **kwargs):
    """Emit a SPHERE every step with small direction jitter (deg)."""
    global emit_head, step, activated_emit_total
    B = getattr(scene, "B", getattr(scene.sim, "_B", 1))

    dJ = jitter_direction(np.asarray(base_direction, dtype=gs.np_float), EMIT_DIR_JITTER_DEG)

    p_size = P_SIZE if p_size is None else p_size
    pts_local = pu.sphere_to_particles(
        p_size=p_size, radius=droplet_size * 0.5, sampler=self._entity.sampler
    ).astype(gs.np_float, copy=False)

    pts_world = pts_local + np.asarray(pos, dtype=gs.np_float)
    n         = pts_world.shape[0]

    pts_world = np.tile(pts_world[None], (B, 1, 1))
    v_single  = (speed * dJ).astype(gs.np_float, copy=False)
    vels      = np.tile(v_single, (B, n, 1))

    def stamp_emit(start, count):
        global activated_emit_total
        activated_emit_total += int(count)
        birth_emit[start:start+count] = step
        if TAU_TO_P1_JITTER > 0.0:
            tau_to_p1_noise[start:start+count] = np.random.uniform(0.0, TAU_TO_P1_JITTER, size=(count,))

    emit_head = _push_block(P0_emit, emit_head, pts_world, vels, stamp_emit)

emitter.emit = types.MethodType(emit_fixed, emitter)

if DRAG_LINEAR > 0.0:
    scene.add_force_field(gs.force_fields.Drag(linear=DRAG_LINEAR, quadratic=0.0))

# ======================= 5) Build & deactivate ==============================
scene.build()
B = getattr(scene, "B", getattr(scene.sim, "_B", 1))

def deactivate_all(entity):
    n = entity.n_particles
    act = np.full((B, n), gs.INACTIVE, dtype=np.int32)
    entity.set_active_arr(scene.sim.cur_substep_local, gs.tensor(act))

for e in (P0_emit, P1_cured):
    deactivate_all(e)

birth_emit = np.full((P0_emit.n_particles,), -1, dtype=np.int32)
if TAU_TO_P1_JITTER > 0.0:
    tau_to_p1_noise = np.zeros((P0_emit.n_particles,), dtype=np.float32)

# ======================= 6) Curing (Emit -> P1) =============================
def promote_to_p1(step, dt):
    global p1_head, activated_p1_total

    nE = P0_emit.n_particles
    posE = np.empty((B, nE, 3), dtype=np.float32)
    velE = np.empty((B, nE, 3), dtype=np.float32)
    CF   = np.empty((B, nE, 3, 3), dtype=np.float32)
    FF   = np.empty((B, nE, 3, 3), dtype=np.float32)
    Jp   = np.empty((B, nE),       dtype=np.float32)
    actE = np.empty((B, nE),       dtype=np.int32)
    P0_emit.get_frame(scene.sim.cur_substep_local, posE, velE, CF, FF, Jp, actE)

    ages = (step - birth_emit) * dt
    tau  = TAU_TO_P1_S if TAU_TO_P1_JITTER <= 0.0 else (TAU_TO_P1_S + (tau_to_p1_noise if tau_to_p1_noise is not None else 0.0))
    idxs = np.nonzero((actE[0] == gs.ACTIVE) & (birth_emit >= 0) & (ages >= tau))[0]
    if idxs.size == 0:
        return

    pos_sel = posE[:, idxs, :]
    vel_sel = velE[:, idxs, :]

    def stamp_p1(start, count):
        global activated_p1_total
        activated_p1_total += int(count)

    p1_head = _push_block(P1_cured, p1_head, pos_sel, vel_sel, stamp_fn=stamp_p1)

    # deactivate originals
    act0 = actE[0]
    act0[idxs] = gs.INACTIVE
    P0_emit.set_active_arr(scene.sim.cur_substep_local, gs.tensor(np.tile(act0[None], (B, 1)).astype(np.int32)))
    birth_emit[idxs] = -1
    if TAU_TO_P1_JITTER > 0.0:
        tau_to_p1_noise[idxs] = 0.0

# ======================= 7) Demo loop =======================================
dt          = scene.sim.dt
steps_total = int(DURATION / dt)

# Simple moving emitter path (helix-like)
radius = 0.22
omega  = 2 * np.pi / DURATION

t_wall_start = time.perf_counter()
for step in range(steps_total):
    t      = step * dt * 4
    angle  = omega * t
    x_off  = radius * np.cos(angle)
    y_off  = radius * np.sin(angle)
    z_emit = 0.1 + 0.1 * (angle / (2*np.pi))

    emit_pos = (x_off, y_off, z_emit)

    emitter.emit(
        droplet_size  = DROPLET_SIZE,   # sphere per step
        pos           = emit_pos,
        base_direction= EMIT_DIR_BASE,
        speed         = EMIT_SPEED,
        p_size        = P_SIZE,
    )

    promote_to_p1(step, dt)

    scene.step()
t_wall_end = time.perf_counter()

# ======================= 8) Summary =========================================
wall_time_s = (t_wall_end - t_wall_start)
sim_time_s  = steps_total * dt
avg_fps     = steps_total / wall_time_s if wall_time_s > 0 else float("nan")
real_over_sim = wall_time_s / sim_time_s if sim_time_s > 0 else float("nan")

def active_count(entity):
    n = entity.n_particles
    B_local = getattr(scene, "B", getattr(scene.sim, "_B", 1))
    d3  = np.empty((B_local, n, 3), dtype=np.float32)
    d33 = np.empty((B_local, n, 3, 3), dtype=np.float32)
    d1  = np.empty((B_local, n), dtype=np.float32)
    act = np.empty((B_local, n), dtype=np.int32)
    entity.get_frame(scene.sim.cur_substep_local, d3, d3, d33, d33, d1, act)
    return int(np.sum(act[0] == gs.ACTIVE))

active_emit  = active_count(P0_emit)
active_p1    = active_count(P1_cured)
active_total = active_emit + active_p1

activated_total_all   = activated_emit_total + activated_p1_total
sampled_capacity_all  = P0_emit.n_particles + P1_cured.n_particles

# domain volume
lb = np.array(LOWER_BOUND, dtype=float)
ub = np.array(UPPER_BOUND, dtype=float)
bbox_edges  = np.maximum(ub - lb, 0.0)
bbox_volume = float(bbox_edges[0] * bbox_edges[1] * bbox_edges[2])

def try_get_grid_density(sc):
    try:
        gd = getattr(sc._sim.mpm_solver, "grid_density", None)
        return float(gd) if gd is not None else None
    except Exception:
        return None

grid_density = try_get_grid_density(scene)
grid_density_str = f"{grid_density:.6g}" if grid_density else "—"

print("\n=== Simulation Summary (for table) ===")
print(f"Method: {METHOD}")
print(f"Simulation: {SIMULATION_LABEL}")
print(f"Bounded box: lower={LOWER_BOUND}, upper={UPPER_BOUND}, volume={bbox_volume:.6f} m^3")
print(f"# of particles (active now / ever activated / sampled capacity): "
      f"{active_total} / {activated_total_all} / {sampled_capacity_all}")
print(f"Particle size (m): {P_SIZE:.6g}")
print(f"Δt (s): {scene.sim.dt:.6g}   Substeps: {scene.sim.substeps}")
print(f"Grid density (cells/m): {grid_density_str}")
print(f"Avg FPS: {avg_fps:.3f}")
print(f"Wall time: {wall_time_s:.3f} s   Sim time: {sim_time_s:.3f} s   Wall/Sim: {real_over_sim:.3f}")

latex_row = (
    f"{METHOD} & {SIMULATION_LABEL} & "
    f"{activated_total_all} & "
    f"{P_SIZE:.6g} & "
    f"{scene.sim.dt:.6g} & "
    f"{scene.sim.substeps} & "
    f"{grid_density_str} & "
    f"— & — & — & — & "
    f"{avg_fps:.3f} \\\\"
)
print("\nLaTeX row:\n" + latex_row)

# Optional viewer (enabled above)
viewer = getattr(scene, "viewer", None)
if viewer is not None:
    viewer.run()