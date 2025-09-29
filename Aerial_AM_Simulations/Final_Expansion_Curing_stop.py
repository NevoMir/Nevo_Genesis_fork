#!/usr/bin/env python3
# 5-layer deposition with alternating colors (red/purple), expansion + curing
# - Emits exactly N_LAYERS circular rings at increasing heights (+0.10 m each layer)
# - Alternates layer colors: red, purple (kept through expand+curing)
# - After 5 layers: emission stops but simulation continues indefinitely
# - Ground plane included for depth perception
# - Cured material yields more easily (lower yield thresholds)

import time
import types
import math
import numpy as np
import genesis as gs
import genesis.utils.particle as pu

# ======================= 0) Tunables ========================================
METHOD                   = "MPM"
SIMULATION_LABEL         = "(1): 5 Layers Alternating Colors"

# Discretization / physics
P_SIZE                   = 0.007              # particle diameter (m)
DT                       = 1e-3               # sim dt (s)
SUBSTEPS                 = 10                 # ↑ for stability with larger expansion
GRAV                     = (0.0, 0.0, -9.81)

# Emitter (sphere-per-step)
DROPLET_SIZE             = 0.015              # sphere diameter (m)
EMIT_SPEED               = 5.0                # initial velocity (m/s)
EMIT_DIR_BASE            = np.array([0.0, 0.0, -1.0], dtype=np.float32)
EMIT_DIR_JITTER_DEG      = 3.0                # requested emitter jitter (deg)

# Layer/path controls
N_LAYERS                 = 5
RADIUS                   = 0.22               # ring radius (m)
SECONDS_PER_LAYER        = 1.2                # time for one circular pass (s)
BASE_Z                   = 0.10               # first layer height (m)
LAYER_Z_STEP             = 0.10               # add 0.10 m per layer

# Expansion/cure staging
TAU_EXPAND_S             = 0.15
TAU_EXPAND_JITTER        = 0.10
TAU_TO_P1_S              = 0.15
TAU_TO_P1_JITTER         = 0.00
EXPANSION_RATIO          = 10.0               # ~10x volume
BASE_EXPAND_R_MULT       = 5.0
EXPAND_RADIUS_MULTIPLIER = BASE_EXPAND_R_MULT * (EXPANSION_RATIO / 5.0) ** (1/3)
MAX_RESAMPLE_TRIES       = 10

# Expansion kinematics (more noise)
EXPANSION_VEL            = 0.6                # base radial impulse (m/s)
EXPANSION_VEL_JITTER     = 0.6                # ± jitter on magnitude
VEL_RADIAL_WEIGHT        = 0.80               # blend with random dir
EXPANSION_EXTRA_NOISE    = 0.25               # extra isotropic speed (m/s) added

# Ground & bounds
GROUND_Z                 = 0.0
COLLISION_MARGIN         = 0.6 * P_SIZE

# Materials / densities
RHO_P0                   = 1400.0
MASS_COMPENSATE          = True

# Drag (slightly higher to help stability)
DRAG_LINEAR              = 1.25

# Auto-sizing heuristics
PACKING_EFF              = 0.90   # fraction “filled” at spacing p_size
SAFETY_MARGIN            = 1.30   # headroom for each carrier

# Domain (bounded box)
LOWER_BOUND              = (-0.6, -0.6, -0.1)
UPPER_BOUND              = ( 0.6,  0.6,  0.70)

# For pool sizing only
DURATION_FOR_SIZING      = max(N_LAYERS * SECONDS_PER_LAYER, 4.0)  # seconds

# Colors (RGB 0..1)
RED_COLOR     = (0.95, 0.20, 0.20)
PURPLE_COLOR  = (0.70, 0.25, 0.85)

# ======================= helpers: emission & sizing ==========================
def estimate_counts_sphere_per_step(dt, duration, sphere_diam, p_size, exp_ratio,
                                    tau_expand, tau_to_p1, packing_eff=0.9, margin=1.3):
    steps = int(round(duration / dt))
    r = 0.5 * sphere_diam
    vol_per_step = (4.0/3.0) * math.pi * r**3
    per_particle_vol = (p_size ** 3) / max(packing_eff, 1e-6)
    n_per_step = max(1, int(math.ceil(vol_per_step / per_particle_vol)))
    n_emit_total = n_per_step * steps

    win_expand_steps = max(1, int(math.ceil(tau_expand / dt)))
    win_p1_steps     = max(1, int(math.ceil(tau_to_p1 / dt)))

    cap_emit      = int(math.ceil(n_per_step * win_expand_steps * margin))
    cap_expanded  = int(math.ceil(n_per_step * exp_ratio * win_p1_steps * margin))
    cap_p1        = int(math.ceil(n_emit_total * exp_ratio * margin))
    return n_per_step, n_emit_total, cap_emit, cap_expanded, cap_p1

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

def random_points_in_sphere(K, R):
    u   = np.random.rand(K).astype(np.float32)
    r   = (R * (u ** (1.0/3.0))).astype(np.float32)[:, None]
    v   = np.random.normal(size=(K,3)).astype(np.float32)
    v  /= (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8)
    return r * v

def in_bounds_mask(points, lower, upper):
    lower = np.asarray(lower, dtype=np.float32)
    upper = np.asarray(upper, dtype=np.float32)
    return np.all((points >= lower[None, :]) & (points <= upper[None, :]), axis=1)

# ======================= 1) Boot Genesis ====================================
gs.init()
scene = gs.Scene(
    vis_options = gs.options.VisOptions(show_world_frame=False),
    sim_options = gs.options.SimOptions(dt=DT, substeps=SUBSTEPS, gravity=GRAV),
    mpm_options = gs.options.MPMOptions(
        lower_bound   = LOWER_BOUND,
        upper_bound   = UPPER_BOUND,
        particle_size = P_SIZE,
    ),
    viewer_options = gs.options.ViewerOptions(res=(2000, 1400), max_FPS=None),
    show_viewer    = True,   # GUI on
)

# ======================= 2) Ground (for depth) ==============================
_ = scene.add_entity(
    morph=gs.morphs.Plane(),
    material=gs.materials.Rigid(needs_coup=True, coup_friction=1000.0, coup_softness=0.001, coup_restitution=0.0),
    surface=gs.surfaces.Default(color=(0.50, 0.50, 0.50)),
)

# ======= 2.5) Auto-size spherical carriers ==================================
n_per_step, n_emit_total, cap_emit, cap_expanded, cap_p1 = estimate_counts_sphere_per_step(
    DT, DURATION_FOR_SIZING, DROPLET_SIZE, P_SIZE, EXPANSION_RATIO, TAU_EXPAND_S, TAU_TO_P1_S,
    PACKING_EFF, SAFETY_MARGIN
)
r_emit     = sphere_radius_for_particles(cap_emit,     P_SIZE, PACKING_EFF)
r_expanded = sphere_radius_for_particles(cap_expanded, P_SIZE, PACKING_EFF)
r_p1       = sphere_radius_for_particles(cap_p1,       P_SIZE, PACKING_EFF)
Z_POS = 0.30

# ======================= 3) Carriers (two color pipelines) ==================
rho_expanded = (RHO_P0 / EXPANSION_RATIO) if MASS_COMPENSATE else RHO_P0

emit_entities      = {}
expanded_entities  = {}
p1_entities        = {}
entity_to_key      = {}   # reverse lookup for the emitter

# ---- RED pipeline
emit_entities["red"] = scene.add_entity(
    morph    = gs.morphs.Sphere(radius=r_emit, pos=(0.0, 0.0, Z_POS)),
    material = gs.materials.MPM.Liquid(viscous=True, rho=RHO_P0),
    surface  = gs.surfaces.Default(vis_mode="particle", color=RED_COLOR),
)
expanded_entities["red"] = scene.add_entity(
    morph    = gs.morphs.Sphere(radius=r_expanded, pos=(0.0, 0.0, Z_POS)),
    material = gs.materials.MPM.Liquid(viscous=True, rho=rho_expanded),
    surface  = gs.surfaces.Default(vis_mode="particle", color=RED_COLOR),
)
p1_entities["red"] = scene.add_entity(
    morph    = gs.morphs.Sphere(radius=r_p1, pos=(0.0, 0.0, Z_POS)),
    material = gs.materials.MPM.ElastoPlastic(
        rho=rho_expanded, E=1.0e6, nu=0.05,
        yield_lower=0.005,   # easier plastic flow
        yield_higher=0.001,
        use_von_mises=True
    ),
    surface  = gs.surfaces.Default(vis_mode="particle", color=RED_COLOR),
)

# ---- PURPLE pipeline
emit_entities["purple"] = scene.add_entity(
    morph    = gs.morphs.Sphere(radius=r_emit, pos=(0.0, 0.0, Z_POS)),
    material = gs.materials.MPM.Liquid(viscous=True, rho=RHO_P0),
    surface  = gs.surfaces.Default(vis_mode="particle", color=PURPLE_COLOR),
)
expanded_entities["purple"] = scene.add_entity(
    morph    = gs.morphs.Sphere(radius=r_expanded, pos=(0.0, 0.0, Z_POS)),
    material = gs.materials.MPM.Liquid(viscous=True, rho=rho_expanded),
    surface  = gs.surfaces.Default(vis_mode="particle", color=PURPLE_COLOR),
)
p1_entities["purple"] = scene.add_entity(
    morph    = gs.morphs.Sphere(radius=r_p1, pos=(0.0, 0.0, Z_POS)),
    material = gs.materials.MPM.ElastoPlastic(
        rho=rho_expanded, E=1.0e6, nu=0.05,
        yield_lower=0.005,
        yield_higher=0.001,
        use_von_mises=True
    ),
    surface  = gs.surfaces.Default(vis_mode="particle", color=PURPLE_COLOR),
)

for k, ent in emit_entities.items():
    entity_to_key[ent] = k

# ======================= 4) Emitter (re-targeted per layer color) ===========
# One emitter; we switch its target entity when the layer color changes.
some_emit_entity = emit_entities["red"]
emitter = scene.add_emitter(
    material      = gs.materials.MPM.Liquid(viscous=False, rho=RHO_P0),
    max_particles = some_emit_entity.n_particles,
    surface       = gs.surfaces.Default(vis_mode="particle", color=(0.15, 0.65, 1.00)),
)
emitter.set_entity(some_emit_entity)

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

emit_heads      = {"red": 0, "purple": 0}
expanded_heads  = {"red": 0, "purple": 0}
p1_heads        = {"red": 0, "purple": 0}

birth_emit         = {}
birth_expanded     = {}
tau_emit_noise     = {}
tau_to_p1_noise    = {}

activated_emit_total     = {"red": 0, "purple": 0}
activated_expanded_total = {"red": 0, "purple": 0}
activated_p1_total       = {"red": 0, "purple": 0}

# step index used to stamp births inside emitter.emit
cur_step_for_emit = 0

def emit_fixed(self, droplet_size=DROPLET_SIZE,
               pos=(0.0,0.0,1.0), base_direction=EMIT_DIR_BASE,
               speed=EMIT_SPEED, p_size=None, **kwargs):
    """Emit a SPHERE every step with small direction jitter (deg) into current self._entity."""
    global activated_emit_total, cur_step_for_emit, emit_heads
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

    key = entity_to_key[self._entity]

    def stamp_emit(start, count):
        activated_emit_total[key] += int(count)
        birth_emit[key][start:start+count] = cur_step_for_emit
        if TAU_EXPAND_JITTER > 0.0:
            tau_emit_noise[key][start:start+count] = np.random.uniform(0.0, TAU_EXPAND_JITTER, size=(count,))

    emit_heads[key] = _push_block(self._entity, emit_heads[key], pts_world, vels, stamp_emit)

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

for k in ("red", "purple"):
    for e in (emit_entities[k], expanded_entities[k], p1_entities[k]):
        deactivate_all(e)

for k in ("red", "purple"):
    birth_emit[k]     = np.full((emit_entities[k].n_particles,),     -1, dtype=np.int32)
    birth_expanded[k] = np.full((expanded_entities[k].n_particles,), -1, dtype=np.int32)
    tau_emit_noise[k] = np.zeros((emit_entities[k].n_particles,), dtype=np.float32) if TAU_EXPAND_JITTER > 0.0 else None
    tau_to_p1_noise[k]= np.zeros((expanded_entities[k].n_particles,), dtype=np.float32) if TAU_TO_P1_JITTER > 0.0 else None

# ======================= 6) Expansion / Curing (per-color) ==================
def promote_expand_for(key, step, dt):
    global expanded_heads, activated_expanded_total

    src = emit_entities[key]
    dst = expanded_entities[key]

    nE = src.n_particles
    posE = np.empty((B, nE, 3), dtype=np.float32)
    velE = np.empty((B, nE, 3), dtype=np.float32)
    CF   = np.empty((B, nE, 3, 3), dtype=np.float32)
    FF   = np.empty((B, nE, 3, 3), dtype=np.float32)
    Jp   = np.empty((B, nE),       dtype=np.float32)
    actE = np.empty((B, nE),       dtype=np.int32)
    src.get_frame(scene.sim.cur_substep_local, posE, velE, CF, FF, Jp, actE)

    ages = (step - birth_emit[key]) * dt
    if TAU_EXPAND_JITTER <= 0.0 or tau_emit_noise[key] is None:
        tau = TAU_EXPAND_S
    else:
        tau = TAU_EXPAND_S + tau_emit_noise[key]

    eligible = (birth_emit[key] >= 0) & (ages >= tau)
    mask = (actE[0] == gs.ACTIVE) & eligible
    idxs = np.nonzero(mask)[0]
    if idxs.size == 0:
        return

    boundary = src._solver.boundary
    lower = np.array(boundary.lower, dtype=np.float32)
    upper = np.array(boundary.upper, dtype=np.float32)

    pos_blocks, vel_blocks = [], []

    base_K  = int(math.floor(EXPANSION_RATIO))
    extra_p = EXPANSION_RATIO - base_K
    R_samp  = EXPAND_RADIUS_MULTIPLIER * P_SIZE

    for idx in idxs:
        K_target = base_K + (1 if np.random.rand() < extra_p else 0)
        if K_target <= 0:
            continue

        base_pos = posE[0, idx, :].astype(np.float32)
        base_vel = velE[0, idx, :].astype(np.float32)

        kept_pos, kept_dir = [], []
        tries = 0
        while len(kept_pos) < K_target and tries < MAX_RESAMPLE_TRIES:
            need    = K_target - len(kept_pos)
            offsets = random_points_in_sphere(need, R_samp)
            dirs    = offsets / (np.linalg.norm(offsets, axis=1, keepdims=True) + 1e-8)
            cand    = base_pos[None, :] + offsets
            cand[:, 2] = np.maximum(cand[:, 2], GROUND_Z + COLLISION_MARGIN)
            inb = in_bounds_mask(cand, lower, upper)
            if np.any(inb):
                kept_pos.append(cand[inb])
                kept_dir.append(dirs[inb])
            tries += 1

        if len(kept_pos) == 0:
            continue

        cand_kept = np.concatenate(kept_pos, axis=0)[:K_target]
        dir_kept  = np.concatenate(kept_dir, axis=0)[:K_target]

        Vexp   = EXPANSION_VEL + (np.random.rand(dir_kept.shape[0]) * 2 - 1) * EXPANSION_VEL_JITTER
        rand_u = np.random.normal(size=dir_kept.shape).astype(np.float32)
        rand_u /= (np.linalg.norm(rand_u, axis=1, keepdims=True) + 1e-8)
        impulse_dir = VEL_RADIAL_WEIGHT * dir_kept + (1.0 - VEL_RADIAL_WEIGHT) * rand_u
        impulse_dir /= (np.linalg.norm(impulse_dir, axis=1, keepdims=True) + 1e-8)

        extra_noise = np.random.normal(size=dir_kept.shape).astype(np.float32)
        extra_noise /= (np.linalg.norm(extra_noise, axis=1, keepdims=True) + 1e-8)

        v_kept = base_vel[None, :] + (Vexp[:, None] * impulse_dir) + (EXPANSION_EXTRA_NOISE * extra_noise)

        pos_blocks.append(np.tile(cand_kept[None, :, :], (B, 1, 1)))
        vel_blocks.append(np.tile(v_kept[None, :, :],    (B, 1, 1)))

    if len(pos_blocks) == 0:
        return

    pos_out = np.concatenate(pos_blocks, axis=1)
    vel_out = np.concatenate(vel_blocks, axis=1)

    def stamp_expanded(start, count):
        activated_expanded_total[key] += int(count)
        birth_expanded[key][start:start+count] = step
        if TAU_TO_P1_JITTER > 0.0 and tau_to_p1_noise[key] is not None:
            tau_to_p1_noise[key][start:start+count] = np.random.uniform(0.0, TAU_TO_P1_JITTER, size=(count,))

    expanded_heads[key] = _push_block(dst, expanded_heads[key], pos_out, vel_out, stamp_expanded)

    # deactivate originals that were promoted
    act0 = actE[0]
    act0[idxs] = gs.INACTIVE
    src.set_active_arr(scene.sim.cur_substep_local, gs.tensor(np.tile(act0[None], (B, 1)).astype(np.int32)))
    birth_emit[key][idxs] = -1
    if TAU_EXPAND_JITTER > 0.0 and tau_emit_noise[key] is not None:
        tau_emit_noise[key][idxs] = 0.0

def promote_to_p1_for(key, step, dt):
    global p1_heads, activated_p1_total

    src = expanded_entities[key]
    dst = p1_entities[key]

    nX = src.n_particles
    posX = np.empty((B, nX, 3), dtype=np.float32)
    velX = np.empty((B, nX, 3), dtype=np.float32)
    CF   = np.empty((B, nX, 3, 3), dtype=np.float32)
    FF   = np.empty((B, nX, 3, 3), dtype=np.float32)
    Jp   = np.empty((B, nX),       dtype=np.float32)
    actX = np.empty((B, nX),       dtype=np.int32)
    src.get_frame(scene.sim.cur_substep_local, posX, velX, CF, FF, Jp, actX)

    ages = (step - birth_expanded[key]) * dt
    if TAU_TO_P1_JITTER <= 0.0 or tau_to_p1_noise[key] is None:
        tau = TAU_TO_P1_S
    else:
        tau = TAU_TO_P1_S + tau_to_p1_noise[key]
    due  = (birth_expanded[key] >= 0) & (ages >= tau)
    idxs = np.nonzero((actX[0] == gs.ACTIVE) & due)[0]
    if idxs.size == 0:
        return

    pos_sel = posX[:, idxs, :]
    vel_sel = velX[:, idxs, :]

    def stamp_p1(start, count):
        activated_p1_total[key] += int(count)

    p1_heads[key] = _push_block(dst, p1_heads[key], pos_sel, vel_sel, stamp_fn=stamp_p1)

    act0 = actX[0]
    act0[idxs] = gs.INACTIVE
    src.set_active_arr(scene.sim.cur_substep_local, gs.tensor(np.tile(act0[None], (B, 1)).astype(np.int32)))
    birth_expanded[key][idxs] = -1
    if TAU_TO_P1_JITTER > 0.0 and tau_to_p1_noise[key] is not None:
        tau_to_p1_noise[key][idxs] = 0.0

# ======================= 8) Emission loop (5 layers) + Indefinite wait ======
dt = scene.sim.dt
omega = 2.0 * math.pi / SECONDS_PER_LAYER

angle = 0.0
layer_idx = 0
emission_enabled = True

print("\n[INFO] Starting deposition of 5 layers (alternating red/purple)…\n")

step = 0
try:
    while True:
        if emission_enabled:
            # choose layer color
            key = "red" if (layer_idx % 2 == 0) else "purple"
            emitter.set_entity(emit_entities[key])

            # circular path at current height
            x_off  = RADIUS * math.cos(angle)
            y_off  = RADIUS * math.sin(angle)
            z_emit = BASE_Z + layer_idx * LAYER_Z_STEP
            emit_pos = (x_off, y_off, z_emit)

            cur_step_for_emit = step  # for birth timestamp

            emitter.emit(
                droplet_size  = DROPLET_SIZE,   # sphere per step
                pos           = emit_pos,
                base_direction= EMIT_DIR_BASE,
                speed         = EMIT_SPEED,
                p_size        = P_SIZE,
            )

            angle += omega * dt
            if angle >= 2.0 * math.pi:
                angle -= 2.0 * math.pi
                layer_idx += 1
                if layer_idx >= N_LAYERS:
                    emission_enabled = False
                    print("[INFO] Finished emitting 5 layers. Continuing simulation indefinitely…")

        # Always process transitions & step physics
        promote_expand_for("red",    step, dt)
        promote_expand_for("purple", step, dt)
        promote_to_p1_for("red",     step, dt)
        promote_to_p1_for("purple",  step, dt)

        scene.step()
        step += 1

except KeyboardInterrupt:
    print("\n[INFO] Stopped by user.\n")

# ======================= 9) Summary =========================================
def active_count(entity):
    n = entity.n_particles
    B_local = getattr(scene, "B", getattr(scene.sim, "_B", 1))
    d3  = np.empty((B_local, n, 3), dtype=np.float32)
    d33 = np.empty((B_local, n, 3, 3), dtype=np.float32)
    d1  = np.empty((B_local, n), dtype=np.float32)
    act = np.empty((B_local, n), dtype=np.int32)
    entity.get_frame(scene.sim.cur_substep_local, d3, d3, d33, d33, d1, act)
    return int(np.sum(act[0] == gs.ACTIVE))

active_total = 0
for k in ("red","purple"):
    active_total += active_count(emit_entities[k])
    active_total += active_count(expanded_entities[k])
    active_total += active_count(p1_entities[k])

# domain volume
lb = np.array(LOWER_BOUND, dtype=float)
ub = np.array(UPPER_BOUND, dtype=float)
bbox_edges = np.maximum(ub - lb, 0.0)
bbox_volume = float(bbox_edges[0] * bbox_edges[1] * bbox_edges[2])

def try_get_grid_density(sc):
    try:
        gd = getattr(sc._sim.mpm_solver, "grid_density", None)
        return float(gd) if gd is not None else None
    except Exception:
        return None

grid_density = try_get_grid_density(scene)
grid_density_str = f"{grid_density:.6g}" if grid_density else "—"

print("\n=== Simulation Summary ===")
print(f"Method: {METHOD}")
print(f"Simulation: {SIMULATION_LABEL}")
print(f"Bounded box: lower={LOWER_BOUND}, upper={UPPER_BOUND}, volume={bbox_volume:.6f} m^3")
print(f"# of particles (active now): {active_total}")
print(f"Particle size (m): {P_SIZE:.6g}")
print(f"Δt (s): {scene.sim.dt:.6g}   Substeps: {scene.sim.substeps}")
print(f"Grid density (cells/m): {grid_density_str}")

# Keep the viewer responsive (it already keeps running in the main loop)
viewer = getattr(scene, "viewer", None)
if viewer is not None:
    print("[INFO] Viewer remains open; close the window to exit.")
    viewer.run()