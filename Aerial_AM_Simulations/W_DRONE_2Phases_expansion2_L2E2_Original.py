#!/usr/bin/env python3
# mpm_liquid_expand_then_viscosify + DRONE FOLLOWER (uses set_pos / set_quat)

import types
import numpy as np
import genesis as gs
import genesis.utils.particle as pu

# ======================= 0) Tunables ========================================
P_SIZE                   = 0.007          # MPM particle size (grid-linked)

# Stage timings
TAU_EXPAND_S             = 0.15
TAU_EXPAND_JITTER        = 0.10
TAU_TO_P1_S              = 0.15
TAU_TO_P1_JITTER         = 0.00

# Expansion geometry & kinematics
EXPANSION_RATIO          = 5.0
EXPAND_RADIUS_MULTIPLIER = 5.0
MAX_RESAMPLE_TRIES       = 8

# Outward impulse for expansion
EXPANSION_VEL            = 0.5
EXPANSION_VEL_JITTER     = 0.4
VEL_RADIAL_WEIGHT        = 0.85

# Ground & spawn safety
GROUND_Z                 = 0.0
COLLISION_MARGIN         = 0.6 * P_SIZE

# Materials / densities
RHO_P0                   = 1400.0
MASS_COMPENSATE          = True

# Drag to tame splashing a bit
DRAG_LINEAR              = 1.0

# ===== Drone follow parameters ==============================================
URDF_PATH = "/home/omenrtx5090/Documents/Aerial_AM_Simulation_Nevo/Drone_files/robot.urdf"

# From your working script: nozzle → origin (base) offset (meters)
NOZZLE_TO_ORIGIN = np.array([-0.145989, 0.224300, 0.316818], dtype=np.float32)
NOZZLE_CLEARANCE = np.array([0.0, 0.0, 0.001], dtype=np.float32)  # 1 mm above emitter

# ======================= 1) Boot Genesis ====================================
gs.init()
scene = gs.Scene(
    sim_options = gs.options.SimOptions(dt=1e-3, substeps=5, gravity=(0.0, 0.0, -9.81)),
    mpm_options = gs.options.MPMOptions(
        lower_bound   = (-1, -1, -0.1),
        upper_bound   = ( 1,  1,  1.6),
        particle_size = P_SIZE,
    ),
    viewer_options = gs.options.ViewerOptions(res=(1000, 700)),#(res=(1120, 760)),
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

# ======================= 3) Pools ===========================================
POOL_RADIUS_EMIT = 0.15
rho_expanded = (RHO_P0 / EXPANSION_RATIO) if MASS_COMPENSATE else RHO_P0

P0_emit = scene.add_entity(
    morph    = gs.morphs.Sphere(radius=POOL_RADIUS_EMIT, pos=(0.0, 0.0, 0.5)),
    material = gs.materials.MPM.Liquid(viscous=False, rho=RHO_P0),
    surface  = gs.surfaces.Default(vis_mode="particle", color=(0.15, 0.65, 1.00)),
)

POOL_RADIUS_EXP = POOL_RADIUS_EMIT * (EXPANSION_RATIO ** (1/3))
P0_expanded = scene.add_entity(
    morph    = gs.morphs.Sphere(radius=POOL_RADIUS_EXP, pos=(0.0, 0.0, 0.5)),
    material = gs.materials.MPM.Liquid(viscous=False, rho=rho_expanded),
    surface  = gs.surfaces.Default(vis_mode="particle", color=(0.98, 0.78, 0.15)),
)

P1_viscous = scene.add_entity(
    morph    = gs.morphs.Sphere(radius=POOL_RADIUS_EXP * 1.5, pos=(0.0, 0.0, 0.50)),
    material = gs.materials.MPM.Liquid(viscous=True, rho=rho_expanded),
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

emit_head      = 0
expanded_head  = 0
p1_head        = 0

birth_emit     = None
birth_expanded = None
tau_emit_noise     = None
tau_to_p1_noise    = None

def emit_fixed(self, droplet_shape="sphere", droplet_size=0.01,
               pos=(0.0,0.0,1.0), direction=(0.0,0.0,-1.0),
               speed=0.6, p_size=None, **kwargs):
    global emit_head
    B = getattr(scene, "B", getattr(scene.sim, "_B", 1))

    direction = np.asarray(direction, dtype=gs.np_float)
    direction /= (np.linalg.norm(direction) + gs.EPS)

    p_size = P_SIZE if p_size is None else p_size
    pts_local = pu.sphere_to_particles(
        p_size=p_size,
        radius=droplet_size * 0.5,
        sampler=self._entity.sampler,
    ).astype(gs.np_float, copy=False)
    pts_world = pts_local + np.asarray(pos, dtype=gs.np_float)
    n         = pts_world.shape[0]

    pts_world = np.tile(pts_world[None], (B, 1, 1))
    v_single  = (speed * direction).astype(gs.np_float, copy=False)
    vels      = np.tile(v_single, (B, n, 1))

    def stamp_emit(start, count):
        birth_emit[start:start+count] = step
        if TAU_EXPAND_JITTER > 0.0:
            tau_emit_noise[start:start+count] = np.random.uniform(0.0, TAU_EXPAND_JITTER, size=(count,))

    emit_head = _push_block(P0_emit, emit_head, pts_world, vels, stamp_emit)

emitter.emit = types.MethodType(emit_fixed, emitter)

if DRAG_LINEAR > 0.0:
    scene.add_force_field(gs.force_fields.Drag(linear=DRAG_LINEAR, quadratic=0.0))

# ======================= 4.5) Add the drone (follows emitter) ===============
drone = scene.add_entity(
    morph=gs.morphs.URDF(
        file=URDF_PATH,
        fixed=False,
        pos=(0.0, 0.0, 0.5),   # temporary; we’ll update every step
    ),
    material=gs.materials.Rigid(rho=800.0),
)

def place_drone_at_emitter(emit_pos_xyz):
    """
    Keep drone upright; move so nozzle sits 1 mm above the emitter.
    Uses set_pos / set_quat (identity) as in your working script.
    """
    emit_pos = np.asarray(emit_pos_xyz, dtype=np.float32)
    nozzle_world = emit_pos + NOZZLE_CLEARANCE
    origin_world = nozzle_world + NOZZLE_TO_ORIGIN  # origin = nozzle + (nozzle→origin)
    drone.set_pos(tuple(origin_world))
    drone.set_quat((1.0, 0.0, 0.0, 0.0))  # identity quaternion, stays “straight”

# ======================= 5) Build & deactivate ==============================
scene.build()
B = getattr(scene, "B", getattr(scene.sim, "_B", 1))

def deactivate_all(entity):
    n = entity.n_particles
    act = np.full((B, n), gs.INACTIVE, dtype=np.int32)
    entity.set_active_arr(scene.sim.cur_substep_local, gs.tensor(act))

for e in (P0_emit, P0_expanded, P1_viscous):
    deactivate_all(e)

birth_emit     = np.full((P0_emit.n_particles,),     -1, dtype=np.int32)
birth_expanded = np.full((P0_expanded.n_particles,), -1, dtype=np.int32)

if TAU_EXPAND_JITTER > 0.0:
    tau_emit_noise = np.zeros((P0_emit.n_particles,), dtype=np.float32)
if TAU_TO_P1_JITTER > 0.0:
    tau_to_p1_noise = np.zeros((P0_expanded.n_particles,), dtype=np.float32)

# Initial placement
place_drone_at_emitter((0.0, 0.0, 0.5))

# ======================= 6) Helpers =========================================
def random_points_in_sphere(K, R):
    u   = np.random.rand(K).astype(np.float32)
    r   = (R * (u ** (1.0/3.0))).astype(np.float32)[:, None]
    v   = np.random.normal(size=(K,3)).astype(np.float32)
    v  /= (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8)
    return r * v

def in_bounds_mask(points, lower, upper):
    return np.all((points >= lower[None, :]) & (points <= upper[None, :]), axis=1)

def active_count(entity):
    n = entity.n_particles
    d3  = np.empty((B, n, 3), dtype=np.float32)
    d33 = np.empty((B, n, 3, 3), dtype=np.float32)
    d1  = np.empty((B, n), dtype=np.float32)
    act = np.empty((B, n), dtype=np.int32)
    entity.get_frame(scene.sim.cur_substep_local, d3, d3, d33, d33, d1, act)
    return int(np.sum(act[0] == gs.ACTIVE))

# ======================= 7) Stage A: expansion ===============================
def promote_expand(step, dt):
    global expanded_head

    nE = P0_emit.n_particles
    posE = np.empty((B, nE, 3), dtype=np.float32)
    velE = np.empty((B, nE, 3), dtype=np.float32)
    CF   = np.empty((B, nE, 3, 3), dtype=np.float32)
    FF   = np.empty((B, nE, 3, 3), dtype=np.float32)
    Jp   = np.empty((B, nE),       dtype=np.float32)
    actE = np.empty((B, nE),       dtype=np.int32)
    P0_emit.get_frame(scene.sim.cur_substep_local, posE, velE, CF, FF, Jp, actE)

    ages = (step - birth_emit) * dt
    tau  = TAU_EXPAND_S if TAU_EXPAND_JITTER <= 0.0 else (TAU_EXPAND_S + (tau_emit_noise if tau_emit_noise is not None else 0.0))
    eligible = (birth_emit >= 0) & (ages >= tau)
    mask = (actE[0] == gs.ACTIVE) & eligible
    idxs = np.nonzero(mask)[0]
    if idxs.size == 0:
        return

    boundary = P0_emit._solver.boundary
    lower = np.array(boundary.lower, dtype=np.float32)
    upper = np.array(boundary.upper, dtype=np.float32)

    pos_blocks, vel_blocks = [], []

    base_K  = int(np.floor(EXPANSION_RATIO))
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

        Vexp   = EXPANSION_VEL + (np.random.rand() * 2 - 1) * EXPANSION_VEL_JITTER
        rand_u = np.random.normal(size=dir_kept.shape).astype(np.float32)
        rand_u /= (np.linalg.norm(rand_u, axis=1, keepdims=True) + 1e-8)
        impulse_dir = VEL_RADIAL_WEIGHT * dir_kept + (1.0 - VEL_RADIAL_WEIGHT) * rand_u
        impulse_dir /= (np.linalg.norm(impulse_dir, axis=1, keepdims=True) + 1e-8)
        v_kept = base_vel[None, :] + Vexp * impulse_dir

        cand_B = np.tile(cand_kept[None, :, :], (B, 1, 1))
        vel_B  = np.tile(v_kept[None, :, :],    (B, 1, 1))
        pos_blocks.append(cand_B)
        vel_blocks.append(vel_B)

    if len(pos_blocks) == 0:
        return

    pos_out = np.concatenate(pos_blocks, axis=1)
    vel_out = np.concatenate(vel_blocks, axis=1)

    def stamp_expanded(start, count):
        birth_expanded[start:start+count] = step
        if TAU_TO_P1_JITTER > 0.0:
            tau_to_p1_noise[start:start+count] = np.random.uniform(0.0, TAU_TO_P1_JITTER, size=(count,))

    expanded_head = _push_block(P0_expanded, expanded_head, pos_out, vel_out, stamp_expanded)

    act0 = actE[0]
    act0[idxs] = gs.INACTIVE
    act_allB = np.tile(act0[None], (B, 1)).astype(np.int32)
    P0_emit.set_active_arr(scene.sim.cur_substep_local, gs.tensor(act_allB))
    birth_emit[idxs] = -1
    if TAU_EXPAND_JITTER > 0.0:
        tau_emit_noise[idxs] = 0.0

# ======================= 8) Stage B: convert to P1 ===========================
def promote_to_p1(step, dt):
    global p1_head

    nX = P0_expanded.n_particles
    posX = np.empty((B, nX, 3), dtype=np.float32)
    velX = np.empty((B, nX, 3), dtype=np.float32)
    CF   = np.empty((B, nX, 3, 3), dtype=np.float32)
    FF   = np.empty((B, nX, 3, 3), dtype=np.float32)
    Jp   = np.empty((B, nX),       dtype=np.float32)
    actX = np.empty((B, nX),       dtype=np.int32)
    P0_expanded.get_frame(scene.sim.cur_substep_local, posX, velX, CF, FF, Jp, actX)

    ages = (step - birth_expanded) * dt
    tau  = TAU_TO_P1_S if TAU_TO_P1_JITTER <= 0.0 else (TAU_TO_P1_S + (tau_to_p1_noise if tau_to_p1_noise is not None else 0.0))
    due  = (birth_expanded >= 0) & (ages >= tau)
    mask = (actX[0] == gs.ACTIVE) & due
    idxs = np.nonzero(mask)[0]
    if idxs.size == 0:
        return

    pos_sel = posX[:, idxs, :]
    vel_sel = velX[:, idxs, :]

    p1_head = _push_block(P1_viscous, p1_head, pos_sel, vel_sel, stamp_fn=None)

    act0 = actX[0]
    act0[idxs] = gs.INACTIVE
    act_allB = np.tile(act0[None], (B, 1)).astype(np.int32)
    P0_expanded.set_active_arr(scene.sim.cur_substep_local, gs.tensor(act_allB))
    birth_expanded[idxs] = -1
    if TAU_TO_P1_JITTER > 0.0:
        tau_to_p1_noise[idxs] = 0.0

# ======================= 9) Demo loop + DRONE FOLLOW ========================
duration    = 4.0
dt          = scene.sim.dt
steps_total = int(duration / dt)

radius = 0.22
omega  = 2 * np.pi / duration
next_report_step = 0

for step in range(steps_total):
    t      = step * dt * 4
    angle  = omega * t
    x_off  = radius * np.cos(angle)
    y_off  = radius * np.sin(angle)
    z_emit = 0.1 + 0.04 * (angle / (2*np.pi))

    emit_pos = (x_off, y_off, z_emit)

    # keep the drone nozzle 1 mm above the emitter
    place_drone_at_emitter(emit_pos)

    emitter.emit(
        droplet_shape = "square",
        droplet_size  = 0.015,
        pos           = emit_pos,
        direction     = (0.0, 0.0, -1.0),
        speed         = 5.0,
        p_size        = P_SIZE,
    )

    promote_expand(step, dt)
    promote_to_p1(step, dt)

    if step >= next_report_step:
        print(f"[t={step*dt:5.2f}s] P0_emit={active_count(P0_emit)}  P0_expanded={active_count(P0_expanded)}  P1={active_count(P1_viscous)}")
        next_report_step = step + int(0.5 / dt)

    scene.step()

scene.viewer.run()