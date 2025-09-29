#!/usr/bin/env python3
# Back to simple omnidirectional expansion (no occupancy gating)
# - Viewer 1000x700
# - Expansion volume ~10x (EXPANSION_RATIO=10) + radius scaled
# - Stability: substeps=8, mild extra drag
# - Extra random speed at expansion
# - Emitter direction jitter = 3 deg
# - Domain prints & particle/timing summaries

import time
import types
import math
import numpy as np
import genesis as gs
import genesis.utils.particle as pu

# High-level: Simulates droplets emitted from a drone nozzle, expand (foam-like) after a delay,
# then cure into a viscoplastic phase. Uses ring buffers for each phase to manage particles.

# ======================= 0) Tunables ========================================
# Core simulation controls (dt, substeps, gravity, duration)
METHOD                   = "MPM"
SIMULATION_LABEL         = "(1): 5 Layers Structure"
P_SIZE                   = 0.007              # particle diameter (m)
DT                       = 1e-3               # sim dt (s)
SUBSTEPS                 = 10                  # ↑ for stability with larger expansion
GRAV                     = (0.0, 0.0, -9.81)
DURATION                 = 4.0                # seconds of simulated time

# Emitter (sphere-per-step) — each step we emit a spherical droplet of particles
DROPLET_SIZE             = 0.015              # sphere diameter (m)
EMIT_SPEED               = 5.0                # initial velocity (m/s)
EMIT_DIR_BASE            = np.array([0.0, 0.0, -1.0], dtype=np.float32)
EMIT_DIR_JITTER_DEG      = 3.0                # requested emitter jitter (deg)

# Expansion/cure staging timings and expansion volume
TAU_EXPAND_S             = 0.15               # time to start expanding (s)
TAU_EXPAND_JITTER        = 0.14
TAU_TO_P1_S              = 0.15
TAU_TO_P1_JITTER         = 0.00
EXPANSION_RATIO          = 10.0               # ~10x volume
BASE_EXPAND_R_MULT       = 5.0               # baseline radius multiplier (was 5)
EXPAND_RADIUS_MULTIPLIER = BASE_EXPAND_R_MULT * (EXPANSION_RATIO / 5.0) ** (1/3)
MAX_RESAMPLE_TRIES       = 10

# Expansion kinematics — how expansion burst velocities are sampled
EXPANSION_VEL            = 0.6                # base radial impulse (m/s)
EXPANSION_VEL_JITTER     = 0.6                # ± jitter on magnitude
VEL_RADIAL_WEIGHT        = 0.80               # blend with random dir
EXPANSION_EXTRA_NOISE    = 0.25               # extra isotropic speed (m/s) added

# Ground & bounds — used to clamp and cull sampled positions
GROUND_Z                 = 0.0
COLLISION_MARGIN         = 0.6 * P_SIZE

# Materials / densities — optional mass compensation reduces rho after expansion
RHO_P0                   = 1400.0
MASS_COMPENSATE          = True

# Drag (slightly higher to help stability)
DRAG_LINEAR              = 1.25

# Drone follow (visual only) — URDF just tracks the emitter visually
URDF_PATH = "/home/omenrtx5090/Documents/Aerial_AM_Simulation_Nevo/Drone_files/robot.urdf"
NOZZLE_TO_ORIGIN = np.array([-0.145989, 0.224300, 0.316818], dtype=np.float32)
NOZZLE_CLEARANCE = np.array([0.0, 0.0, 0.001], dtype=np.float32)

# Auto-sizing heuristics
PACKING_EFF              = 0.90   # fraction “filled” at spacing p_size
SAFETY_MARGIN            = 1.30   # headroom for each carrier

# Domain (bounded box) — particles are kept within this AABB
LOWER_BOUND              = (-0.6, -0.6, -0.05)
UPPER_BOUND              = ( 0.6,  0.6,  0.65)

# ======================= helpers: emission & sizing ==========================
def estimate_counts_sphere_per_step(dt, duration, sphere_diam, p_size, exp_ratio,
                                    tau_expand, tau_to_p1, packing_eff=0.9, margin=1.3):
    """Estimate per-step emission count and ring-buffer capacities for emit, expanded, and P1 phases."""
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
    """Compute minimal sphere radius to hold n_target particles at given packing efficiency."""
    V = (n_target * (p_size ** 3)) / max(packing_eff, 1e-6)
    r_cubed = (3.0 * V) / (4.0 * math.pi)
    r = max((r_cubed ** (1.0/3.0)), 1.5 * p_size)
    return r

def jitter_direction(base_dir: np.ndarray, jitter_deg: float) -> np.ndarray:
    """Return a unit direction that is base_dir perturbed by a small angular jitter in degrees."""
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
    """Sample K points uniformly inside a sphere of radius R."""
    u   = np.random.rand(K).astype(np.float32)
    r   = (R * (u ** (1.0/3.0))).astype(np.float32)[:, None]
    v   = np.random.normal(size=(K,3)).astype(np.float32)
    v  /= (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8)
    return r * v

def in_bounds_mask(points, lower, upper):
    """Boolean mask indicating which points lie within [lower, upper] bounds (inclusive)."""
    return np.all((points >= lower[None, :]) & (points <= upper[None, :]), axis=1)

# ======================= 1) Boot Genesis ====================================
# Create scene, solver, and viewer; configure MPM grid and particle size.
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
    show_viewer    = True,   # keep headless for perf; flip True if you want to see it
)

# ======================= 2) Ground ==========================================
# Add a rigid ground plane with strong coupling to MPM particles.
_ = scene.add_entity(
    morph=gs.morphs.Plane(),
    material=gs.materials.Rigid(needs_coup=True, coup_friction=1000.0, coup_softness=0.001, coup_restitution=0.0),
    surface=gs.surfaces.Default(color=(0.50, 0.50, 0.50)),
)

# ======= 2.5) Auto-size spherical carriers ==================================
# Pre-compute ring-buffer capacities and sphere radii for each phase to ensure enough room.
n_per_step, n_emit_total, cap_emit, cap_expanded, cap_p1 = estimate_counts_sphere_per_step(
    DT, DURATION, DROPLET_SIZE, P_SIZE, EXPANSION_RATIO, TAU_EXPAND_S, TAU_TO_P1_S,
    PACKING_EFF, SAFETY_MARGIN
)
# Radii are the minimal spheres that can hold the worst-case number of particles for each phase.
r_emit     = sphere_radius_for_particles(cap_emit,     P_SIZE, PACKING_EFF)
r_expanded = sphere_radius_for_particles(cap_expanded, P_SIZE, PACKING_EFF)
r_p1       = sphere_radius_for_particles(cap_p1,       P_SIZE, PACKING_EFF)
Z_POS = 0.30  # vertical placement for all carrier spheres and the visual drone

# ======================= 3) Carriers (Spheres) ==============================
# P0_emit: freshly emitted liquid, P0_expanded: expanded/foamed liquid, P1_viscous: cured viscoplastic
rho_expanded = (RHO_P0 / EXPANSION_RATIO) if MASS_COMPENSATE else RHO_P0
P0_emit = scene.add_entity(
    morph    = gs.morphs.Sphere(radius=r_emit, pos=(0.0, 0.0, Z_POS)),
    material = gs.materials.MPM.Liquid(viscous=True, rho=RHO_P0),
    surface  = gs.surfaces.Default(vis_mode="particle", color=(0.15, 0.65, 1.00)),
)
# Expanded phase: viscous=True
P0_expanded = scene.add_entity(
    morph    = gs.morphs.Sphere(radius=r_expanded, pos=(0.0, 0.0, Z_POS)),
    material = gs.materials.MPM.Liquid(viscous=True, rho=rho_expanded),
    surface  = gs.surfaces.Default(vis_mode="particle", color=(0.98, 0.78, 0.15)),
)

P1_viscous = scene.add_entity(
    morph    = gs.morphs.Sphere(radius=r_p1, pos=(0.0, 0.0, Z_POS)),
    material = gs.materials.MPM.ElastoPlastic(rho=rho_expanded, E=1.0e6, nu=0.05),
    surface  = gs.surfaces.Default(vis_mode="particle", color=(0.20, 0.95, 0.25)),
)

# ======================= 4) Emitter (into P0_emit) ==========================
# Emitter inserts particles into the P0_emit ring buffer each step.
emitter = scene.add_emitter(
    material      = gs.materials.MPM.Liquid(viscous=False, rho=RHO_P0),
    max_particles = P0_emit.n_particles,
    surface       = gs.surfaces.Default(vis_mode="particle", color=(0.15, 0.65, 1.00)),
)
emitter.set_entity(P0_emit)  # bind emitter output to the P0_emit carrier

def _write_block(entity, start_idx, pts_world, vels):
    """Write positions/velocities into an entity’s particle buffer at start_idx and mark them active."""
    n   = pts_world.shape[1]                     # number of particles in this block
    f   = scene.sim.cur_substep_local            # current substep index for synchronized writes
    sol = entity._solver                         # low-level solver handle
    # Write pos/vel and mark ACTIVE for the contiguous slice [start_idx, start_idx + n)
    sol._kernel_set_particles_pos(f, entity.particle_start + start_idx, n, pts_world)
    sol._kernel_set_particles_vel(f, entity.particle_start + start_idx, n, vels)
    sol._kernel_set_particles_active(f, entity.particle_start + start_idx, n, gs.ACTIVE)

def _push_block(entity, head, pts_world, vels, stamp_fn=None):
    """Ring-buffer insert for a block of particles, handling wraparound and optional stamping; returns new head."""
    cap = entity.n_particles                     # total capacity of the ring buffer
    n   = pts_world.shape[1]                     # number of new particles to insert
    rem = cap - head                             # how many slots remain until end
    if n <= rem:
        # Fits in one contiguous segment
        _write_block(entity, head, pts_world, vels)
        if stamp_fn: stamp_fn(head, n)           # metadata stamping for [head, head+n)
        head = (head + n) % cap                  # advance head
    else:
        # Wrap-around: split write into [head:head+rem) and [0:n-rem)
        _write_block(entity, head, pts_world[:, :rem, :], vels[:, :rem, :])
        if stamp_fn: stamp_fn(head, rem)
        _write_block(entity, 0,    pts_world[:, rem:, :],  vels[:, rem:, :])
        if stamp_fn: stamp_fn(0,   n - rem)
        head = (n - rem) % cap                   # new head is after the wrapped tail
    return head

# Ring-buffer heads and birth-time metadata for each phase
emit_head      = 0
expanded_head  = 0
p1_head        = 0

birth_emit     = None
birth_expanded = None
tau_emit_noise     = None
tau_to_p1_noise    = None

# instrumentation counters (ever-activated)
activated_emit_total     = 0
activated_expanded_total = 0
activated_p1_total       = 0

def emit_fixed(self, droplet_size=DROPLET_SIZE,
               pos=(0.0,0.0,1.0), base_direction=EMIT_DIR_BASE,
               speed=EMIT_SPEED, p_size=None, **kwargs):
    """Emit a sphere of particles each step with small direction jitter; stamp birth time and optional jitter."""
    global emit_head, step, activated_emit_total
    B = getattr(scene, "B", getattr(scene.sim, "_B", 1))  # batch dimension (usually 1)

    # Slightly jitter the base direction to avoid perfectly coherent streams
    dJ = jitter_direction(np.asarray(base_direction, dtype=gs.np_float), EMIT_DIR_JITTER_DEG)

    # Generate particles in local droplet sphere, then translate to world position
    p_size = P_SIZE if p_size is None else p_size
    pts_local = pu.sphere_to_particles(p_size=p_size, radius=droplet_size * 0.5, sampler=self._entity.sampler).astype(gs.np_float, copy=False)
    pts_world = pts_local + np.asarray(pos, dtype=gs.np_float)
    n         = pts_world.shape[0]

    # Tile across batches and build initial velocity along the (jittered) direction
    pts_world = np.tile(pts_world[None], (B, 1, 1))
    v_single  = (speed * dJ).astype(gs.np_float, copy=False)
    vels      = np.tile(v_single, (B, n, 1))

    # Stamp birth step and per-particle expansion jitter for the written range
    def stamp_emit(start, count):
        global activated_emit_total
        activated_emit_total += int(count)
        birth_emit[start:start+count] = step
        if TAU_EXPAND_JITTER > 0.0:
            tau_emit_noise[start:start+count] = np.random.uniform(0.0, TAU_EXPAND_JITTER, size=(count,))

    # Insert into P0_emit ring buffer
    emit_head = _push_block(P0_emit, emit_head, pts_world, vels, stamp_emit)

# Replace emitter.emit with our fixed-function sphere-per-step emitter
emitter.emit = types.MethodType(emit_fixed, emitter)

# Optional linear drag to stabilize larger bursts
if DRAG_LINEAR > 0.0:
    scene.add_force_field(gs.force_fields.Drag(linear=DRAG_LINEAR, quadratic=0.0))

# ======================= 4.5) Drone (visual only) ===========================
# The drone visually follows the nozzle; simulation-wise it is decoupled from particles.
drone = scene.add_entity(
    morph=gs.morphs.URDF(file=URDF_PATH, fixed=False, pos=(0.0, 0.0, Z_POS)),
    material=gs.materials.Rigid(rho=800.0),
)
def place_drone_at_emitter(emit_pos_xyz):
    """Place the visual drone so its nozzle aligns with the current emitter position."""
    # Compute drone origin given nozzle desired world position
    emit_pos = np.asarray(emit_pos_xyz, dtype=np.float32)
    nozzle_world = emit_pos + NOZZLE_CLEARANCE
    origin_world = nozzle_world + NOZZLE_TO_ORIGIN
    drone.set_pos(tuple(origin_world))
    drone.set_quat((1.0, 0.0, 0.0, 0.0))  # keep upright orientation

# ======================= 5) Build & deactivate ==============================
scene.build()
B = getattr(scene, "B", getattr(scene.sim, "_B", 1))

def deactivate_all(entity):
    """Deactivate all particles for the given entity across batches."""
    n = entity.n_particles
    act = np.full((B, n), gs.INACTIVE, dtype=np.int32)
    entity.set_active_arr(scene.sim.cur_substep_local, gs.tensor(act))

# Start with all carriers empty/inactive
for e in (P0_emit, P0_expanded, P1_viscous):
    deactivate_all(e)

# Birth-step trackers and optional jitter buffers (indexed by ring-buffer slot)
birth_emit     = np.full((P0_emit.n_particles,),     -1, dtype=np.int32)
birth_expanded = np.full((P0_expanded.n_particles,), -1, dtype=np.int32)
if TAU_EXPAND_JITTER > 0.0:  tau_emit_noise  = np.zeros((P0_emit.n_particles,), dtype=np.float32)
if TAU_TO_P1_JITTER > 0.0:   tau_to_p1_noise = np.zeros((P0_expanded.n_particles,), dtype=np.float32)

# ======================= 6) Expansion =======================================
def promote_expand(step, dt):
    """Promote aged emit particles to expanded phase by spawning ~expansion_ratio clones with radial/random impulse."""
    global expanded_head, activated_expanded_total

    # 1) Read current frame data for all P0_emit particles
    nE = P0_emit.n_particles
    posE = np.empty((B, nE, 3), dtype=np.float32)
    velE = np.empty((B, nE, 3), dtype=np.float32)
    CF   = np.empty((B, nE, 3, 3), dtype=np.float32)
    FF   = np.empty((B, nE, 3, 3), dtype=np.float32)
    Jp   = np.empty((B, nE),       dtype=np.float32)
    actE = np.empty((B, nE),       dtype=np.int32)
    P0_emit.get_frame(scene.sim.cur_substep_local, posE, velE, CF, FF, Jp, actE)

    # 2) Compute per-particle ages and determine who is eligible to expand
    ages = (step - birth_emit) * dt
    tau  = TAU_EXPAND_S if TAU_EXPAND_JITTER <= 0.0 else (TAU_EXPAND_S + (tau_emit_noise if tau_emit_noise is not None else 0.0))
    eligible = (birth_emit >= 0) & (ages >= tau)
    mask = (actE[0] == gs.ACTIVE) & eligible
    idxs = np.nonzero(mask)[0]
    if idxs.size == 0:
        return  # nothing to expand this step

    # 3) Get domain bounds for culling/clamping sampled points
    boundary = P0_emit._solver.boundary
    lower = np.array(boundary.lower, dtype=np.float32)
    upper = np.array(boundary.upper, dtype=np.float32)

    # 4) For each eligible source particle, sample K_target offspring positions and velocities
    pos_blocks, vel_blocks = [], []

    base_K  = int(math.floor(EXPANSION_RATIO))  # integer part of expansion ratio
    extra_p = EXPANSION_RATIO - base_K          # fractional part as Bernoulli
    R_samp  = EXPAND_RADIUS_MULTIPLIER * P_SIZE # sampling radius scaled to target volume growth

    for idx in idxs:
        # 4a) Decide offspring count K_target for this source particle
        K_target = base_K + (1 if np.random.rand() < extra_p else 0)
        if K_target <= 0:
            continue

        # 4b) Capture source state (pos/vel)
        base_pos = posE[0, idx, :].astype(np.float32)
        base_vel = velE[0, idx, :].astype(np.float32)

        # 4c) Repeatedly sample candidate offsets inside sphere until we have K_target points in-bounds
        kept_pos, kept_dir = [], []
        tries = 0
        while len(kept_pos) < K_target and tries < MAX_RESAMPLE_TRIES:
            need    = K_target - len(kept_pos)
            offsets = random_points_in_sphere(need, R_samp)                      # isotropic offsets
            dirs    = offsets / (np.linalg.norm(offsets, axis=1, keepdims=True) + 1e-8)  # radial directions
            cand    = base_pos[None, :] + offsets                                 # world positions
            cand[:, 2] = np.maximum(cand[:, 2], GROUND_Z + COLLISION_MARGIN)      # clamp above ground
            inb = in_bounds_mask(cand, lower, upper)                               # cull out-of-bounds
            if np.any(inb):
                kept_pos.append(cand[inb])
                kept_dir.append(dirs[inb])
            tries += 1

        if len(kept_pos) == 0:
            continue  # could not place any offspring; skip

        # 4d) Concatenate and trim to K_target
        cand_kept = np.concatenate(kept_pos, axis=0)[:K_target]
        dir_kept  = np.concatenate(kept_dir, axis=0)[:K_target]

        # 4e) Build burst velocities: radial impulse + random direction + extra isotropic noise
        Vexp   = EXPANSION_VEL + (np.random.rand(dir_kept.shape[0]) * 2 - 1) * EXPANSION_VEL_JITTER
        rand_u = np.random.normal(size=dir_kept.shape).astype(np.float32)
        rand_u /= (np.linalg.norm(rand_u, axis=1, keepdims=True) + 1e-8)
        impulse_dir = VEL_RADIAL_WEIGHT * dir_kept + (1.0 - VEL_RADIAL_WEIGHT) * rand_u
        impulse_dir /= (np.linalg.norm(impulse_dir, axis=1, keepdims=True) + 1e-8)

        # Extra isotropic noise magnitude helps break symmetry and stabilize
        extra_noise = np.random.normal(size=dir_kept.shape).astype(np.float32)
        extra_noise /= (np.linalg.norm(extra_noise, axis=1, keepdims=True) + 1e-8)
        v_kept = base_vel[None, :] + (Vexp[:, None] * impulse_dir) + (EXPANSION_EXTRA_NOISE * extra_noise)

        # 4f) Tile for batch dimension and collect for a single bulk write
        pos_blocks.append(np.tile(cand_kept[None, :, :], (B, 1, 1)))
        vel_blocks.append(np.tile(v_kept[None, :, :],    (B, 1, 1)))

    if len(pos_blocks) == 0:
        return  # nothing to insert

    # 5) Concatenate across all sources and write into P0_expanded ring buffer
    pos_out = np.concatenate(pos_blocks, axis=1)
    vel_out = np.concatenate(vel_blocks, axis=1)

    # Stamp birth step (for curing) and optional curing jitter for the written range
    def stamp_expanded(start, count):
        global activated_expanded_total
        activated_expanded_total += int(count)
        birth_expanded[start:start+count] = step
        if TAU_TO_P1_JITTER > 0.0:
            tau_to_p1_noise[start:start+count] = np.random.uniform(0.0, TAU_TO_P1_JITTER, size=(count,))

    expanded_head = _push_block(P0_expanded, expanded_head, pos_out, vel_out, stamp_expanded)

    # 6) Deactivate the source particles that were promoted and clear their metadata
    act0 = actE[0]
    act0[idxs] = gs.INACTIVE
    P0_emit.set_active_arr(scene.sim.cur_substep_local, gs.tensor(np.tile(act0[None], (B, 1)).astype(np.int32)))
    birth_emit[idxs] = -1
    if TAU_EXPAND_JITTER > 0.0:
        tau_emit_noise[idxs] = 0.0

# ======================= 7.5) Convert to P1 (curing) ========================
def promote_to_p1(step, dt):
    """Promote aged expanded particles to P1 (solid/viscoplastic) while preserving positions and velocities."""
    global p1_head, activated_p1_total

    # 1) Read current frame data for all P0_expanded particles
    nX = P0_expanded.n_particles
    posX = np.empty((B, nX, 3), dtype=np.float32)
    velX = np.empty((B, nX, 3), dtype=np.float32)
    CF   = np.empty((B, nX, 3, 3), dtype=np.float32)
    FF   = np.empty((B, nX, 3, 3), dtype=np.float32)
    Jp   = np.empty((B, nX),       dtype=np.float32)
    actX = np.empty((B, nX),       dtype=np.int32)
    P0_expanded.get_frame(scene.sim.cur_substep_local, posX, velX, CF, FF, Jp, actX)

    # 2) Compute ages and decide who is due to cure into P1
    ages = (step - birth_expanded) * dt
    tau  = TAU_TO_P1_S if TAU_TO_P1_JITTER <= 0.0 else (TAU_TO_P1_S + (tau_to_p1_noise if tau_to_p1_noise is not None else 0.0))
    due  = (birth_expanded >= 0) & (ages >= tau)
    idxs = np.nonzero((actX[0] == gs.ACTIVE) & due)[0]
    if idxs.size == 0:
        return  # nothing to cure this step

    # 3) Preserve positions and velocities for transfer
    pos_sel = posX[:, idxs, :]
    vel_sel = velX[:, idxs, :]

    # Stamp P1 activation count for the written range
    def stamp_p1(start, count):
        global activated_p1_total
        activated_p1_total += int(count)

    # 4) Insert into P1_viscous ring buffer and advance head
    p1_head = _push_block(P1_viscous, p1_head, pos_sel, vel_sel, stamp_fn=stamp_p1)

    # 5) Deactivate the expanded particles that have cured and clear their metadata
    act0 = actX[0]
    act0[idxs] = gs.INACTIVE
    P0_expanded.set_active_arr(scene.sim.cur_substep_local, gs.tensor(np.tile(act0[None], (B, 1)).astype(np.int32)))
    birth_expanded[idxs] = -1
    if TAU_TO_P1_JITTER > 0.0:
        tau_to_p1_noise[idxs] = 0.0

# ======================= 8) Demo loop =======================================
dt          = scene.sim.dt
steps_total = int(DURATION / dt)

# Drive the nozzle in a slow circle and oscillate z slightly over time
radius = 0.22
omega  = 2 * np.pi / DURATION

t_wall_start = time.perf_counter()
for step in range(steps_total):
    t      = step * dt * 4                             # scale time to speed up the path
    angle  = omega * t
    x_off  = radius * np.cos(angle)
    y_off  = radius * np.sin(angle)
    z_emit = 0.1 + 0.1 * (angle / (2*np.pi))           # simple gradual z change

    emit_pos = (x_off, y_off, z_emit)
    place_drone_at_emitter(emit_pos)                   # move the visual drone

    # Emit, then try to expand and cure what is due this step
    emitter.emit(
        droplet_size  = DROPLET_SIZE,   # sphere per step
        pos           = emit_pos,
        base_direction= EMIT_DIR_BASE,
        speed         = EMIT_SPEED,
        p_size        = P_SIZE,
    )
    promote_expand(step, dt)
    promote_to_p1(step, dt)

    scene.step()                                       # advance simulation substeps/frame
t_wall_end = time.perf_counter()

# ======================= 9) Summary =========================================
# Compute performance stats and current active counts per phase
wall_time_s = (t_wall_end - t_wall_start)
sim_time_s  = steps_total * dt
avg_fps     = steps_total / wall_time_s if wall_time_s > 0 else float("nan")
real_over_sim = wall_time_s / sim_time_s if sim_time_s > 0 else float("nan")

def active_count(entity):
    """Count currently active particles for the given entity."""
    # Read active flags and sum across the ring buffer
    n = entity.n_particles
    B_local = getattr(scene, "B", getattr(scene.sim, "_B", 1))
    d3  = np.empty((B_local, n, 3), dtype=np.float32)
    d33 = np.empty((B_local, n, 3, 3), dtype=np.float32)
    d1  = np.empty((B_local, n), dtype=np.float32)
    act = np.empty((B_local, n), dtype=np.int32)
    entity.get_frame(scene.sim.cur_substep_local, d3, d3, d33, d33, d1, act)
    return int(np.sum(act[0] == gs.ACTIVE))

# Tally active particles now and totals ever activated across all phases
active_emit     = active_count(P0_emit)
active_expanded = active_count(P0_expanded)
active_p1       = active_count(P1_viscous)
active_total    = active_emit + active_expanded + active_p1
activated_total_all = activated_emit_total + activated_expanded_total + activated_p1_total
sampled_capacity_all = P0_emit.n_particles + P0_expanded.n_particles + P1_viscous.n_particles

# domain volume (for logging)
lb = np.array(LOWER_BOUND, dtype=float)
ub = np.array(UPPER_BOUND, dtype=float)
bbox_edges = np.maximum(ub - lb, 0.0)
bbox_volume = float(bbox_edges[0] * bbox_edges[1] * bbox_edges[2])

def try_get_grid_density(sc):
    """Attempt to read grid density from the MPM solver; return None if unavailable."""
    try:
        gd = getattr(sc._sim.mpm_solver, "grid_density", None)
        return float(gd) if gd is not None else None
    except Exception:
        return None

# Print a concise summary and a LaTeX-ready row for tables
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

# LaTeX row (EOS/SPH fields N/A here)
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