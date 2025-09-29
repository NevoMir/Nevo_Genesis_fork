#!/usr/bin/env python3
# mpm_liquid_aging_promotion_expanding_with_impulse.py
#
# Emit fresh low-viscous liquid; after TAU_S seconds, each aged particle
# expands into ~R new particles sampled inside a small sphere and given an
# outward velocity impulse to physically increase occupied volume.

import types
import numpy as np
import genesis as gs
import genesis.utils.particle as pu

# ======================= 0) Tunables ========================================
P_SIZE                   = 0.02          # MPM particle size (grid-linked)
TAU_S                    = 0.3           # seconds before expansion
TAU_JITTER               = 0.1           # add U[0,TAU_JITTER] per particle
EXPANSION_RATIO          = 2.0           # ~particle/volume multiplier after curing
EXPAND_RADIUS_MULTIPLIER = 2.0           # spawn sphere radius = this * P_SIZE
MAX_RESAMPLE_TRIES       = 8             # tries per particle to get in-bounds samples

# Impulse settings (key to actually push neighbors and increase volume)
EXPANSION_VEL            = 0.5           # m/s radial impulse magnitude
EXPANSION_VEL_JITTER     = 0.4           # add U[-j, +j] m/s
VEL_RADIAL_WEIGHT        = 0.85          # 1.0 = purely radial, 0.0 = purely random

# Ground (avoid spawning below) — set to your plane height if different
GROUND_Z                 = 0.0
COLLISION_MARGIN         = 0.6 * P_SIZE  # keep spawns at least this above ground

# Material choices
USE_BINGHAM              = False         # True -> cured ElastoPlastic with yield
FRESH_RHO                = 1400.0        # kg/m^3 (wet paste)
MASS_COMPENSATE          = True          # scale cured rho by 1/EXPANSION_RATIO

# ======================= 1) Boot Genesis ====================================
gs.init()

scene = gs.Scene(
    sim_options = gs.options.SimOptions(
        dt       = 1e-3,
        substeps = 5,
        gravity  = (0.0, 0.0, -9.81),
    ),
    mpm_options = gs.options.MPMOptions(
        lower_bound   = (-4, -4, -1.0),
        upper_bound   = ( 4,  4,  1.6),
        particle_size = P_SIZE,
    ),
    viewer_options = gs.options.ViewerOptions(res=(1000, 700)),
    show_viewer    = True,
)

# ======================= 2) Ground ==========================================
ground = scene.add_entity(
    morph=gs.morphs.Plane(),  # z=0 plane
    material=gs.materials.Rigid(
        needs_coup=True,
        coup_friction=0.9,
        coup_softness=0.001,
        coup_restitution=0.0,
    ),
    surface=gs.surfaces.Default(color=(0.5, 0.5, 0.5)),
)

# ======================= 3) Fresh vs Cured pools ============================
POOL_RADIUS = 0.35
fresh_liquid = scene.add_entity(
    morph    = gs.morphs.Sphere(radius=POOL_RADIUS, pos=( 3.0, 0.0, 1.0)),
    material = gs.materials.MPM.Liquid(viscous=False, rho=FRESH_RHO),
    surface  = gs.surfaces.Default(vis_mode="particle", color=(0.15, 0.45, 0.85)),
)

# Capacity for cured pool: ≈ EXPANSION_RATIO × fresh (add safety factor)
cured_capacity_radius = POOL_RADIUS * (EXPANSION_RATIO ** (1/3)) * 1.35
cured_rho = (FRESH_RHO / EXPANSION_RATIO) if MASS_COMPENSATE else FRESH_RHO

if USE_BINGHAM:
    cured_material = gs.materials.MPM.ElastoPlastic(
        E=1.5e6, nu=0.30, rho=cured_rho,
        use_von_mises=True, von_mises_yield_stress=2.0e4
    )
else:
    cured_material = gs.materials.MPM.Liquid(viscous=True, rho=cured_rho)

# Keep capacity sphere inside solver bounds (lowered z)
cured_liquid = scene.add_entity(
    morph    = gs.morphs.Sphere(radius=cured_capacity_radius, pos=(-3.0, 0.0, 0.55)),
    material = cured_material,
    surface  = gs.surfaces.Default(vis_mode="particle", color=(0.85, 0.45, 0.15)),
)

# ======================= 4) Emitter (into fresh pool) =======================
emitter = scene.add_emitter(
    material      = gs.materials.MPM.Liquid(viscous=False),
    max_particles = fresh_liquid.n_particles,
    surface       = gs.surfaces.Default(vis_mode="particle", color=(0.15, 0.45, 0.85)),
)
emitter.set_entity(fresh_liquid)

def _write_block(entity, start_idx, pts_world, vels):
    """Write contiguous block (B,N,3)."""
    n   = pts_world.shape[1]
    f   = scene.sim.cur_substep_local
    sol = entity._solver
    sol._kernel_set_particles_pos(f, entity.particle_start + start_idx, n, pts_world)
    sol._kernel_set_particles_vel(f, entity.particle_start + start_idx, n, vels)
    sol._kernel_set_particles_active(f, entity.particle_start + start_idx, n, gs.ACTIVE)

fresh_head  = 0
cured_head  = 0
fresh_birth = None  # init after build
tau_noise   = None  # optional per-slot jitter

def emit_fixed(self, droplet_shape="sphere", droplet_size=0.01,
               pos=(0.0,0.0,1.0), direction=(0.0,0.0,-1.0),
               speed=0.4, p_size=None, **kwargs):
    """Emit into 'fresh' pool and stamp birth step."""
    global fresh_head

    B = getattr(scene, "B", getattr(scene.sim, "_B", 1))

    direction = np.asarray(direction, dtype=gs.np_float)
    direction /= (np.linalg.norm(direction) + gs.EPS)

    p_size = P_SIZE if p_size is None else p_size

    pts_local = pu.sphere_to_particles(
        p_size=p_size,
        radius=droplet_size * 0.5,
        sampler=self._entity.sampler,
    ).astype(gs.np_float, copy=False)                            # (N,3)

    pts_world = pts_local + np.asarray(pos, dtype=gs.np_float)   # (N,3)
    n         = pts_world.shape[0]

    pts_world = np.tile(pts_world[None], (B, 1, 1))              # (B,N,3)
    v_single  = (speed * direction).astype(gs.np_float, copy=False)
    vels      = np.tile(v_single, (B, n, 1))

    cap = fresh_liquid.n_particles
    rem = cap - fresh_head
    if n <= rem:
        _write_block(fresh_liquid, fresh_head, pts_world, vels)
        fresh_birth[fresh_head:fresh_head+n] = step
        if TAU_JITTER > 0.0:
            tau_noise[fresh_head:fresh_head+n] = np.random.uniform(0.0, TAU_JITTER, size=(n,))
        fresh_head = (fresh_head + n) % cap
    else:
        _write_block(fresh_liquid, fresh_head, pts_world[:, :rem, :], vels[:, :rem, :])
        fresh_birth[fresh_head:fresh_head+rem] = step
        if TAU_JITTER > 0.0:
            tau_noise[fresh_head:fresh_head+rem] = np.random.uniform(0.0, TAU_JITTER, size=(rem,))
        _write_block(fresh_liquid, 0,          pts_world[:, rem:, :], vels[:, rem:, :])
        fresh_birth[0:n-rem] = step
        if TAU_JITTER > 0.0:
            tau_noise[0:n-rem] = np.random.uniform(0.0, TAU_JITTER, size=(n-rem,))
        fresh_head = (n - rem) % cap

# Patch emitter
emitter.emit = types.MethodType(emit_fixed, emitter)

# Optional global drag (tame splashing but still allow expansion to develop)
drag = scene.add_force_field(gs.force_fields.Drag(linear=1.0, quadratic=0.0))

# ======================= 5) Build & deactivate all ==========================
scene.build()
B = getattr(scene, "B", getattr(scene.sim, "_B", 1))

fresh_birth = np.full((fresh_liquid.n_particles,), -1, dtype=np.int32)
if TAU_JITTER > 0.0:
    tau_noise = np.zeros((fresh_liquid.n_particles,), dtype=np.float32)

def deactivate_all(entity):
    n = entity.n_particles
    act = np.full((B, n), gs.INACTIVE, dtype=np.int32)
    entity.set_active_arr(scene.sim.cur_substep_local, gs.tensor(act))

deactivate_all(fresh_liquid)
deactivate_all(cured_liquid)

# ======================= 6) Helpers =========================================
def random_points_in_sphere(K, R):
    """(K,3) offsets uniformly sampled inside a sphere of radius R."""
    u   = np.random.rand(K).astype(np.float32)
    r   = (R * (u ** (1.0/3.0))).astype(np.float32)[:, None]  # uniform in volume
    vec = np.random.normal(size=(K,3)).astype(np.float32)
    vec /= (np.linalg.norm(vec, axis=1, keepdims=True) + 1e-8)
    return r * vec

def in_bounds_mask(points, lower, upper):
    """points: (K,3) -> mask within [lower, upper]."""
    return np.all((points >= lower[None, :]) & (points <= upper[None, :]), axis=1)

# ======================= 7) Promotion with expansion + outward impulse ======
def promote_aged(step, dt):
    """Replace each sufficiently old fresh particle with ~EXPANSION_RATIO duplicates that get an outward velocity impulse."""
    global cured_head

    nF = fresh_liquid.n_particles
    posF = np.empty((B, nF, 3), dtype=np.float32)
    velF = np.empty((B, nF, 3), dtype=np.float32)
    CF   = np.empty((B, nF, 3, 3), dtype=np.float32)
    FF   = np.empty((B, nF, 3, 3), dtype=np.float32)
    JpF  = np.empty((B, nF),       dtype=np.float32)
    actF = np.empty((B, nF),       dtype=np.int32)
    fresh_liquid.get_frame(scene.sim.cur_substep_local, posF, velF, CF, FF, JpF, actF)

    # Age condition (with optional jitter)
    ages = (step - fresh_birth) * dt
    tau  = TAU_S if TAU_JITTER <= 0.0 else (TAU_S + (tau_noise if tau_noise is not None else 0.0))
    age_ok = (fresh_birth >= 0) & (ages >= tau)

    mask   = (actF[0] == gs.ACTIVE) & age_ok
    idxs   = np.nonzero(mask)[0]
    if idxs.size == 0:
        return

    # Solver boundary extents
    boundary = fresh_liquid._solver.boundary
    lower = np.array(boundary.lower, dtype=np.float32)
    upper = np.array(boundary.upper, dtype=np.float32)

    pos_blocks = []
    vel_blocks = []

    base_K  = int(np.floor(EXPANSION_RATIO))
    extra_p = EXPANSION_RATIO - base_K
    R_samp  = EXPAND_RADIUS_MULTIPLIER * P_SIZE

    for idx in idxs:
        # Non-integer ratios via stochastic rounding
        K_target = base_K + (1 if np.random.rand() < extra_p else 0)
        if K_target <= 0:
            continue

        base_pos = posF[0, idx, :].astype(np.float32)   # (3,)
        base_vel = velF[0, idx, :].astype(np.float32)   # (3,)

        # Try to gather valid candidates (in-bounds & above floor)
        kept_pos = []
        kept_dir = []  # unit directions used for impulses (radial-ish)
        tries = 0
        while len(kept_pos) < K_target and tries < MAX_RESAMPLE_TRIES:
            need = K_target - len(kept_pos)
            offsets = random_points_in_sphere(need, R_samp)      # (need,3)
            # make sure we have directions for impulses
            dirs = offsets / (np.linalg.norm(offsets, axis=1, keepdims=True) + 1e-8)

            cand = base_pos[None, :] + offsets                   # (need,3)
            # keep above ground
            cand[:, 2] = np.maximum(cand[:, 2], GROUND_Z + COLLISION_MARGIN)

            inb = in_bounds_mask(cand, lower, upper)
            if np.any(inb):
                kept_pos.append(cand[inb])
                kept_dir.append(dirs[inb])
            tries += 1

        if len(kept_pos) == 0:
            continue

        cand_kept = np.concatenate(kept_pos, axis=0)[:K_target]  # (K,3)
        dir_kept  = np.concatenate(kept_dir, axis=0)[:K_target]  # (K,3)

        # Build outward + random impulse
        # v_new = base_vel + Vexp * (w * dir + (1-w) * rand_unit)
        Vexp   = EXPANSION_VEL + (np.random.rand() * 2 - 1) * EXPANSION_VEL_JITTER
        rand_u = np.random.normal(size=dir_kept.shape).astype(np.float32)
        rand_u /= (np.linalg.norm(rand_u, axis=1, keepdims=True) + 1e-8)
        impulse_dir = VEL_RADIAL_WEIGHT * dir_kept + (1.0 - VEL_RADIAL_WEIGHT) * rand_u
        impulse_dir /= (np.linalg.norm(impulse_dir, axis=1, keepdims=True) + 1e-8)
        v_kept = base_vel[None, :] + Vexp * impulse_dir          # (K,3)

        # Form (B,K,3)
        cand_B = np.tile(cand_kept[None, :, :], (B, 1, 1))
        vel_B  = np.tile(v_kept[None, :, :],    (B, 1, 1))

        pos_blocks.append(cand_B)
        vel_blocks.append(vel_B)

    if len(pos_blocks) == 0:
        return

    # Concatenate all new cured particles: (B, Nnew, 3)
    pos_out = np.concatenate(pos_blocks, axis=1)
    vel_out = np.concatenate(vel_blocks, axis=1)
    Nnew    = pos_out.shape[1]

    # Capacity / wrap handling for cured pool
    cap = cured_liquid.n_particles
    rem = cap - cured_head
    if Nnew <= rem:
        _write_block(cured_liquid, cured_head, pos_out, vel_out)
        cured_head = (cured_head + Nnew) % cap
    else:
        _write_block(cured_liquid, cured_head, pos_out[:, :rem, :], vel_out[:, :rem, :])
        _write_block(cured_liquid, 0,          pos_out[:, rem:, :],  vel_out[:, rem:, :])
        cured_head = (Nnew - rem) % cap

    # Deactivate originals in fresh pool & free their slots
    act0 = actF[0]
    act0[idxs] = gs.INACTIVE
    act_allB = np.tile(act0[None], (B, 1))
    fresh_liquid.set_active_arr(scene.sim.cur_substep_local, gs.tensor(act_allB))
    fresh_birth[idxs] = -1
    if TAU_JITTER > 0.0:
        tau_noise[idxs] = 0.0

# ======================= 8) Demo loop =======================================
duration    = 3.5
dt          = scene.sim.dt
steps_total = int(duration / dt)

# Simple circular path emitter
radius = 0.2
omega  = 2 * np.pi / duration

for step in range(steps_total):
    t      = step * dt * 4
    angle  = omega * t
    x_off  = radius * np.cos(angle)
    y_off  = radius * np.sin(angle)
    turns  = angle / (2 * np.pi)
    z_emit = 0.05 + 0.04 * turns

    emitter.emit(
        droplet_shape = "square",
        droplet_size  = 0.02,
        pos           = (x_off, y_off, z_emit),
        direction     = (0.0, 0.0, -1.0),
        speed         = 1.0,
        p_size        = P_SIZE,
    )

    promote_aged(step, dt)
    scene.step()

scene.viewer.run()