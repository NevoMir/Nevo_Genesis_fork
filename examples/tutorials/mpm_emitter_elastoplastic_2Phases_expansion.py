#!/usr/bin/env python3
# mpm_liquid_aging_promotion_expanding_boundfix.py
#
# Cementitious/foam-like curing with apparent expansion:
#  - Emit "fresh" low-viscosity liquid.
#  - After τ seconds, promote each aged particle into K duplicates
#    placed on a small sphere around the original position (volume growth).
#  - Deactivate the original fresh slot. Optionally reduce cured rho ≈ rho/K.

import types
import numpy as np
import genesis as gs
import genesis.utils.particle as pu

# ======================= 0) Tunables ========================================
P_SIZE         = 0.03          # solver particle sizing (grid-linked)
POOL_RADIUS    = 0.35          # capacity radius for fresh pool
SPLIT_K        = 4             # duplicates per aged particle (expansion factor)
EXPAND_RADIUS  = 1.2 * P_SIZE  # spread distance for duplicates
TAU_S          = 0.6           # seconds before a particle "cures"
USE_BINGHAM    = False         # set True to use ElastoPlastic for cured phase

# (Optional) materials’ densities (set to None to use defaults)
FRESH_RHO = 1400.0
CURED_RHO = (FRESH_RHO / SPLIT_K) if FRESH_RHO is not None else None  # ↓ to offset splitting

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
        upper_bound   = ( 4,  4,  1.5),  # effective boundary will be slightly tighter
        particle_size = P_SIZE,
    ),
    viewer_options = gs.options.ViewerOptions(res=(900, 700)),
    show_viewer    = True,
)

# ======================= 2) Ground ==========================================
ground = scene.add_entity(
    morph=gs.morphs.Plane(),
    material=gs.materials.Rigid(
        needs_coup=True,
        coup_friction=0.9,
        coup_softness=0.001,
        coup_restitution=0.0,
    ),
    surface=gs.surfaces.Default(color=(0.5, 0.5, 0.5)),
)

# ======================= 3) Fresh vs Cured pools ============================
fresh_liquid = scene.add_entity(
    morph    = gs.morphs.Sphere(radius=POOL_RADIUS, pos=( 3.0, 0.0, 1.0)),
    material = gs.materials.MPM.Liquid(viscous=False, rho=(FRESH_RHO if FRESH_RHO is not None else 1000.0)),
    surface  = gs.surfaces.Default(vis_mode="particle", color=(0.15, 0.45, 0.85)),
)

# Make the cured pool big enough for splitting (capacity ≈ K× of fresh).
cured_capacity_radius = POOL_RADIUS * (SPLIT_K ** (1/3)) * 1.25  # ~0.69 for K=4

if USE_BINGHAM:
    cured_material = gs.materials.MPM.ElastoPlastic(
        E=1.5e6, nu=0.30,
        rho=(CURED_RHO if CURED_RHO is not None else 800.0),
        use_von_mises=True, von_mises_yield_stress=2.0e4
    )
else:
    cured_material = gs.materials.MPM.Liquid(
        viscous=True,
        rho=(CURED_RHO if CURED_RHO is not None else 800.0)
    )

# IMPORTANT FIX: lower z from 1.0 → 0.55 so the sphere stays inside the solver boundary
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

# Helpers to write contiguous blocks of (B,N,3)
def _write_block(entity, start_idx, pts_world, vels):
    n   = pts_world.shape[1]
    f   = scene.sim.cur_substep_local
    sol = entity._solver
    sol._kernel_set_particles_pos(f, entity.particle_start + start_idx, n, pts_world)
    sol._kernel_set_particles_vel(f, entity.particle_start + start_idx, n, vels)
    sol._kernel_set_particles_active(f, entity.particle_start + start_idx, n, gs.ACTIVE)

fresh_head = 0
cured_head = 0
fresh_birth = None  # init after build()

def emit_fixed(self, droplet_shape="sphere", droplet_size=0.01,
               pos=(0.0,0.0,1.0), direction=(0.0,0.0,-1.0),
               speed=0.4, p_size=None, **kwargs):
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
        fresh_head = (fresh_head + n) % cap
    else:
        _write_block(fresh_liquid, fresh_head, pts_world[:, :rem, :], vels[:, :rem, :])
        fresh_birth[fresh_head:fresh_head+rem] = step
        _write_block(fresh_liquid, 0,          pts_world[:, rem:, :], vels[:, rem:, :])
        fresh_birth[0:n-rem] = step
        fresh_head = (n - rem) % cap

# Bind patched emit
emitter.emit = types.MethodType(emit_fixed, emitter)

# Mild global drag to stabilize splashing
drag = scene.add_force_field(gs.force_fields.Drag(linear=1.5, quadratic=0.0))

# ======================= 5) Build & deactivate all ==========================
scene.build()
B = getattr(scene, "B", getattr(scene.sim, "_B", 1))

fresh_birth = np.full((fresh_liquid.n_particles,), -1, dtype=np.int32)

def deactivate_all(entity):
    n = entity.n_particles
    act = np.full((B, n), gs.INACTIVE, dtype=np.int32)
    entity.set_active_arr(scene.sim.cur_substep_local, gs.tensor(act))

deactivate_all(fresh_liquid)
deactivate_all(cured_liquid)

# ======================= 6) Promotion with expansion ========================
def _unit_dirs(K):
    if K == 4:
        base = np.array([[ 1,  1,  1],
                         [-1, -1,  1],
                         [-1,  1, -1],
                         [ 1, -1, -1]], dtype=np.float32)
        base /= np.linalg.norm(base, axis=1, keepdims=True) + 1e-8
        return base
    i = np.arange(K, dtype=np.float32) + 0.5
    phi = np.arccos(1 - 2*i/K)
    theta = np.pi * (1 + 5**0.5) * i
    x = np.cos(theta) * np.sin(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(phi)
    return np.stack([x,y,z], axis=1).astype(np.float32)

UNIT_DIRS = _unit_dirs(SPLIT_K)  # (K,3)

def promote_aged(step, dt):
    """Split each aged fresh particle into K cured duplicates placed radially (EXPAND_RADIUS)."""
    global cured_head

    nF = fresh_liquid.n_particles
    posF = np.empty((B, nF, 3), dtype=np.float32)
    velF = np.empty((B, nF, 3), dtype=np.float32)
    CF   = np.empty((B, nF, 3, 3), dtype=np.float32)
    FF   = np.empty((B, nF, 3, 3), dtype=np.float32)
    JpF  = np.empty((B, nF),       dtype=np.float32)
    actF = np.empty((B, nF),       dtype=np.int32)
    fresh_liquid.get_frame(scene.sim.cur_substep_local, posF, velF, CF, FF, JpF, actF)

    age_ok = (fresh_birth >= 0) & ((step - fresh_birth) * dt >= TAU_S)
    mask   = (actF[0] == gs.ACTIVE) & age_ok
    idxs   = np.nonzero(mask)[0]
    if idxs.size == 0:
        return

    pos_sel = posF[:, idxs, :]       # (B,k,3)
    vel_sel = velF[:, idxs, :]       # (B,k,3)
    k = idxs.size

    # Build K duplicates for each original: (B, k*K, 3)
    pos_rep = np.repeat(pos_sel, SPLIT_K, axis=1)  # (B,kK,3)
    vel_rep = np.repeat(vel_sel, SPLIT_K, axis=1)

    dirs = np.tile(UNIT_DIRS[None, :, :], (k, 1, 1))            # (k,K,3)
    dirs = dirs.reshape(1, k*SPLIT_K, 3).astype(np.float32)     # (1,kK,3)
    offsets = EXPAND_RADIUS * dirs                              # (1,kK,3)
    pos_out = pos_rep + offsets                                 # (B,kK,3)
    vel_out = vel_rep                                           # or add slight outward bias

    # Capacity/ wrap handling for cured pool
    kK  = k * SPLIT_K
    cap = cured_liquid.n_particles
    rem = cap - cured_head
    if kK <= rem:
        _write_block(cured_liquid, cured_head, pos_out, vel_out)
        cured_head = (cured_head + kK) % cap
    else:
        _write_block(cured_liquid, cured_head, pos_out[:, :rem, :], vel_out[:, :rem, :])
        _write_block(cured_liquid, 0,          pos_out[:, rem:, :], vel_out[:, rem:, :])
        cured_head = (kK - rem) % cap

    # Deactivate originals in fresh pool
    actF0 = actF[0]
    actF0[idxs] = gs.INACTIVE
    act_allB = np.tile(actF0[None], (B, 1))
    fresh_liquid.set_active_arr(scene.sim.cur_substep_local, gs.tensor(act_allB))

    # Free those slots
    fresh_birth[idxs] = -1

# ======================= 7) Run =============================================
duration    = 3.5
dt          = scene.sim.dt
steps_total = int(duration / dt)

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
        droplet_size  = 0.01,
        pos           = (x_off, y_off, z_emit),
        direction     = (0.0, 0.0, -1.0),
        speed         = 1.0,
        p_size        = P_SIZE,
    )

    promote_aged(step, dt)
    scene.step()

scene.viewer.run()