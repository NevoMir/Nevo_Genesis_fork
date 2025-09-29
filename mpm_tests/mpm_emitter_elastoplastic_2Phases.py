#!/usr/bin/env python3
# mpm_liquid_aging_promotion_fixed.py
#
# Age-based "curing" for cementitious extrusion in Genesis MPM:
# Emit as low-viscosity, then promote particles to a higher-viscosity (or yield-stress) material after τ seconds.
# Fixes: call set_active_arr AFTER scene.build(), and pass a (B, N) activeness array.

import types
import numpy as np
import genesis as gs
import genesis.utils.particle as pu

# =============== 1) Boot Genesis ============================================
gs.init()

# =============== 2) Scene & solver options =================================
scene = gs.Scene(
    sim_options = gs.options.SimOptions(
        dt       = 1e-3,
        substeps = 5,
        gravity  = (0.0, 0.0, -9.81),
    ),
    mpm_options = gs.options.MPMOptions(
        lower_bound   = (-4, -4, -1.0),
        upper_bound   = ( 4,  4,  1.5),
        particle_size = 0.03,
    ),
    viewer_options = gs.options.ViewerOptions(res=(900, 700)),
    show_viewer    = True,
)

# =============== 3) Ground ==================================================
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

# =============== 4) Two liquid "pools": fresh vs cured ======================
POOL_RADIUS = 0.35  # controls capacity; increase if you hit capacity
P_SIZE      = 0.03

fresh_liquid = scene.add_entity(
    morph    = gs.morphs.Sphere(radius=POOL_RADIUS, pos=( 3.0, 0.0, 1.0)),
    material = gs.materials.MPM.Liquid(viscous=False),  # "less viscous"
    surface  = gs.surfaces.Default(vis_mode="particle", color=(0.15, 0.45, 0.85)),
)

# Option A (binary more-viscous liquid):
cured_material = gs.materials.MPM.Liquid(viscous=True)

# Option B (Bingham-like paste after curing): try this instead of `Liquid(viscous=True)`
# cured_material = gs.materials.MPM.ElastoPlastic(
#     E=1.5e6, nu=0.30, rho=1600.0,
#     use_von_mises=True, von_mises_yield_stress=2.0e4
# )

cured_liquid = scene.add_entity(
    morph    = gs.morphs.Sphere(radius=POOL_RADIUS, pos=(-3.0, 0.0, 1.0)),
    material = cured_material,
    surface  = gs.surfaces.Default(vis_mode="particle", color=(0.85, 0.45, 0.15)),
)

# =============== 5) Emitter (writes into the FRESH pool) ====================
emitter = scene.add_emitter(
    material      = gs.materials.MPM.Liquid(viscous=False),  # placeholder; we patch emit() to target fresh entity slots
    max_particles = fresh_liquid.n_particles,
    surface       = gs.surfaces.Default(vis_mode="particle", color=(0.15, 0.45, 0.85)),
)
emitter.set_entity(fresh_liquid)

# Low-level helpers to set contiguous blocks
def _write_block(entity, start_idx, pts_world, vels):
    # pts_world, vels: numpy (B,N,3)
    n   = pts_world.shape[1]
    f   = scene.sim.cur_substep_local
    sol = entity._solver
    sol._kernel_set_particles_pos(f, entity.particle_start + start_idx, n, pts_world)
    sol._kernel_set_particles_vel(f, entity.particle_start + start_idx, n, vels)
    sol._kernel_set_particles_active(f, entity.particle_start + start_idx, n, gs.ACTIVE)

# Ring buffer pointers per pool
fresh_head = 0
cured_head = 0

# Birth step per fresh slot
fresh_birth = None  # we will size it after build() when we know capacities & B for activeness arrays

def emit_fixed(self, droplet_shape="sphere", droplet_size=0.01,
               pos=(0.0,0.0,1.0), direction=(0.0,0.0,-1.0),
               speed=0.4, p_size=None, **kwargs):
    global fresh_head

    # robust B getter across versions
    B = getattr(scene, "B", getattr(scene.sim, "_B", 1))

    direction = np.asarray(direction, dtype=gs.np_float)
    direction /= (np.linalg.norm(direction) + gs.EPS)

    p_size = P_SIZE if p_size is None else p_size

    # particle cloud for this droplet
    pts_local = pu.sphere_to_particles(
        p_size=p_size,
        radius=droplet_size * 0.5,
        sampler=self._entity.sampler,
    ).astype(gs.np_float, copy=False)                            # (N,3)

    pts_world = pts_local + np.asarray(pos, dtype=gs.np_float)   # (N,3)
    n         = pts_world.shape[0]

    # Make (B,N,3)
    pts_world = np.tile(pts_world[None], (B, 1, 1))
    v_single  = (speed * direction).astype(gs.np_float, copy=False)
    vels      = np.tile(v_single, (B, n, 1))

    # write as a contiguous block in the fresh pool (handle wrap-around)
    cap = fresh_liquid.n_particles
    rem = cap - fresh_head
    if n <= rem:
        _write_block(fresh_liquid, fresh_head, pts_world, vels)
        fresh_birth[fresh_head:fresh_head+n] = step  # record birth step
        fresh_head = (fresh_head + n) % cap
    else:
        _write_block(fresh_liquid, fresh_head, pts_world[:, :rem, :], vels[:, :rem, :])
        fresh_birth[fresh_head:fresh_head+rem] = step
        _write_block(fresh_liquid, 0,          pts_world[:, rem:, :], vels[:, rem:, :])
        fresh_birth[0:n-rem] = step
        fresh_head = (n - rem) % cap

# bind the patch
emitter.emit = types.MethodType(emit_fixed, emitter)

# Optional global drag to tame splashing
drag = scene.add_force_field(gs.force_fields.Drag(linear=1.5, quadratic=0.0))

# =============== 6) Build ================================================
scene.build()

# Now that the solver is initialized, we can safely:
#  (a) create birth array
#  (b) deactivate all particles using a (B, N) activeness array
B = getattr(scene, "B", getattr(scene.sim, "_B", 1))
fresh_birth = np.full((fresh_liquid.n_particles,), -1, dtype=np.int32)

def deactivate_all(entity):
    n = entity.n_particles
    act = np.full((B, n), gs.INACTIVE, dtype=np.int32)  # shape (B, N)
    entity.set_active_arr(scene.sim.cur_substep_local, gs.tensor(act))

deactivate_all(fresh_liquid)
deactivate_all(cured_liquid)

# =============== 7) Promotion logic (fresh -> cured after τ) ================
TAU_S = 0.6  # seconds after which a particle is considered "cured"

def promote_aged(step, dt):
    """Move any fresh particle older than TAU_S into cured pool."""
    global cured_head

    nF = fresh_liquid.n_particles
    # pull fresh state (positions/velocities/activeness)
    posF = np.empty((B, nF, 3), dtype=np.float32)
    velF = np.empty((B, nF, 3), dtype=np.float32)
    CF   = np.empty((B, nF, 3, 3), dtype=np.float32)
    FF   = np.empty((B, nF, 3, 3), dtype=np.float32)
    JpF  = np.empty((B, nF),       dtype=np.float32)
    actF = np.empty((B, nF),       dtype=np.int32)
    fresh_liquid.get_frame(scene.sim.cur_substep_local, posF, velF, CF, FF, JpF, actF)

    # Which slots are active & old enough?
    age_ok = (fresh_birth >= 0) & ((step - fresh_birth) * dt >= TAU_S)
    mask   = (actF[0] == gs.ACTIVE) & age_ok
    idxs   = np.nonzero(mask)[0]
    if idxs.size == 0:
        return

    # Gather current state
    pos_sel = posF[:, idxs, :]          # (B,k,3)
    vel_sel = velF[:, idxs, :]

    # Write them contiguously into cured pool (handle wrap)
    k   = idxs.size
    cap = cured_liquid.n_particles
    rem = cap - cured_head
    if k <= rem:
        _write_block(cured_liquid, cured_head, pos_sel, vel_sel)
        cured_head = (cured_head + k) % cap
    else:
        _write_block(cured_liquid, cured_head, pos_sel[:, :rem, :], vel_sel[:, :rem, :])
        _write_block(cured_liquid, 0,          pos_sel[:, rem:, :], vel_sel[:, rem:, :])
        cured_head = (k - rem) % cap

    # Deactivate those slots in fresh pool (pass a full (B,N) array back)
    actF0 = actF[0]
    actF0[idxs] = gs.INACTIVE
    act_allB = np.tile(actF0[None], (B, 1))
    fresh_liquid.set_active_arr(scene.sim.cur_substep_local, gs.tensor(act_allB))

    # Free the slots
    fresh_birth[idxs] = -1

# =============== 8) Run =====================================================
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

    # age-based promotion
    promote_aged(step, dt)

    scene.step()

scene.viewer.run()