#!/usr/bin/env python3
# mpm_liquid_to_elastoplastic_with_tau_jitter.py
#
# Particles are emitted as viscous liquid. After an individual, randomized delay
# τ_i ~ TAU_BASE_S + U[0, TAU_JITTER_S], each particle is "promoted" into a
# cured pool that uses an ElastoPlastic material (Bingham-like paste).
# Implementation detail: we emulate a material swap by copying a particle's
# state into the cured entity, then deactivating the original slot.

import types
import numpy as np
import genesis as gs
import genesis.utils.particle as pu

# ─────────────────────────────────────────────────────────────────────────────
# 0) Tunables
# ─────────────────────────────────────────────────────────────────────────────
P_SIZE       = 0.03          # MPM particle size
POOL_RADIUS  = 0.35          # capacity sphere radius for each pool
TAU_BASE_S   = 0.60          # base time to cure
TAU_JITTER_S = 0.30          # added uniform jitter per particle in seconds (0 = none)

# Cured (elastoplastic) defaults — tweak for your paste
EP_E        = 1.5e6          # Young's modulus (Pa)
EP_NU       = 0.30           # Poisson's ratio
EP_RHO      = 1600.0         # density (kg/m^3)
EP_YIELD    = 2.0e4          # von Mises yield stress (Pa) ~ 20 kPa

# Optional mild damping to keep things tame
DRAG_LINEAR = 1.5

# ─────────────────────────────────────────────────────────────────────────────
# 1) Boot Genesis
# ─────────────────────────────────────────────────────────────────────────────
gs.init()

# ─────────────────────────────────────────────────────────────────────────────
# 2) Scene & solver options
# ─────────────────────────────────────────────────────────────────────────────
scene = gs.Scene(
    sim_options = gs.options.SimOptions(
        dt       = 1e-3,
        substeps = 5,
        gravity  = (0.0, 0.0, -9.81),
    ),
    mpm_options = gs.options.MPMOptions(
        lower_bound   = (-4, -4, -1.0),
        upper_bound   = ( 4,  4,  1.5),
        particle_size = P_SIZE,
    ),
    viewer_options = gs.options.ViewerOptions(res=(900, 700)),
    show_viewer    = True,
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
# 4) Pools: fresh (viscous liquid) → cured (elastoplastic)
# ─────────────────────────────────────────────────────────────────────────────
fresh_liquid = scene.add_entity(
    morph    = gs.morphs.Sphere(radius=POOL_RADIUS, pos=( 3.0, 0.0, 1.0)),
    material = gs.materials.MPM.Liquid(viscous=False),  # fresh phase = viscous liquid
    surface  = gs.surfaces.Default(vis_mode="particle", color=(0.15, 0.45, 0.85)),
)

cured_material = gs.materials.MPM.ElastoPlastic(
    E=EP_E, nu=EP_NU, rho=EP_RHO,
    use_von_mises=True, von_mises_yield_stress=EP_YIELD
)

cured_liquid = scene.add_entity(
    morph    = gs.morphs.Sphere(radius=POOL_RADIUS, pos=(-3.0, 0.0, 1.0)),
    material = cured_material,
    surface  = gs.surfaces.Default(vis_mode="particle", color=(0.85, 0.45, 0.15)),
)

# ─────────────────────────────────────────────────────────────────────────────
# 5) Emitter (writes into the FRESH pool)
# ─────────────────────────────────────────────────────────────────────────────
emitter = scene.add_emitter(
    material      = gs.materials.MPM.Liquid(),  # placeholder; actual target is entity below, changing parameters have no effect
    max_particles = fresh_liquid.n_particles,
    surface       = gs.surfaces.Default(vis_mode="particle", color=(0.15, 0.45, 0.85)),
)
emitter.set_entity(fresh_liquid)

def _write_block(entity, start_idx, pts_world, vels):
    """Low-level contiguous write of (B,N,3) pos/vel + activate."""
    n   = pts_world.shape[1]
    f   = scene.sim.cur_substep_local
    sol = entity._solver
    sol._kernel_set_particles_pos(f, entity.particle_start + start_idx, n, pts_world)
    sol._kernel_set_particles_vel(f, entity.particle_start + start_idx, n, vels)
    sol._kernel_set_particles_active(f, entity.particle_start + start_idx, n, gs.ACTIVE)

# Ring buffer heads
fresh_head = 0
cured_head = 0

# Per-slot bookkeeping (allocated after build when we know capacities & B)
fresh_birth_steps   = None         # int step when slot was (re)filled
fresh_deadline_step = None         # int step threshold when it should cure (with jitter)

def emit_fixed(self, droplet_shape="sphere", droplet_size=0.01,
               pos=(0.0,0.0,1.0), direction=(0.0,0.0,-1.0),
               speed=0.4, p_size=None, **kwargs):
    """Emit into the 'fresh' pool, stamp birth step, and set per-slot cure deadline with jitter."""
    global fresh_head

    B = getattr(scene, "B", getattr(scene.sim, "_B", 1))
    dt = scene.sim.dt

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

    # (B,N,3)
    pts_world = np.tile(pts_world[None], (B, 1, 1))
    v_single  = (speed * direction).astype(gs.np_float, copy=False)
    vels      = np.tile(v_single, (B, n, 1))

    # contiguous block write with wrap-around handling
    cap = fresh_liquid.n_particles
    rem = cap - fresh_head

    def _stamp_deadlines(start, count):
        # per-particle τ_i = TAU_BASE_S + U[0, TAU_JITTER_S]
        if TAU_JITTER_S > 0.0:
            jitter = np.random.uniform(0.0, TAU_JITTER_S, size=(count,))
        else:
            jitter = np.zeros((count,), dtype=np.float64)
        tau_i   = TAU_BASE_S + jitter
        # convert to step thresholds
        deadline = step + np.ceil(tau_i / dt).astype(np.int32)
        fresh_birth_steps[start:start+count]   = step
        fresh_deadline_step[start:start+count] = deadline

    if n <= rem:
        _write_block(fresh_liquid, fresh_head, pts_world, vels)
        _stamp_deadlines(fresh_head, n)
        fresh_head = (fresh_head + n) % cap
    else:
        # tail
        _write_block(fresh_liquid, fresh_head, pts_world[:, :rem, :], vels[:, :rem, :])
        _stamp_deadlines(fresh_head, rem)
        # wrap to head
        _write_block(fresh_liquid, 0,          pts_world[:, rem:, :], vels[:, rem:, :])
        _stamp_deadlines(0, n - rem)
        fresh_head = (n - rem) % cap

# bind the patch
emitter.emit = types.MethodType(emit_fixed, emitter)

# Optional global drag
if DRAG_LINEAR > 0.0:
    _ = scene.add_force_field(gs.force_fields.Drag(linear=DRAG_LINEAR, quadratic=0.0))

# ─────────────────────────────────────────────────────────────────────────────
# 6) Build (and then deactivate all)
# ─────────────────────────────────────────────────────────────────────────────
scene.build()

B = getattr(scene, "B", getattr(scene.sim, "_B", 1))
fresh_birth_steps   = np.full((fresh_liquid.n_particles,), -1, dtype=np.int32)
fresh_deadline_step = np.full((fresh_liquid.n_particles,), -1, dtype=np.int32)

def deactivate_all(entity):
    n = entity.n_particles
    act = np.full((B, n), gs.INACTIVE, dtype=np.int32)  # shape (B, N)
    entity.set_active_arr(scene.sim.cur_substep_local, gs.tensor(act))

deactivate_all(fresh_liquid)
deactivate_all(cured_liquid)

# ─────────────────────────────────────────────────────────────────────────────
# 7) Promotion logic (fresh → cured at per-slot deadline)
# ─────────────────────────────────────────────────────────────────────────────
def promote_aged(step, dt):
    """Move any fresh particle whose deadline has passed into the cured pool (ElastoPlastic)."""
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

    # Which slots are active & due?
    due = (fresh_deadline_step >= 0) & (step >= fresh_deadline_step)
    mask = (actF[0] == gs.ACTIVE) & due
    idxs = np.nonzero(mask)[0]
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
    fresh_birth_steps[idxs]   = -1
    fresh_deadline_step[idxs] = -1

# ─────────────────────────────────────────────────────────────────────────────
# 8) Run
# ─────────────────────────────────────────────────────────────────────────────
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