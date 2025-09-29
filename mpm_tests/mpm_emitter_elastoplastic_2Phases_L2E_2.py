#!/usr/bin/env python3
# four_phase_curing_pipeline_equal_last_transition_FIXED_INT32.py
#
# 4-stage curing with stochastic deadlines:
#   P0: Liquid(viscous=False)  -> P1: Liquid(viscous=True)     (short)
#   P1: Liquid(viscous=True)   -> P2: ElastoPlastic (soft)     (medium)
#   P2: ElastoPlastic (soft)   -> P3: ElastoPlastic (final)    (≈ same timing as P1→P2)
#
# Fix: use int32 for activeness arrays (set_active_arr), avoiding unsupported torch.uint8.

import types
import numpy as np
import genesis as gs
import genesis.utils.particle as pu

# ─────────────────────────────────────────────────────────────────────────────
# Tunables
# ─────────────────────────────────────────────────────────────────────────────
P_SIZE = 0.03

TAU01_BASE_S, TAU01_JITTER_S = 0.15, 0.05   # P0→P1  (short)
TAU12_BASE_S, TAU12_JITTER_S = 0.80, 0.20   # P1→P2
TAU23_BASE_S, TAU23_JITTER_S = 0.80, 0.20   # P2→P3  (≈ same as P1→P2)

RHO_LIQ = 1150.0

# P2: soft gel
EP2_E, EP2_NU, EP2_RHO = 5.0e5, 0.45, 1150.0
EP2_YL, EP2_YH = 0.003, 0.010
EP2_VON_MISES = False

# P3: final
EP3_E, EP3_NU, EP3_RHO = 2.0e9, 0.35, 1150.0
EP3_YL, EP3_YH = 0.010, 0.030
EP3_VON_MISES = False

DRAG_LINEAR = 1.0

# Carrier sizes (radii): P3 largest, P1 second largest, P0/P2 smaller
R_P0, R_P1, R_P2, R_P3 = 0.25, 0.45, 0.28, 0.65

# ─────────────────────────────────────────────────────────────────────────────
# Scene
# ─────────────────────────────────────────────────────────────────────────────
gs.init()
scene = gs.Scene(
    sim_options = gs.options.SimOptions(dt=1e-3, substeps=5, gravity=(0,0,-9.81)),
    mpm_options = gs.options.MPMOptions(lower_bound=(-4,-4,-1.0), upper_bound=(4,4,1.6), particle_size=P_SIZE),
    viewer_options = gs.options.ViewerOptions(res=(1120, 760)),
    show_viewer=True,
)

# Ground
_ = scene.add_entity(
    morph=gs.morphs.Plane(),
    material=gs.materials.Rigid(needs_coup=True, coup_friction=0.9, coup_softness=0.001, coup_restitution=0.0),
    surface=gs.surfaces.Default(color=(0.50, 0.50, 0.50)),
)

# Carriers with DISTINCT colors
P0 = scene.add_entity(
    morph=gs.morphs.Sphere(radius=R_P0, pos=( 3.0,  0.0, 0.92)),
    material=gs.materials.MPM.Liquid(viscous=False, rho=RHO_LIQ),
    surface=gs.surfaces.Default(vis_mode="particle", color=(0.05, 0.85, 1.00)),  # cyan
)
P1 = scene.add_entity(
    morph=gs.morphs.Sphere(radius=R_P1, pos=(-3.0,  0.0, 0.82)),
    material=gs.materials.MPM.Liquid(viscous=True,  rho=RHO_LIQ),
    surface=gs.surfaces.Default(vis_mode="particle", color=(0.20, 0.95, 0.25)),  # lime
)
P2 = scene.add_entity(
    morph=gs.morphs.Sphere(radius=R_P2, pos=(-2.5,  1.7, 0.62)),
    material=gs.materials.MPM.ElastoPlastic(
        E=EP2_E, nu=EP2_NU, rho=EP2_RHO,
        use_von_mises=EP2_VON_MISES, yield_lower=EP2_YL, yield_higher=EP2_YH),
    surface=gs.surfaces.Default(vis_mode="particle", color=(0.98, 0.82, 0.10)),  # gold
)
P3 = scene.add_entity(
    morph=gs.morphs.Sphere(radius=R_P3, pos=(-2.5, -1.7, 0.58)),
    material=gs.materials.MPM.ElastoPlastic(
        E=EP3_E, nu=EP3_NU, rho=EP3_RHO,
        use_von_mises=EP3_VON_MISES, yield_lower=EP3_YL, yield_higher=EP3_YH),
    surface=gs.surfaces.Default(vis_mode="particle", color=(0.88, 0.15, 0.88)),  # magenta
)

# Emitter → P0
emitter = scene.add_emitter(
    material=gs.materials.MPM.Liquid(viscous=False, rho=RHO_LIQ),
    max_particles=P0.n_particles,
    surface=gs.surfaces.Default(vis_mode="particle", color=(0.05, 0.85, 1.00)),
)
emitter.set_entity(P0)

def _write_block(entity, start_idx, pts_world, vels):
    n = pts_world.shape[1]
    f = scene.sim.cur_substep_local
    sol = entity._solver
    sol._kernel_set_particles_pos(f, entity.particle_start + start_idx, n, pts_world)
    sol._kernel_set_particles_vel(f, entity.particle_start + start_idx, n, vels)
    sol._kernel_set_particles_active(f, entity.particle_start + start_idx, n, gs.ACTIVE)

def _push_block(entity, head, pts_world, vels, stamp_fn):
    cap, n = entity.n_particles, pts_world.shape[1]
    rem = cap - head
    if n <= rem:
        _write_block(entity, head, pts_world, vels)
        if stamp_fn: stamp_fn(head, n)
        head = (head + n) % cap
    else:
        _write_block(entity, head, pts_world[:, :rem, :], vels[:, :rem, :])
        if stamp_fn: stamp_fn(head, rem)
        _write_block(entity, 0,    pts_world[:, rem:, :],  vels[:, rem:, :])
        if stamp_fn: stamp_fn(0, n - rem)
        head = (n - rem) % cap
    return head

head0 = head1 = head2 = head3 = 0
birth0 = dead0 = None
birth1 = dead1 = None
birth2 = dead2 = None

def emit_fixed(self, droplet_shape="sphere", droplet_size=0.010,
               pos=(0.0,0.0,1.0), direction=(0.0,0.0,-1.0),
               speed=0.9, p_size=None, **kwargs):
    global head0
    B  = getattr(scene, "B", getattr(scene.sim, "_B", 1))
    dt = scene.sim.dt

    direction = np.asarray(direction, dtype=gs.np_float); direction /= (np.linalg.norm(direction) + gs.EPS)
    p_size = P_SIZE if p_size is None else p_size

    pts_local = pu.sphere_to_particles(p_size=p_size, radius=droplet_size * 0.5, sampler=self._entity.sampler).astype(gs.np_float, copy=False)
    pts_world = pts_local + np.asarray(pos, dtype=gs.np_float)
    n = pts_world.shape[0]

    pts_world = np.tile(pts_world[None], (B, 1, 1))
    v_single  = (speed * direction).astype(gs.np_float, copy=False)
    vels      = np.tile(v_single, (B, n, 1))

    def stamp_p0(start, count):
        birth0[start:start+count] = step
        jitter = np.random.uniform(0.0, TAU01_JITTER_S, size=(count,)) if TAU01_JITTER_S > 0 else 0.0
        tau  = TAU01_BASE_S + jitter
        dead = step + np.ceil(tau / dt).astype(np.int32)
        dead0[start:start+count] = dead

    head0 = _push_block(P0, head0, pts_world, vels, stamp_p0)

emitter.emit = types.MethodType(emit_fixed, emitter)

if DRAG_LINEAR > 0.0:
    scene.add_force_field(gs.force_fields.Drag(linear=DRAG_LINEAR, quadratic=0.0))

# Build & deactivate (INT32)
scene.build()
B = getattr(scene, "B", getattr(scene.sim, "_B", 1))

def deactivate_all(entity):
    n = entity.n_particles
    act = np.full((B, n), gs.INACTIVE, dtype=np.int32)  # back to int32
    entity.set_active_arr(scene.sim.cur_substep_local, gs.tensor(act))

for e in (P0, P1, P2, P3):
    deactivate_all(e)

birth0 = np.full((P0.n_particles,), -1, dtype=np.int32); dead0 = np.full_like(birth0, -1)
birth1 = np.full((P1.n_particles,), -1, dtype=np.int32); dead1 = np.full_like(birth1, -1)
birth2 = np.full((P2.n_particles,), -1, dtype=np.int32); dead2 = np.full_like(birth2, -1)

def _promote(src, dst, head_dst, birth_src, dead_src, birth_dst, dead_dst,
             tau_next_base, tau_next_jitter):
    nS = src.n_particles
    posS = np.empty((B, nS, 3), dtype=np.float32); velS = np.empty((B, nS, 3), dtype=np.float32)
    CF   = np.empty((B, nS, 3, 3), dtype=np.float32); FF   = np.empty((B, nS, 3, 3), dtype=np.float32)
    Jp   = np.empty((B, nS), dtype=np.float32);      actS  = np.empty((B, nS), dtype=np.int32)
    src.get_frame(scene.sim.cur_substep_local, posS, velS, CF, FF, Jp, actS)

    due  = (dead_src >= 0) & (step >= dead_src)
    mask = (actS[0] == gs.ACTIVE) & due
    idxs = np.nonzero(mask)[0]
    if idxs.size == 0:
        return head_dst

    pos_sel = posS[:, idxs, :]; vel_sel = velS[:, idxs, :]
    k = idxs.size
    dt = scene.sim.dt

    def stamp_dst(start, count):
        if birth_dst is None or dead_dst is None or count == 0:
            return
        birth_dst[start:start+count] = step
        jitter = np.random.uniform(0.0, tau_next_jitter, size=(count,)) if tau_next_jitter > 0 else 0.0
        tau  = tau_next_base + jitter
        dead = step + np.ceil(tau / dt).astype(np.int32)
        dead_dst[start:start+count] = dead

    head_dst = _push_block(dst, head_dst, pos_sel, vel_sel, stamp_dst)

    act0 = actS[0]; act0[idxs] = gs.INACTIVE
    act_allB = np.tile(act0[None], (B, 1)).astype(np.int32)
    src.set_active_arr(scene.sim.cur_substep_local, gs.tensor(act_allB))

    birth_src[idxs] = -1; dead_src[idxs] = -1
    return head_dst

# Robust active counter (via get_frame)
def active_count(entity):
    n = entity.n_particles
    dummy3  = np.empty((B, n, 3), dtype=np.float32)
    dummy33 = np.empty((B, n, 3, 3), dtype=np.float32)
    dummy1  = np.empty((B, n), dtype=np.float32)
    act     = np.empty((B, n), dtype=np.int32)
    entity.get_frame(scene.sim.cur_substep_local, dummy3, dummy3, dummy33, dummy33, dummy1, act)
    return int(np.sum(act[0] == gs.ACTIVE))

# Run
duration    = 6.0
dt          = scene.sim.dt
steps_total = int(duration / dt)

next_report_step = 0
radius, omega = 0.22, 2*np.pi/duration

for step in range(steps_total):
    t = step * dt * 4
    angle = omega * t
    x_off, y_off = radius*np.cos(angle), radius*np.sin(angle)
    z_emit = 0.05 + 0.04 * (angle / (2*np.pi))

    emitter.emit(
        droplet_shape="square",
        droplet_size=0.010,
        pos=(x_off, y_off, z_emit),
        direction=(0.0, 0.0, -1.0),
        speed=1.0,
        p_size=P_SIZE,
    )

    head1 = _promote(P0, P1, head1, birth0, dead0, birth1, dead1, TAU12_BASE_S, TAU12_JITTER_S)
    head2 = _promote(P1, P2, head2, birth1, dead1, birth2, dead2, TAU23_BASE_S, TAU23_JITTER_S)
    head3 = _promote(P2, P3, head3, birth2, dead2, None, None, 0.0, 0.0)

    if step >= next_report_step:
        print(f"[t={step*dt:5.2f}s] P0={active_count(P0)}  P1={active_count(P1)}  P2={active_count(P2)}  P3={active_count(P3)}")
        next_report_step = step + int(0.5 / dt)

    scene.step()

scene.viewer.run()