#!/usr/bin/env python3
# three_phase_with_independent_expansion_FIXED_droplet.py
#
# 3-stage curing + independent expansion:
#   P0: Liquid(viscous=False) --(τ01±j)--> P1: Liquid(viscous=True) --(τ12±j)--> P2: ElastoPlastic (soft)
#   Expansion (τexp±j) is independent. When due, a particle splits into ~R duplicates of the SAME PHASE,
#   gets an outward velocity impulse, the original is deactivated, and the duplicates receive fresh
#   deadlines (expansion + next-phase transition if applicable).

import types
import numpy as np
import genesis as gs
import genesis.utils.particle as pu

# ======================= 0) Tunables ========================================
P_SIZE = 0.02
DROPLET_SIZE = 4.0 * P_SIZE   # ensure sphere radius >= ~2 * P_SIZE

# Phase transition timings (means + jitter)
TAU01_BASE_S, TAU01_JITTER_S = 0.20, 0.05   # P0 -> P1
TAU12_BASE_S, TAU12_JITTER_S = 0.80, 0.20   # P1 -> P2

# Expansion timings (independent of phase)
TAU_EXP_BASE_S   = 0.60
TAU_EXP_JITTER_S = 0.10

# Expansion geometry & impulse
EXPANSION_RATIO          = 2.0
EXPAND_RADIUS_MULTIPLIER = 2.0
MAX_RESAMPLE_TRIES       = 8
EXPANSION_VEL            = 0.5
EXPANSION_VEL_JITTER     = 0.4
VEL_RADIAL_WEIGHT        = 0.85

# Materials
RHO_LIQ = 1150.0
# P2 (soft/green)
EP2_E, EP2_NU, EP2_RHO = 5.0e5, 0.45, 1150.0
EP2_YL, EP2_YH         = 0.003, 0.010     # lower < upper
EP2_VON_MISES          = False

# Mild damping
DRAG_LINEAR = 1.0

# ======================= 1) Scene ===========================================
gs.init()
scene = gs.Scene(
    sim_options = gs.options.SimOptions(dt=1e-4, substeps=25, gravity=(0.0, 0.0, -9.81)),
    mpm_options = gs.options.MPMOptions(
        lower_bound=(-4, -4, -1.0), upper_bound=(4, 4, 1.6), particle_size=P_SIZE
    ),
    viewer_options = gs.options.ViewerOptions(res=(1100, 740)),
    show_viewer=True,
)

# Ground
_ = scene.add_entity(
    morph=gs.morphs.Plane(),
    material=gs.materials.Rigid(needs_coup=True, coup_friction=0.9, coup_softness=0.001, coup_restitution=0.0),
    surface=gs.surfaces.Default(color=(0.5, 0.5, 0.5)),
)
GROUND_Z = 0.0
COLLISION_MARGIN = 0.6 * P_SIZE

# ======================= 2) Carriers (enlarged for expansion) ===============
# P2 largest, P1 second, P0 smaller but larger than before due to expansion.
R_P0, R_P1, R_P2 = 0.45, 0.70, 0.85

P0 = scene.add_entity(
    morph=gs.morphs.Sphere(radius=R_P0, pos=( 3.0,  0.0, 0.90)),
    material=gs.materials.MPM.Liquid(viscous=False, rho=RHO_LIQ),
    surface=gs.surfaces.Default(vis_mode="particle", color=(0.05, 0.85, 1.00)),  # cyan
)
P1 = scene.add_entity(
    morph=gs.morphs.Sphere(radius=R_P1, pos=(-3.0,  0.0, 0.75)),
    material=gs.materials.MPM.Liquid(viscous=True,  rho=RHO_LIQ),
    surface=gs.surfaces.Default(vis_mode="particle", color=(0.20, 0.95, 0.25)),  # lime
)
P2 = scene.add_entity(
    morph=gs.morphs.Sphere(radius=R_P2, pos=(-2.5,  1.6, 0.60)),
    material=gs.materials.MPM.ElastoPlastic(
        E=EP2_E, nu=EP2_NU, rho=EP2_RHO,
        use_von_mises=EP2_VON_MISES,
        yield_lower=EP2_YL, yield_higher=EP2_YH,
    ),
    surface=gs.surfaces.Default(vis_mode="particle", color=(0.98, 0.82, 0.10)),  # gold
)

# ======================= 3) Emitter → P0 ====================================
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

head0 = head1 = head2 = 0

# Deadlines arrays (allocated after build)
birth01 = dead01 = None   # P0->P1
birth12 = dead12 = None   # P1->P2
exp_dead0 = exp_dead1 = exp_dead2 = None  # per-pool expansion deadlines

def emit_fixed(self, droplet_shape="sphere", droplet_size=None,
               pos=(0.0,0.0,1.0), direction=(0.0,0.0,-1.0),
               speed=1.1, p_size=None, **kwargs):
    """Emit into P0; set P0→P1 deadline and P0 expansion deadline."""
    global head0
    B  = getattr(scene, "B", getattr(scene.sim, "_B", 1))
    dt = scene.sim.dt

    direction = np.asarray(direction, dtype=gs.np_float); direction /= (np.linalg.norm(direction) + gs.EPS)
    p_size = P_SIZE if p_size is None else p_size
    droplet_size = DROPLET_SIZE if droplet_size is None else droplet_size

    # ---- robust particle cloud sampling ----
    rad = float(droplet_size) * 0.5
    pts_local = pu.sphere_to_particles(p_size=p_size, radius=rad, sampler=self._entity.sampler).astype(gs.np_float, copy=False)

    if pts_local.shape[0] == 0:
        # auto-upsize until we get at least 1 point (cap at ~3*p_size radius)
        for scale in (1.25, 1.5, 2.0, 3.0):
            pts_local = pu.sphere_to_particles(p_size=p_size, radius=max(rad, scale * p_size), sampler=self._entity.sampler).astype(gs.np_float, copy=False)
            if pts_local.shape[0] > 0:
                break
        if pts_local.shape[0] == 0:
            # last resort: single seed at the center
            pts_local = np.zeros((1,3), dtype=gs.np_float)

    pts_world = pts_local + np.asarray(pos, dtype=gs.np_float)
    n = pts_local.shape[0]

    pts_world = np.tile(pts_world[None], (B, 1, 1))
    v_single  = (speed * direction).astype(gs.np_float, copy=False)
    vels      = np.tile(v_single, (B, n, 1))

    def stamp(start, count):
        birth01[start:start+count] = step
        j01 = np.random.uniform(0.0, TAU01_JITTER_S, size=(count,)) if TAU01_JITTER_S > 0 else 0.0
        dead01[start:start+count] = step + np.ceil((TAU01_BASE_S + j01) / dt).astype(np.int32)
        jE  = np.random.uniform(0.0, TAU_EXP_JITTER_S, size=(count,)) if TAU_EXP_JITTER_S > 0 else 0.0
        exp_dead0[start:start+count] = step + np.ceil((TAU_EXP_BASE_S + jE) / dt).astype(np.int32)

    head0 = _push_block(P0, head0, pts_world, vels, stamp)

# Patch the emitter (still pass droplet_size when calling to be compatible with native signature)
emitter.emit = types.MethodType(emit_fixed, emitter)

# ======================= 4) Build & deactivate ==============================
if DRAG_LINEAR > 0.0:
    scene.add_force_field(gs.force_fields.Drag(linear=DRAG_LINEAR, quadratic=0.0))

scene.build()
B = getattr(scene, "B", getattr(scene.sim, "_B", 1))

def deactivate_all(entity):
    n = entity.n_particles
    act = np.full((B, n), gs.INACTIVE, dtype=np.int32)
    entity.set_active_arr(scene.sim.cur_substep_local, gs.tensor(act))

for e in (P0, P1, P2):
    deactivate_all(e)

birth01 = np.full((P0.n_particles,), -1, dtype=np.int32); dead01 = np.full_like(birth01, -1)
birth12 = np.full((P1.n_particles,), -1, dtype=np.int32); dead12 = np.full_like(birth12, -1)
exp_dead0 = np.full((P0.n_particles,), -1, dtype=np.int32)
exp_dead1 = np.full((P1.n_particles,), -1, dtype=np.int32)
exp_dead2 = np.full((P2.n_particles,), -1, dtype=np.int32)

# ======================= 5) Helpers =========================================
def random_points_in_sphere(K, R):
    u   = np.random.rand(K).astype(np.float32)
    r   = (R * (u ** (1.0/3.0))).astype(np.float32)[:, None]
    vec = np.random.normal(size=(K,3)).astype(np.float32)
    vec /= (np.linalg.norm(vec, axis=1, keepdims=True) + 1e-8)
    return r * vec

def in_bounds_mask(points, lower, upper):
    return np.all((points >= lower[None, :]) & (points <= upper[None, :]), axis=1)

# ======================= 6) Expansion (same-phase) ==========================
def make_stamp_new(exp_dead_arr, next_birth_arr, next_dead_arr, tau_next_base, tau_next_jitter):
    """Closure to stamp deadlines for newly written duplicates in a pool."""
    dt = scene.sim.dt
    def _stamp(start, count):
        # expansion deadlines for the same pool
        jE = np.random.uniform(0.0, TAU_EXP_JITTER_S, size=(count,)) if TAU_EXP_JITTER_S > 0 else 0.0
        exp_dead_arr[start:start+count] = step + np.ceil((TAU_EXP_BASE_S + jE) / dt).astype(np.int32)
        # if this pool also transitions forward, stamp its next-phase deadlines
        if next_birth_arr is not None and next_dead_arr is not None and tau_next_base > 0.0:
            next_birth_arr[start:start+count] = step
            jN = np.random.uniform(0.0, tau_next_jitter, size=(count,)) if tau_next_jitter > 0 else 0.0
            next_dead_arr[start:start+count]  = step + np.ceil((tau_next_base + jN) / dt).astype(np.int32)
    return _stamp

def expand_pool(pool, head, exp_dead_arr, next_birth_arr, next_dead_arr, tau_next_base, tau_next_jitter):
    """
    Expand due particles in 'pool'. New particles remain in the same pool.
    Returns (head, pos_out, vel_out, stamp_fn, idxs_to_deactivate).
    If no expansion happens, returns (head, None, None, None, None).
    """
    nS = pool.n_particles
    posS = np.empty((B, nS, 3), dtype=np.float32)
    velS = np.empty((B, nS, 3), dtype=np.float32)
    CF   = np.empty((B, nS, 3, 3), dtype=np.float32)
    FF   = np.empty((B, nS, 3, 3), dtype=np.float32)
    Jp   = np.empty((B, nS),       dtype=np.float32)
    actS = np.empty((B, nS),       dtype=np.int32)
    pool.get_frame(scene.sim.cur_substep_local, posS, velS, CF, FF, Jp, actS)

    due  = (exp_dead_arr >= 0) & (step >= exp_dead_arr)
    mask = (actS[0] == gs.ACTIVE) & due
    idxs = np.nonzero(mask)[0]
    if idxs.size == 0:
        return head, None, None, None, None

    # Boundaries
    boundary = pool._solver.boundary
    lower = np.array(boundary.lower, dtype=np.float32)
    upper = np.array(boundary.upper, dtype=np.float32)

    R_samp  = EXPAND_RADIUS_MULTIPLIER * P_SIZE
    base_K  = int(np.floor(EXPANSION_RATIO))
    extra_p = EXPANSION_RATIO - base_K

    pos_blocks, vel_blocks = [], []

    for idx in idxs:
        K_target = base_K + (1 if np.random.rand() < extra_p else 0)
        if K_target <= 0:
            continue

        base_pos = posS[0, idx, :].astype(np.float32)
        base_vel = velS[0, idx, :].astype(np.float32)

        kept_pos, kept_dir = [], []
        tries = 0
        while len(kept_pos) < K_target and tries < MAX_RESAMPLE_TRIES:
            need    = K_target - len(kept_pos)
            offsets = random_points_in_sphere(need, R_samp)
            dirs    = offsets / (np.linalg.norm(offsets, axis=1, keepdims=True) + 1e-8)
            cand    = base_pos[None, :] + offsets
            # keep above ground
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
        return head, None, None, None, None

    pos_out = np.concatenate(pos_blocks, axis=1)
    vel_out = np.concatenate(vel_blocks, axis=1)
    stamp_fn = make_stamp_new(exp_dead_arr, next_birth_arr, next_dead_arr, tau_next_base, tau_next_jitter)

    return head, pos_out, vel_out, stamp_fn, idxs

# ======================= 7) Promotions ======================================
def promote(src, dst, head_dst, birth_src, dead_src, birth_dst, dead_dst,
            tau_next_base, tau_next_jitter, exp_dead_src, exp_dead_dst):
    nS = src.n_particles
    posS = np.empty((B, nS, 3), dtype=np.float32)
    velS = np.empty((B, nS, 3), dtype=np.float32)
    CF   = np.empty((B, nS, 3, 3), dtype=np.float32)
    FF   = np.empty((B, nS, 3, 3), dtype=np.float32)
    Jp   = np.empty((B, nS),       dtype=np.float32)
    actS = np.empty((B, nS),       dtype=np.int32)
    src.get_frame(scene.sim.cur_substep_local, posS, velS, CF, FF, Jp, actS)

    due  = (dead_src >= 0) & (step >= dead_src)
    mask = (actS[0] == gs.ACTIVE) & due
    idxs = np.nonzero(mask)[0]
    if idxs.size == 0:
        return head_dst

    pos_sel = posS[:, idxs, :]
    vel_sel = velS[:, idxs, :]
    dt = scene.sim.dt

    def stamp_dst(start, count):
        # next-phase deadlines for promoted particles
        if birth_dst is not None and dead_dst is not None and tau_next_base > 0.0:
            birth_dst[start:start+count] = step
            jN = np.random.uniform(0.0, tau_next_jitter, size=(count,)) if tau_next_jitter > 0 else 0.0
            dead_dst[start:start+count]  = step + np.ceil((tau_next_base + jN) / dt).astype(np.int32)
        # expansion deadlines in destination pool
        jE = np.random.uniform(0.0, TAU_EXP_JITTER_S, size=(count,)) if TAU_EXP_JITTER_S > 0 else 0.0
        exp_dead_dst[start:start+count] = step + np.ceil((TAU_EXP_BASE_S + jE) / dt).astype(np.int32)

    head_dst = _push_block(dst, head_dst, pos_sel, vel_sel, stamp_dst)

    # deactivate promoted in src; clear their deadlines
    # (activeness via get_frame to read, then write back)
    act_read = np.empty((B, nS), dtype=np.int32)
    src.get_frame(scene.sim.cur_substep_local,
                  np.empty((B,nS,3),dtype=np.float32),
                  np.empty((B,nS,3),dtype=np.float32),
                  np.empty((B,nS,3,3),dtype=np.float32),
                  np.empty((B,nS,3,3),dtype=np.float32),
                  np.empty((B,nS),dtype=np.float32),
                  act_read)
    a0 = act_read[0]; a0[idxs] = gs.INACTIVE
    src.set_active_arr(scene.sim.cur_substep_local, gs.tensor(np.tile(a0[None], (B,1)).astype(np.int32)))

    birth_src[idxs] = -1; dead_src[idxs] = -1; exp_dead_src[idxs] = -1
    return head_dst

# ======================= 8) Run =============================================
duration    = 6.0
dt          = scene.sim.dt
steps_total = int(duration / dt)

radius, omega = 0.22, 2 * np.pi / duration

for step in range(steps_total):
    # emitter on a little circle
    t = step * dt * 4
    angle = omega * t
    x_off, y_off = radius * np.cos(angle), radius * np.sin(angle)
    z_emit = 0.05 + 0.04 * (angle / (2*np.pi))
    emitter.emit(
        droplet_shape="square",
        droplet_size=DROPLET_SIZE,  # <<< pass it explicitly; compatible with native signature or our patch
        pos=(x_off, y_off, z_emit),
        direction=(0.0, 0.0, -1.0),
        speed=1.1,
        p_size=P_SIZE,
    )

    # --------- EXPANSION FIRST (phase-consistent splits) ---------
    # P0 expansions (duplicates stay in P0, also stamp P0→P1 for duplicates)
    head0, pos0, vel0, stamp0, idxs0 = expand_pool(P0, head0, exp_dead0, birth01, dead01,
                                                   TAU01_BASE_S, TAU01_JITTER_S)
    if pos0 is not None:
        head0 = _push_block(P0, head0, pos0, vel0, stamp0)
        if idxs0 is not None and len(idxs0) > 0:
            nS = P0.n_particles
            act = np.empty((B, nS), dtype=np.int32)
            P0.get_frame(scene.sim.cur_substep_local,
                         np.empty((B,nS,3),dtype=np.float32),
                         np.empty((B,nS,3),dtype=np.float32),
                         np.empty((B,nS,3,3),dtype=np.float32),
                         np.empty((B,nS,3,3),dtype=np.float32),
                         np.empty((B,nS),dtype=np.float32),
                         act)
            a0 = act[0]; a0[idxs0] = gs.INACTIVE
            P0.set_active_arr(scene.sim.cur_substep_local, gs.tensor(np.tile(a0[None], (B,1)).astype(np.int32)))
            birth01[idxs0] = -1; dead01[idxs0] = -1; exp_dead0[idxs0] = -1

    # P1 expansions (duplicates stay in P1, also stamp P1→P2 for duplicates)
    head1, pos1, vel1, stamp1, idxs1 = expand_pool(P1, head1, exp_dead1, birth12, dead12,
                                                   TAU12_BASE_S, TAU12_JITTER_S)
    if pos1 is not None:
        head1 = _push_block(P1, head1, pos1, vel1, stamp1)
        if idxs1 is not None and len(idxs1) > 0:
            nS = P1.n_particles
            act = np.empty((B, nS), dtype=np.int32)
            P1.get_frame(scene.sim.cur_substep_local,
                         np.empty((B,nS,3),dtype=np.float32),
                         np.empty((B,nS,3),dtype=np.float32),
                         np.empty((B,nS,3,3),dtype=np.float32),
                         np.empty((B,nS,3,3),dtype=np.float32),
                         np.empty((B,nS),dtype=np.float32),
                         act)
            a1 = act[0]; a1[idxs1] = gs.INACTIVE
            P1.set_active_arr(scene.sim.cur_substep_local, gs.tensor(np.tile(a1[None], (B,1)).astype(np.int32)))
            birth12[idxs1] = -1; dead12[idxs1] = -1; exp_dead1[idxs1] = -1

    # P2 expansions (terminal phase in this 3-stage setup)
    head2, pos2, vel2, stamp2, idxs2 = expand_pool(P2, head2, exp_dead2, None, None, 0.0, 0.0)
    if pos2 is not None:
        head2 = _push_block(P2, head2, pos2, vel2, stamp2)
        if idxs2 is not None and len(idxs2) > 0:
            nS = P2.n_particles
            act = np.empty((B, nS), dtype=np.int32)
            P2.get_frame(scene.sim.cur_substep_local,
                         np.empty((B,nS,3),dtype=np.float32),
                         np.empty((B,nS,3),dtype=np.float32),
                         np.empty((B,nS,3,3),dtype=np.float32),
                         np.empty((B,nS,3,3),dtype=np.float32),
                         np.empty((B,nS),dtype=np.float32),
                         act)
            a2 = act[0]; a2[idxs2] = gs.INACTIVE
            P2.set_active_arr(scene.sim.cur_substep_local, gs.tensor(np.tile(a2[None], (B,1)).astype(np.int32)))
            exp_dead2[idxs2] = -1

    # --------- PROMOTIONS (after expansion) ---------
    head1 = promote(P0, P1, head1, birth01, dead01, birth12, dead12,
                    TAU12_BASE_S, TAU12_JITTER_S, exp_dead0, exp_dead1)
    head2 = promote(P1, P2, head2, birth12, dead12, None, None,
                    0.0, 0.0, exp_dead1, exp_dead2)

    scene.step()

scene.viewer.run()