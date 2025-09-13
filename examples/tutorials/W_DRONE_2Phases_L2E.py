#!/usr/bin/env python3
import types
import numpy as np
import genesis as gs
import genesis.utils.particle as pu

# ── Tunables ────────────────────────────────────────────────────────────────
P_SIZE       = 0.03
POOL_RADIUS  = 0.35
TAU_BASE_S   = 0.60
TAU_JITTER_S = 0.30

EP_E   = 1.5e6
EP_NU  = 0.30
EP_RHO = 1600.0
EP_YIELD = 2.0e4
DRAG_LINEAR = 1.5

# Nozzle→Origin offset from your measurement (meters)
NOZZLE_TO_ORIGIN = np.array([-0.145989, 0.224300, 0.316818], dtype=np.float32)
NOZZLE_CLEARANCE = np.array([0.0, 0.0, 0.001], dtype=np.float32)  # 1 mm up

# ── Boot Genesis & Scene ────────────────────────────────────────────────────
gs.init()
scene = gs.Scene(
    sim_options = gs.options.SimOptions(dt=1e-3, substeps=5, gravity=(0.0, 0.0, -9.81)),
    mpm_options = gs.options.MPMOptions(
        lower_bound=(-4,-4,-1.0), upper_bound=(4,4,1.5), particle_size=P_SIZE),
    viewer_options = gs.options.ViewerOptions(res=(900, 700)),
    show_viewer=True,
)

# ground
scene.add_entity(
    morph=gs.morphs.Plane(),
    material=gs.materials.Rigid(needs_coup=True, coup_friction=0.9,
                                coup_softness=0.001, coup_restitution=0.0),
    surface=gs.surfaces.Default(color=(0.5, 0.5, 0.5)),
)

# pools
fresh_liquid = scene.add_entity(
    morph=gs.morphs.Sphere(radius=POOL_RADIUS, pos=( 3.0, 0.0, 1.0)),
    material=gs.materials.MPM.Liquid(viscous=False),
    surface=gs.surfaces.Default(vis_mode="particle", color=(0.15, 0.45, 0.85)),
)
cured_material = gs.materials.MPM.ElastoPlastic(
    E=EP_E, nu=EP_NU, rho=EP_RHO, use_von_mises=True, von_mises_yield_stress=EP_YIELD
)
cured_liquid = scene.add_entity(
    morph=gs.morphs.Sphere(radius=POOL_RADIUS, pos=(-3.0, 0.0, 1.0)),
    material=cured_material,
    surface=gs.surfaces.Default(vis_mode="particle", color=(0.85, 0.45, 0.15)),
)

# emitter writing into fresh pool
emitter = scene.add_emitter(
    material=gs.materials.MPM.Liquid(),
    max_particles=fresh_liquid.n_particles,
    surface=gs.surfaces.Default(vis_mode="particle", color=(0.15, 0.45, 0.85)),
)
emitter.set_entity(fresh_liquid)

def _write_block(entity, start_idx, pts_world, vels):
    n   = pts_world.shape[1]
    f   = scene.sim.cur_substep_local
    sol = entity._solver
    sol._kernel_set_particles_pos(f, entity.particle_start + start_idx, n, pts_world)
    sol._kernel_set_particles_vel(f, entity.particle_start + start_idx, n, vels)
    sol._kernel_set_particles_active(f, entity.particle_start + start_idx, n, gs.ACTIVE)

fresh_head = 0
cured_head = 0
fresh_birth_steps   = None
fresh_deadline_step = None

def emit_fixed(self, droplet_shape="sphere", droplet_size=0.01,
               pos=(0.0,0.0,1.0), direction=(0.0,0.0,-1.0),
               speed=0.4, p_size=None, **kwargs):
    global fresh_head
    B  = getattr(scene, "B", getattr(scene.sim, "_B", 1))
    dt = scene.sim.dt

    direction = np.asarray(direction, dtype=gs.np_float)
    direction /= (np.linalg.norm(direction) + gs.EPS)

    p_size = P_SIZE if p_size is None else p_size
    pts_local = pu.sphere_to_particles(
        p_size=p_size, radius=droplet_size * 0.5, sampler=self._entity.sampler
    ).astype(gs.np_float, copy=False)                       # (N,3)

    pts_world = pts_local + np.asarray(pos, dtype=gs.np_float)  # (N,3)
    n = pts_world.shape[0]

    pts_world = np.tile(pts_world[None], (B, 1, 1))             # (B,N,3)
    v_single  = (speed * direction).astype(gs.np_float, copy=False)
    vels      = np.tile(v_single, (B, n, 1))

    cap = fresh_liquid.n_particles
    rem = cap - fresh_head

    def _stamp_deadlines(start, count):
        if TAU_JITTER_S > 0.0:
            jitter = np.random.uniform(0.0, TAU_JITTER_S, size=(count,))
        else:
            jitter = np.zeros((count,), dtype=np.float64)
        tau_i = TAU_BASE_S + jitter
        deadline = step + np.ceil(tau_i / dt).astype(np.int32)
        fresh_birth_steps[start:start+count]   = step
        fresh_deadline_step[start:start+count] = deadline

    if n <= rem:
        _write_block(fresh_liquid, fresh_head, pts_world, vels)
        _stamp_deadlines(fresh_head, n)
        fresh_head = (fresh_head + n) % cap
    else:
        _write_block(fresh_liquid, fresh_head, pts_world[:, :rem, :], vels[:, :rem, :])
        _stamp_deadlines(fresh_head, rem)
        _write_block(fresh_liquid, 0,          pts_world[:, rem:, :], vels[:, rem:, :])
        _stamp_deadlines(0, n - rem)
        fresh_head = (n - rem) % cap

# bind the patched emitter
emitter.emit = types.MethodType(emit_fixed, emitter)

# optional global drag
if DRAG_LINEAR > 0.0:
    scene.add_force_field(gs.force_fields.Drag(linear=DRAG_LINEAR, quadratic=0.0))

# ── Add the drone ───────────────────────────────────────────────────────────
drone = scene.add_entity(
    gs.morphs.URDF(
        file="/home/omenrtx5090/Documents/Aerial_AM_Simulation_Nevo/Drone_files/robot.urdf",
        fixed=False,  # dynamic on paper, but we'll overwrite pose each step
        pos=(0.0, 0.0, 0.4),
    ),
    material=gs.materials.Rigid(rho=800.0),
)

# ── Build and initialize pools inactive ─────────────────────────────────────
scene.build()

B = getattr(scene, "B", getattr(scene.sim, "_B", 1))
fresh_birth_steps   = np.full((fresh_liquid.n_particles,), -1, dtype=np.int32)
fresh_deadline_step = np.full((fresh_liquid.n_particles,), -1, dtype=np.int32)

def deactivate_all(entity):
    n = entity.n_particles
    act = np.full((B, n), gs.INACTIVE, dtype=np.int32)
    entity.set_active_arr(scene.sim.cur_substep_local, gs.tensor(act))

deactivate_all(fresh_liquid)
deactivate_all(cured_liquid)

# ── Promotion logic (fresh → cured) ─────────────────────────────────────────
def promote_aged(step, dt):
    global cured_head
    nF = fresh_liquid.n_particles

    posF = np.empty((B, nF, 3), dtype=np.float32)
    velF = np.empty((B, nF, 3), dtype=np.float32)
    CF   = np.empty((B, nF, 3, 3), dtype=np.float32)
    FF   = np.empty((B, nF, 3, 3), dtype=np.float32)
    JpF  = np.empty((B, nF),       dtype=np.float32)
    actF = np.empty((B, nF),       dtype=np.int32)
    fresh_liquid.get_frame(scene.sim.cur_substep_local, posF, velF, CF, FF, JpF, actF)

    due  = (fresh_deadline_step >= 0) & (step >= fresh_deadline_step)
    mask = (actF[0] == gs.ACTIVE) & due
    idxs = np.nonzero(mask)[0]
    if idxs.size == 0:
        return

    pos_sel = posF[:, idxs, :]
    vel_sel = velF[:, idxs, :]

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

    actF0 = actF[0]
    actF0[idxs] = gs.INACTIVE
    act_allB = np.tile(actF0[None], (B, 1))
    fresh_liquid.set_active_arr(scene.sim.cur_substep_local, gs.tensor(act_allB))

    fresh_birth_steps[idxs]   = -1
    fresh_deadline_step[idxs] = -1

# ── Helper: place drone so nozzle tracks the emitter ────────────────────────
def place_drone_at_emitter(emit_pos_xyz):
    emit_pos = np.asarray(emit_pos_xyz, dtype=np.float32)
    nozzle_world = emit_pos + NOZZLE_CLEARANCE         # 1 mm above emitter
    origin_world = nozzle_world + NOZZLE_TO_ORIGIN     # origin = nozzle + (nozzle→origin)
    drone.set_pos(tuple(origin_world))
    drone.set_quat((1.0, 0.0, 0.0, 0.0))               # keep “straight”

# ── Run ─────────────────────────────────────────────────────────────────────
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

    emit_pos = (x_off, y_off, z_emit)
    place_drone_at_emitter(emit_pos)

    emitter.emit(
        droplet_shape="square",
        droplet_size=0.01,
        pos=emit_pos,
        direction=(0.0, 0.0, -1.0),
        speed=1.0,
        p_size=P_SIZE,
    )

    promote_aged(step, dt)
    scene.step()

scene.viewer.run()