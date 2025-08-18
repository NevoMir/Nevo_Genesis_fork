#!/usr/bin/env python3
# mpm_emitter2_ok.py
#
# Genesis SPH/PBD demo — omnidirectional particle burst every step.
#
# Follows the exact structure you requested:
#   1) gs.init()
#   2) scene = gs.Scene(...)
#   3) add ground plane
#   4) add “carrier” sphere
#   5) emitter = scene.add_emitter(...)
#   6) scene.build()
#   7) emit&simulate
#   8) scene.viewer.run()
#
# The script monkey-patches emitter.emit_omni() to fix two issues in Genesis
# 0.7.x: broadcasting ((N,)×(N,3)) and missing batch dimension (B,N,3).
#
# Tested with:  Python 3.10 • taichi 1.6.x • genesis 0.7.x  (GPU & CPU)

import types
import numpy as np
import genesis as gs
import genesis.utils.particle as pu      # needed inside the patch

# ────────────────────────────────────────────────────────────────────────────
# 1) Initialise Genesis
# ────────────────────────────────────────────────────────────────────────────
gs.init()

# ────────────────────────────────────────────────────────────────────────────
# 2) Create the scene
# ────────────────────────────────────────────────────────────────────────────
scene = gs.Scene(
    sim_options    = gs.options.SimOptions(
                         dt=1e-3,
                         substeps=5,
                         gravity=(0.0, 0.0, -9.81)     # ← add this
                     ),
    # pbd_options    = gs.options.PBDOptions(
    #                      gravity=(0.0, 0.0, -9.81)     # ← and this
    #                  ),
    viewer_options = gs.options.ViewerOptions(res=(800, 600)),
    show_viewer    = True,
)

# 3) Ground plane ------------------------------------------------------------
scene.add_entity(
    morph    = gs.morphs.Plane(),
    material = gs.materials.Rigid(),
    surface  = gs.surfaces.Default(color=(0.5, 0.5, 0.5)),
)

# 4) “Carrier” sphere for the emitter to sample ------------------------------
carrier = scene.add_entity(
    morph    = gs.morphs.Sphere(radius=0.1, pos=(0.0, 0.0, 1.0)),
    material = gs.materials.PBD.Particle(),
    surface  = gs.surfaces.Default(vis_mode="particle", color=(0.2, 0.2, 0.2)),
)

# 5) Add the emitter (built-in Genesis emitter object) -----------------------
emitter = scene.add_emitter(
    material      = gs.materials.PBD.Particle(),   # must be PBD.Particle or PBD.Liquid
    max_particles = 20000,
    surface       = gs.surfaces.Default(vis_mode="particle", color=(1.0, 0.2, 0.2)),
)
#emitter.set_entity(carrier)                        # give it something to sample

# ────────────────────────────────────────────────────────────────────────────
# Patch the broadcast + batch-dim bugs in Genesis-0.7.x emit_omni()
#    • correct vector/scalar broadcast
#    • add (batch, N, 3) dimension expected by the Taichi kernel
# ────────────────────────────────────────────────────────────────────────────
def emit_omni_fixed(self, source_radius=0.1, pos=(0.5, 0.5, 1.0),
                    speed=1.0, particle_size=None):
    assert self._entity is not None

    pos = np.asarray(pos, dtype=gs.np_float)
    if particle_size is None:
        particle_size = self._solver.particle_size

    # sample a thin spherical shell (local → world)
    positions_local = pu.shell_to_particles(
        p_size       = particle_size,
        outer_radius = source_radius,
        inner_radius = source_radius * 0.4,
        sampler      = self._entity.sampler,
    ).astype(gs.np_float, copy=False)            # ▼ ensure f32
    positions = pos + positions_local            # (N,3) f32

    # add batch dimension  (B, N, 3)
    positions = np.tile(positions[None], (self._sim._B, 1, 1))

    if not self._solver.boundary.is_inside(positions):
        gs.raise_exception("Emitted particles are outside the boundary.")

    dists = np.linalg.norm(positions_local, axis=1)          # (N,)
    # velocity ⇒ shape (N,3)  (DON’T tile across batch)
    vels  = (positions_local *
            ((speed / (dists + gs.EPS))[:, None])
            ).astype(gs.np_float, copy=False)  

    n   = positions.shape[1]
    idx = self._entity.particle_start + self._next_particle

    self._solver._kernel_set_particles_pos(
        self._sim.cur_substep_local, idx, n, positions
    )
    self._solver._kernel_set_particles_vel(
        self._sim.cur_substep_local, idx, n, vels
    )
    self._solver._kernel_set_particles_active(
        self._sim.cur_substep_local, idx, n, gs.ACTIVE
    )

    self._next_particle += n
    if self._next_particle + n > self._entity.n_particles:
        self._next_particle = 0

# Bind the new method to THIS emitter instance
emitter.emit_omni = types.MethodType(emit_omni_fixed, emitter)

# 6) Build the scene ---------------------------------------------------------
scene.build()

# 7) Emit & simulate ---------------------------------------------------------
# ─── 7) Emit & simulate for 5 seconds -------------------------------
DURATION      = 5.0            # seconds
SIM_DT        = 1e-3           # must match sim_options.dt
total_steps   = int(DURATION / SIM_DT)

for _ in range(total_steps):
    emitter.emit_omni(
        source_radius = 0.03,
        pos           = (0.0, 0.0, 1.0),
        speed         = 1.0,
        particle_size = 0.03,
    )
    scene.step()

# 8) Hand control to the viewer ---------------------------------------------
scene.viewer.run()