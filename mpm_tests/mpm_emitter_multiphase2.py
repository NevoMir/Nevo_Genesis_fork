#!/usr/bin/env python3
# mpm_emitter_foam_demo.py
#
# Genesis MPM demo — foam curing: liquid -> expansion -> hardening.
# Requires your updated Emitter class that implements:
#   - Emitter.set_carriers([...])
#   - Emitter.emit_foam(...)

import numpy as np
import genesis as gs

# ─── 1) Boot Genesis ────────────────────────────────────────────────────────
gs.init()

# ─── 2) Scene & solver options ──────────────────────────────────────────────
scene = gs.Scene(
    sim_options = gs.options.SimOptions(
        dt       = 1e-3,
        substeps = 5,
        gravity  = (0.0, 0.0, -9.81),
    ),
    mpm_options = gs.options.MPMOptions(
        lower_bound   = (-4, -4,  0.0),
        upper_bound   = ( 4,  4,  1.5),
        particle_size = 0.03,        # ↓ cell size ⇒ more v-verts
    ),
    viewer_options = gs.options.ViewerOptions(res=(800, 600)),
    show_viewer    = True,
)

# ─── 3) Rigid ground ────────────────────────────────────────────────────────
scene.add_entity(
    morph    = gs.morphs.Plane(),
    material = gs.materials.Rigid(),
    surface  = gs.surfaces.Default(color=(0.5, 0.5, 0.5)),
)

# ─── 4) Carriers (age buckets) that the emitter samples ─────────────────────
# Start carriers as Liquid; later, individual buckets can harden.
K = 6
carriers = []
for k in range(K):
    carriers.append(
        scene.add_entity(
            morph    = gs.morphs.Sphere(radius=0.35, pos=(3.0, 0.0, 1.0)),
            material = gs.materials.MPM.Liquid(),         # start liquid
            surface  = gs.surfaces.Default(
                vis_mode="particle",
                color=(0.25, 0.12 + 0.08*k, 0.05),
            ),
        )
    )

# 5) Emitter with foam support ------------------------------------------------
emitter = scene.add_emitter(
    material      = gs.materials.MPM.Liquid(),   # dummy; entity material is used
    max_particles = 50_000,
    surface       = gs.surfaces.Default(
        vis_mode="particle",
        color=(0.8, 0.3, 0.1),
    ),
)

# Register multiple carriers so batches age independently
# (requires your updated Emitter class exposing set_carriers)
emitter.set_carriers(carriers)

# ─── 6) Build & run  (emitter follows a 0.6-m circle centred at origin) ────
scene.build()

duration      = 5.0                       # seconds
dt            = scene.sim.dt              # 1e-3
steps_total   = int(duration / dt)

radius        = 0.3                       # 0.6 m diameter
omega         = 2 * np.pi / duration      # one revolution over 'duration' (we'll 4× time below)

# Foam curing profile & expansion controls (tweak freely)
liquid_duration        = 0.25             # s
expansion_duration     = 0.40             # s
cure_total             = 0.90             # s
expansion_ratio        = 1.8              # ~target radius growth across expansion window
expansion_rings        = 6
expansion_shell_radius = 0.015            # m (base; will ramp toward target)
expansion_shell_speed  = 0.2              # m/s
expansion_p_size       = 0.03             # m

# Final hardened material
hard_mat = gs.materials.MPM.ElastoPlastic(
    E=6e5, nu=0.3, rho=1000.0,
    yield_lower=0.03, yield_higher=0.25,
    use_von_mises=True,
    von_mises_yield_stress=6e5 * 0.03
)

# Emission geometry/kinematics
droplet_shape = "sphere"
droplet_size  = 0.02
p_size_emit   = 0.04
speed_emit    = 0.4

for step in range(steps_total):
    t        = step * dt * 4.0               # 4× time scaling (faster orbit)
    x_offset = radius * np.cos(omega * t)
    y_offset = radius * np.sin(omega * t)

    emitter.emit_foam(
        droplet_shape = droplet_shape,
        droplet_size  = droplet_size,
        pos           = (x_offset, y_offset, 0.15),   # moving nozzle center
        direction     = (0.0, 0.0, -1.0),             # always downward
        theta         = 0.0,
        speed         = speed_emit,
        p_size        = p_size_emit,
        # curing schedule
        liquid_duration    = liquid_duration,
        expansion_duration = expansion_duration,
        cure_total         = cure_total,
        # expansion controls
        expansion_ratio        = expansion_ratio,
        expansion_rings        = expansion_rings,
        expansion_shell_radius = expansion_shell_radius,
        expansion_shell_speed  = expansion_shell_speed,
        expansion_particle_size= expansion_p_size,
        # hardening (per-bucket flip when bucket age >= cure_total)
        hardening_material = hard_mat,
        # bucket behavior
        use_buckets          = True,
        bucket_stride_steps  = 6
    )

    scene.step()

scene.viewer.run()