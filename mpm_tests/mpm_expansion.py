#!/usr/bin/env python3
# mpm_foamlike_swelling_public_api.py
# Foam-like volumetric growth via small-strain plastic dilation (no internal pressure).

import taichi as ti
import genesis as gs

gs.init()

scene = gs.Scene(
    sim_options=gs.options.SimOptions(
        dt=1e-3,          # stable dt
        substeps=20,      # smaller substep_dt for robustness
        gravity=(0.0, 0.0, -9.81),
    ),
    mpm_options=gs.options.MPMOptions(
        lower_bound=(-0.8, -0.8, 0.0),
        upper_bound=( 0.8,  0.8, 1.2),
        grid_density=96,
    ),
    show_viewer=True,
)

# Ground
scene.add_entity(morph=gs.morphs.Plane())

# Soft elastoplastic sphere that can dilate at tiny tension
center = (0.0, 0.0, 0.45)
radius = 0.12
blob = scene.add_entity(
    material=gs.materials.MPM.ElastoPlastic(
        E=1.1e5,
        nu=0.18,
        rho=950.0,
        sampler="pbs",
        use_von_mises=False,  # allow volumetric plasticity (non-isochoric)
        # KEY: make tensile yield tiny so even gentle growth causes plastic dilation
        yield_lower=0.02,
        yield_higher=0.006,   # << small (default in docs is ~0.0045)
    ),
    morph=gs.morphs.Sphere(pos=center, radius=radius),
    # surface=gs.surfaces.Default(color=(0.85, 0.5, 0.7, 1.0), vis_mode="visual"),  # skinned surface
)

# Isotropic "growth velocity" (NOT pressure). It gently makes v ≈ γ(t) * (x - center).
@ti.data_oriented
class IsoGrowthVel:
    def __init__(self, cx, cy, cz, gamma_ss, tau, kv, R):
        self.cx, self.cy, self.cz = float(cx), float(cy), float(cz)
        self.gamma_ss = float(gamma_ss)  # steady-state growth rate (1/s)
        self.tau = float(tau)            # ramp time (s)
        self.kv = float(kv)              # velocity servo gain (1/s)
        self.R  = float(R)               # radius of influence

    @ti.func
    def gamma(self, t):
        # Smooth ramp to steady growth: gamma(t) = gamma_ss * (1 - exp(-t/tau))
        return self.gamma_ss * (1.0 - ti.exp(-t / self.tau))

    @ti.func
    def get_acc(self, pos, vel, t, i):
        dx, dy, dz = pos[0]-self.cx, pos[1]-self.cy, pos[2]-self.cz
        d = ti.sqrt(dx*dx + dy*dy + dz*dz + 1e-12)
        # wide support so growth remains uniform as it expands
        x = ti.min(1.0, d / self.R)
        s = 3.0*x*x - 2.0*x*x*x  # smoothstep
        w = 1.0 - s              # center 1 → fades to 0 near R

        g = self.gamma(t)
        # target velocity for uniform swelling
        vtx, vty, vtz = g*dx, g*dy, g*dz

        # servo acceleration that nudges vel → target (units m/s^2)
        ax = self.kv * (vtx - vel[0]) * w
        ay = self.kv * (vty - vel[1]) * w
        az = self.kv * (vtz - vel[2]) * w
        return ti.Vector([ax, ay, az])

# Gentle, steady growth; tuned so it swells clearly but stays tame under gravity
scene.add_force_field(IsoGrowthVel(
    center[0], center[1], center[2],
    gamma_ss=0.12,   # ≈ 12% linear growth per second at steady state
    tau=1.0,         # ~1 s ramp-in
    kv=10.0,         # tight but not stiff
    R=radius*5.0,    # cover future size
))

# A touch of damping keeps it foam-like (remove if you prefer)
scene.add_force_field(gs.force_fields.Drag(linear=6.0))

scene.build()
for _ in range(4000):  # ~4 seconds
    scene.step()