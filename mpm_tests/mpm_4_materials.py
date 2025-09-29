import genesis as gs

########################## init ##########################
gs.init()


########################## create a scene ##########################
scene = gs.Scene(
    sim_options=gs.options.SimOptions(
        dt=1e-3,
        substeps=10,
    ),
    mpm_options=gs.options.MPMOptions(
        lower_bound=(-1.0, -1.0, 0.0),
        upper_bound=(1.0, 1.0, 2.0),
    ),
    vis_options=gs.options.VisOptions(
        visualize_mpm_boundary=True,
    ),
    viewer_options=gs.options.ViewerOptions(
        camera_fov=30,
        res=(960, 640),
    ),
    show_viewer=True,
)

########################## define materials ##########################
# 1) Default ElastoPlastic
default_mat = gs.materials.MPM.ElastoPlastic()

# 2) Soft “squishy” foam
soft_foam = gs.materials.MPM.ElastoPlastic(
    E=5e4,             # Young’s modulus ≃ 5×10⁴ Pa
    nu=0.3,            
    rho=100.0,         
    sampler='pbs',
    yield_lower=0.02,  # yield at ≃2% strain
    yield_higher=0.10, # densify by ≃10%
    use_von_mises=True,
    von_mises_yield_stress=5e4 * 0.02  # ≃1 000 Pa
)

# 3) Medium “cushiony” foam
medium_foam = gs.materials.MPM.ElastoPlastic(
    E=2e5,
    nu=0.3,
    rho=200.0,
    sampler='pbs',
    yield_lower=0.03,
    yield_higher=0.20,
    use_von_mises=True,
    von_mises_yield_stress=2e5 * 0.03  # ≃6 000 Pa
)

# 4) Firm “dense” foam
firm_foam = gs.materials.MPM.ElastoPlastic(
    E=5e5,
    nu=0.3,
    rho=300.0,
    sampler='pbs',
    yield_lower=0.05,
    yield_higher=0.30,
    use_von_mises=True,
    von_mises_yield_stress=5e5 * 0.05  # ≃25 000 Pa
)

########################## entities ##########################
# ground plane
plane = scene.add_entity(
    morph=gs.morphs.Plane(),
)

# drop four spheres 1 m up, spaced along x
r = 0.1
zs = 1.0
xs = [-0.7, -0.3, 0.3, 0.7]
mats = [default_mat, soft_foam, medium_foam, firm_foam]
colors = [
    (0.4, 1.0, 0.4),  # default: green
    (0.4, 0.4, 1.0),  # soft: blue
    (1.0, 0.7, 0.0),  # medium: orange
    (1.0, 0.2, 0.2),  # firm: red
]

for x, mat, col in zip(xs, mats, colors):
    scene.add_entity(
        material=mat,
        morph=gs.morphs.Sphere(pos=(x, 0.0, zs), radius=r),
        surface=gs.surfaces.Default(color=col, vis_mode="particle"),
    )

########################## build & run ##########################
scene.build()

for _ in range(1000):
    scene.step()
    