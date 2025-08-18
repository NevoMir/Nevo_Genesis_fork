import genesis as gs
import numpy as np

########################## init ##########################
gs.init()

########################## create a scene ##########################
scene = gs.Scene(
    sim_options=gs.options.SimOptions(
        dt=1e-3,        # simulation timestep
        substeps=10,    # substeps per dt
    ),
    mpm_options=gs.options.MPMOptions(
        lower_bound=(-1.0, -1.0, 0.0),
        upper_bound=( 1.0,  1.0, 2.0),
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

########################## add an MPM emitter ##########################
# default ElastoPlastic material
mpm_material = gs.materials.MPM.Base()

# emitter: allow up to 20k particles in flight
emitter = scene.add_emitter(
    material=mpm_material,
    max_particles=20_000,
    surface=gs.surfaces.Default(
        color=(1.0, 0.5, 0.2),
        vis_mode="particle",
    ),
)

########################## build the scene ##########################
scene.build()

########################## spawn in a circle for 5 seconds ##########################
radius = 0.5            # meters
height = 0.8            # z position
total_time = 5.0        # seconds
dt = 1e-3               # must match sim_options.dt
steps = int(total_time / dt)
particles_per_step = 200  # you can adjust this

for i in range(steps):
    t = i * dt
    # compute angle around the circle (one full revolution in total_time)
    theta = 2 * np.pi * (t / total_time)
    # nozzle position in world coords
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    z = height
    pos = (x, y, z)
    vel = (0.0, 0.0, 0.0)
    # emit in this step
    emitter.emit(pos, vel, count=particles_per_step)
    # advance one timestep
    scene.step()