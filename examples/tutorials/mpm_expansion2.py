#!/usr/bin/env python3
"""
mpm_expand_demo.py
Test that MPM particle *size* grows during the simulation.

Usage (defaults shown):
  python mpm_expand_demo.py \
      --growth-rate 0.5 \
      --mode preserve_density \
      --init-size 0.012 \
      --grid-density 96 \
      --steps 1500 \
      --show-viewer 1
"""

import argparse
import sys
import time

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--growth-rate", type=float, default=0.5,
                    help="Multiplicative diameter growth rate [1/s]. Each substep: size <- size*(1+rate*dt)")
    ap.add_argument("--mode", type=str, default="preserve_density",
                    choices=["preserve_density", "preserve_mass"],
                    help="Mass scaling behavior as size changes")
    ap.add_argument("--init-size", type=float, default=0.012,
                    help="Initial particle diameter [m]")
    ap.add_argument("--grid-density", type=float, default=96.0,
                    help="Grid cells per meter (sets dx=1/grid_density)")
    ap.add_argument("--steps", type=int, default=1500, help="Number of outer steps to run")
    ap.add_argument("--dt", type=float, default=4e-3, help="Outer step dt [s]")
    ap.add_argument("--substeps", type=int, default=10, help="Substeps per step")
    ap.add_argument("--show-viewer", type=int, default=1, help="Show live viewer (1) or headless (0)")
    return ap.parse_args()

def main():
    args = parse_args()

    try:
        import genesis as gs
    except Exception:
        print("Failed to import genesis. Make sure Genesis is installed and on PYTHONPATH.", file=sys.stderr)
        raise

    # --------- Init Genesis ---------
    gs.init()  # choose backend automatically

    # --------- Build Scene with MPM and our growth options ---------
    # Preflight: verify the patched fields exist on MPMOptions (Pydantic v1/v2 compatible)
    try:
        fields = getattr(gs.options.MPMOptions, "model_fields", None)  # Pydantic v2
        if fields is None:
            fields = getattr(gs.options.MPMOptions, "__fields__", {})  # Pydantic v1
    except Exception:
        fields = {}
    if "particle_size_growth_rate" not in fields:
        raise RuntimeError(
            "Your Genesis build doesn't expose `particle_size_growth_rate` on MPMOptions.\n"
            "Please ensure your repo has the modified MPMOptions and MPMSolver."
        )

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=args.dt,
            substeps=args.substeps,
            # gravity=(0.0, 0.0, 0.0),
        ),
        mpm_options=gs.options.MPMOptions(
            # Solver domain (keep tight for performance)
            lower_bound=(-0.6, -0.6, 0.0),
            upper_bound=( 0.6,  0.6, 0.9),

            # Discretization + dynamic-size controls
            grid_density=args.grid_density,
            particle_size=args.init_size,
            particle_size_growth_rate=args.growth_rate,        # NEW
            particle_size_growth_mode=args.mode,               # NEW
            particle_size_min=1e-5,
            particle_size_max=None,
        ),
        vis_options=gs.options.VisOptions(
            visualize_mpm_boundary=True,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_fov=35,
            res=(1280, 720),
            max_FPS=90,
        ),
        show_viewer=bool(args.show_viewer),
    )

    # --------- Add a ground plane ----------
    scene.add_entity(
        morph=gs.morphs.Plane(),
        surface=gs.surfaces.Default(
            color=(0.85, 0.85, 0.9, 1.0),
            vis_mode='visual',
        ),
    )

    # --------- Add an elastoplastic sphere blob (the one that will grow) ----------
    scene.add_entity(
        material=gs.materials.MPM.ElastoPlastic(
            E=1e6,      # Young's modulus
            nu=0.3,     # Poisson ratio
            rho=600.0,  # density; used with preserve_density mode
            sampler='pbs',
        ),
        morph=gs.morphs.Sphere(
            pos=(0.0, 0.0, 0.25),
            radius=0.11,
        ),
        surface=gs.surfaces.Default(
            color=(0.3, 0.95, 0.45, 1.0),
            vis_mode='particle',  # show internal particles to observe growth clearly
        ),
    )

    # Optional reference sphere that stays the same size
    scene.add_entity(
        material=gs.materials.MPM.Elastic(E=8e4, nu=0.35, rho=800.0, sampler='pbs'),
        morph=gs.morphs.Sphere(pos=(0.28, 0.0, 0.25), radius=0.08),
        surface=gs.surfaces.Default(color=(0.9, 0.4, 0.4, 1.0), vis_mode='particle'),
    )

    # --------- Build & optionally adjust camera ----------
    scene.build()
    try:
        cam = scene.get_camera()
        cam.set_pose(look_at=(0.0, 0.0, 0.25), dist=0.8, azimuth=30.0, elevation=25.0)
    except Exception:
        pass

    mpm_solver = scene.sim.mpm_solver

    print("[MPM Expand Demo] Starting run...")
    print(f"  dt={args.dt}, substeps={args.substeps}, outer steps={args.steps}")
    print(f"  grid_density={mpm_solver.grid_density} -> dx={mpm_solver.dx:.6f} m")
    print(f"  init particle_size={mpm_solver.particle_size:.6f} m (radius={mpm_solver.particle_radius:.6f} m)")
    print(f"  growth_rate={args.growth_rate}  mode={args.mode}")

    # --------- Run ---------
    horizon = int(args.steps)
    t0 = time.time()
    for i in range(horizon):
        scene.step()
        if (i % 50) == 0:
            cur_size = mpm_solver.particle_size
            print(f"step {i:5d} | particle_size = {cur_size:.6f} m  | p_vol_real ~ {mpm_solver.p_vol_real:.3e}")

    elapsed = time.time() - t0
    print(f"[MPM Expand Demo] Done {horizon} steps in {elapsed:.2f}s")
    print(f"Final particle_size = {mpm_solver.particle_size:.6f} m")

    if args.show_viewer:
        print("Close the window to exit.")
        try:
            import time as _time
            _time.sleep(0.2)
        except Exception:
            pass

if __name__ == "__main__":
    main()