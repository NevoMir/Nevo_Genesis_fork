import numpy as np
import taichi as ti

import genesis as gs
import genesis.utils.geom as gu
import genesis.utils.particle as pu
from genesis.repr_base import RBC


@ti.data_oriented
class Emitter(RBC):
    """
    A particle emitter for fluid or material simulation.

    The Emitter manages the generation of particles into the simulation domain, allowing directional or omnidirectional
    emissions with various droplet shapes. It supports resetting, shape-based emission, and spherical omni-emission.

    Parameters
    ----------
    max_particles : int
        The maximum number of particles that this emitter can handle.
    """

    def __init__(self, max_particles):
        self._uid = gs.UID()
        self._entity = None

        self._max_particles = max_particles

        self._acc_droplet_len = 0.0  # accumulated droplet length to be emitted

        gs.logger.info(
            f"Creating ~<{self._repr_type()}>~. id: ~~~<{self._uid}>~~~, max_particles: ~<{max_particles}>~."
        )

    def set_entity(self, entity):
        """
        Assign an entity to the emitter and initialize relevant simulation and solver references.

        Parameters
        ----------
        entity : Entity
            The entity to associate with the emitter. This entity should contain the solver, simulation context, and particle sampler.
        """
        self._entity = entity
        self._sim = entity.sim
        self._solver = entity.solver
        self._next_particle = 0
        gs.logger.info(f"~<{self._repr_briefer()}>~ created using ~<{entity._repr_briefer()}.")

    def reset(self):
        """
        Reset the emitter's internal particle index to start emitting from the beginning.
        """
        self._next_particle = 0

    def emit(
        self,
        droplet_shape,
        droplet_size,
        droplet_length=None,
        pos=(0.5, 0.5, 1.0),
        direction=(0, 0, -1),
        theta=0.0,
        speed=1.0,
        p_size=None,
    ):
        """
        Emit particles in a specified shape and direction from a nozzle.

        Parameters
        ----------
        droplet_shape : str
            The shape of the emitted droplet. Options: "circle", "sphere", "square", "rectangle".
        droplet_size : float or tuple
            Size of the droplet. A single float for symmetric shapes, or a tuple of (width, height) for rectangles.
        droplet_length : float, optional
            Length of the droplet in the emitting direction. If None, calculated from speed and simulation timing.
        pos : tuple of float
            World position of the nozzle from which the droplet is emitted.
        direction : tuple of float
            Direction vector of the emitted droplet.
        theta : float
            Rotation angle (in radians) around the droplet axis.
        speed : float
            Emission speed of the particles.
        p_size : float, optional
            Particle size used for filling the droplet. Defaults to the solver's particle size.

        Raises
        ------
        Exception
            If the shape is unsupported or the emission would place particles outside the simulation boundary.
        """
        assert self._entity is not None

        if droplet_shape in ["circle", "sphere", "square"]:
            assert isinstance(droplet_size, (int, float))
        elif droplet_shape == "rectangle":
            assert isinstance(droplet_size, (tuple, list)) and len(droplet_size) == 2
        else:
            gs.raise_exception(f"Unsupported nozzle shape: {droplet_shape}.")

        direction = np.asarray(direction, dtype=gs.np_float)
        if np.linalg.norm(direction) < gs.EPS:
            gs.raise_exception("Zero-length direction.")
        else:
            direction = gu.normalize(direction)

        p_size = self._solver.particle_size if p_size is None else p_size

        if droplet_length is None:
            # Use the speed to determine the length of the droplet in the emitting direction
            droplet_length = speed * self._solver.substep_dt * self._sim.substeps + self._acc_droplet_len
            if droplet_length < p_size:  # too short, so we should not emit
                self._acc_droplet_len = droplet_length
                droplet_length = 0.0
            else:
                self._acc_droplet_len = 0.0

        if droplet_length > 0.0:
            if droplet_shape == "circle":
                positions = pu.cylinder_to_particles(
                    p_size=p_size,
                    radius=droplet_size / 2,
                    height=droplet_length,
                    sampler=self._entity.sampler,
                )
            elif droplet_shape == "sphere":  # sphere droplet ignores droplet_length
                positions = pu.sphere_to_particles(
                    p_size=p_size,
                    radius=droplet_size / 2,
                    sampler=self._entity.sampler,
                )
            elif droplet_shape == "square":
                positions = pu.box_to_particles(
                    p_size=p_size,
                    size=np.array([droplet_size, droplet_size, droplet_length]),
                    sampler=self._entity.sampler,
                )
            elif droplet_shape == "rectangle":
                positions = pu.box_to_particles(
                    p_size=p_size,
                    size=np.array([droplet_size[0], droplet_size[1], droplet_length]),
                    sampler=self._entity.sampler,
                )
            else:
                gs.raise_exception()

            positions = gu.transform_by_trans_R(
                positions.astype(gs.np_float, copy=False),
                np.asarray(pos, dtype=gs.np_float),
                gu.z_up_to_R(direction) @ gu.axis_angle_to_R(np.array([0.0, 0.0, 1.0], dtype=gs.np_float), theta),
            )

            positions = np.tile(positions[np.newaxis], (self._sim._B, 1, 1))

            if not self._solver.boundary.is_inside(positions):
                gs.raise_exception("Emitted particles are outside the boundary.")

            n_particles = positions.shape[1]

            # Expand vels with batch dimension
            vels = speed * direction
            vels = np.tile(vels.reshape((1, 1, -1)), (self._sim._B, n_particles, 1))

            if n_particles > self._entity.n_particles:
                gs.logger.warning(
                    f"Number of particles to emit ({n_particles}) at the current step is larger than the maximum number of particles ({self._entity.n_particles})."
                )

            self._solver._kernel_set_particles_pos(
                self._sim.cur_substep_local,
                self._entity.particle_start + self._next_particle,
                n_particles,
                positions,
            )
            self._solver._kernel_set_particles_vel(
                self._sim.cur_substep_local,
                self._entity.particle_start + self._next_particle,
                n_particles,
                vels,
            )
            self._solver._kernel_set_particles_active(
                self._sim.cur_substep_local,
                self._entity.particle_start + self._next_particle,
                n_particles,
                gs.ACTIVE,
            )

            self._next_particle += n_particles

            # recycle particles
            if self._next_particle + n_particles > self._entity.n_particles:
                self._next_particle = 0

            gs.logger.debug(f"Emitted {n_particles} particles. Next particle index: {self._next_particle}.")

        else:
            gs.logger.debug("Droplet length is too short for current step. Skipping to next step.")

    def emit_omni(self, source_radius=0.1, pos=(0.5, 0.5, 1.0), speed=1.0, particle_size=None):
        """
        Use a sphere-shaped source to emit particles in all directions.

        Parameters:
        ----------
        source_radius: float, optional
            The radius of the sphere source. Particles will be emitted from a shell with inner radius using
            '0.8 * source_radius' and outer radius using source_radius.
        pos: array_like, shape=(3,)
            The center of the sphere source.
        speed: float
            The speed of the emitted particles.
        particle_size: float | None
            The size (diameter) of the emitted particles. The actual number of particles emitted is determined by the
            volume of the sphere source and the size of the particles. If None, the solver's particle size is used.
            Note that this particle size only affects computation for number of particles emitted, not the actual size
            of the particles in simulation and rendering.
        """
        assert self._entity is not None

        pos = np.asarray(pos, dtype=gs.np_float)

        if particle_size is None:
            particle_size = self._solver.particle_size

        positions_ = pu.shell_to_particles(
            p_size=particle_size,
            outer_radius=source_radius,
            inner_radius=source_radius * 0.4,
            sampler=self._entity.sampler,
        )
        positions = pos + positions_

        if not self._solver.boundary.is_inside(positions):
            gs.raise_exception("Emitted particles are outside the boundary.")

        dists = np.linalg.norm(positions_, axis=1)
        positions[dists < gs.EPS] = gs.EPS
        vels = (speed / (dists + gs.EPS)) * positions_

        n_particles = len(positions)
        if n_particles > self._entity.n_particles:
            gs.logger.warning(
                f"Number of particles to emit ({n_particles}) at the current step is larger than the maximum number "
                f"of particles ({self._entity.n_particles})."
            )

        self._solver._kernel_set_particles_pos(
            self._sim.cur_substep_local,
            self._entity.particle_start + self._next_particle,
            n_particles,
            positions,
        )
        self._solver._kernel_set_particles_vel(
            self._sim.cur_substep_local,
            self._entity.particle_start + self._next_particle,
            n_particles,
            vels,
        )
        self._solver._kernel_set_particles_active(
            self._sim.cur_substep_local,
            self._entity.particle_start + self._next_particle,
            n_particles,
            gs.ACTIVE,
        )

        self._next_particle += n_particles

        # recycle particles
        if self._next_particle + n_particles > self._entity.n_particles:
            self._next_particle = 0

        gs.logger.debug(f"Emitted {n_particles} particles. Next particle index: {self._next_particle}.")

    @property
    def uid(self):
        """The unique identifier of the emitter."""
        return self._uid

    @property
    def entity(self):
        """The entity associated with the emitter."""
        return self._entity

    @property
    def max_particles(self):
        """The maximum number of particles this emitter can emit."""
        return self._max_particles

    @property
    def solver(self):
        """The solver used by the emitter's associated entity."""
        return self._solver

    @property
    def next_particle(self):
        """The index of the next particle to be emitted."""
        return self._next_particle























# import numpy as np
# import taichi as ti

# import genesis as gs
# import genesis.utils.geom as gu
# import genesis.utils.particle as pu
# from genesis.repr_base import RBC


# @ti.data_oriented
# class Emitter(RBC):
#     """
#     Extended Emitter with foam-curing support.

#     Backward-compatible methods:
#       - set_entity(entity)
#       - reset()
#       - emit(...)
#       - emit_omni(...)

#     New foam-related additions:
#       - set_carriers(entities)         # optional: multi-carrier age buckets
#       - emit_foam(...)                 # liquid → expansion → hardening
#     """

#     # ──────────────────────────────────────────────────────────────────────
#     # Constructor & basic setup
#     # ──────────────────────────────────────────────────────────────────────
#     def __init__(self, max_particles):
#         self._uid = gs.UID()
#         self._entity = None

#         self._max_particles = max_particles

#         self._acc_droplet_len = 0.0  # accumulated droplet length to be emitted
#         self._next_particle = 0

#         # Foam extensions (all optional unless emit_foam is used)
#         self._carriers = None                 # list[entity], if using buckets
#         self._next_particle_map = {}          # per-carrier ring pointers
#         self._foam_enabled = False
#         self._foam_cfg = None                 # filled on first emit_foam call
#         self._foam_step = 0                   # increments each emit_foam call
#         self._foam_time = 0.0                 # internal time tracker (s)
#         self._bucket_age = []                 # seconds since last "fresh" batch
#         self._bucket_last_pos = []            # remember last emit position
#         self._bucket_is_hardened = []         # flags
#         self._bucket_stride_steps = 6         # default stride (can override)

#         gs.logger.info(
#             f"Creating ~<{self._repr_type()}>~. id: ~~~<{self._uid}>~~~, max_particles: ~<{max_particles}>~."
#         )

#     # ──────────────────────────────────────────────────────────────────────
#     # Carriers & book-keeping
#     # ──────────────────────────────────────────────────────────────────────
#     def set_entity(self, entity):
#         """
#         Bind a single carrier entity (legacy behavior). For foam with proper
#         phase separation, prefer set_carriers([...]) to model age buckets.
#         """
#         self._entity = entity
#         self._sim = entity.sim
#         self._solver = entity.solver
#         self._next_particle = 0
#         gs.logger.info(f"~<{self._repr_briefer()}>~ created using ~<{entity._repr_briefer()}>.")

#     def set_carriers(self, entities):
#         """
#         Optional: register multiple carrier entities. emit_foam can rotate through
#         them so that each emitted batch ages independently (liquid/expansion/cure)
#         without changing solver internals.
#         """
#         assert isinstance(entities, (list, tuple)) and len(entities) > 0
#         self._carriers = list(entities)
#         # All carriers must share the same sim/solver
#         sims = {e.sim for e in self._carriers}
#         solvers = {e.solver for e in self._carriers}
#         assert len(sims) == 1 and len(solvers) == 1, "All carriers must share the same sim/solver."

#         self._entity = self._carriers[0]
#         self._sim = self._entity.sim
#         self._solver = self._entity.solver

#         # Per-carrier ring pointers + foam ages
#         self._next_particle_map = {id(e): 0 for e in self._carriers}
#         self._bucket_age = [0.0 for _ in self._carriers]
#         self._bucket_last_pos = [None for _ in self._carriers]
#         self._bucket_is_hardened = [False for _ in self._carriers]

#         gs.logger.info(
#             f"~<{self._repr_briefer()}>~ registered {len(self._carriers)} carriers for foam age buckets."
#         )

#     def _get_next_pointer(self, entity, n_particles):
#         """
#         Ring-buffer pointer within the given entity's particle slice.
#         Supports multi-carrier mode.
#         """
#         if self._carriers is None:
#             start = entity.particle_start + self._next_particle
#             idx = self._next_particle
#             # recycle on overflow
#             if idx + n_particles > entity.n_particles:
#                 idx = 0
#             return start, idx
#         else:
#             key = id(entity)
#             idx = self._next_particle_map[key]
#             start = entity.particle_start + idx
#             if idx + n_particles > entity.n_particles:
#                 idx = 0
#                 start = entity.particle_start
#             return start, idx

#     def _advance_pointer(self, entity, n_particles):
#         if self._carriers is None:
#             self._next_particle += n_particles
#             if self._next_particle + n_particles > self._entity.n_particles:
#                 self._next_particle = 0
#         else:
#             key = id(entity)
#             self._next_particle_map[key] += n_particles
#             if self._next_particle_map[key] + n_particles > entity.n_particles:
#                 self._next_particle_map[key] = 0

#     def reset(self):
#         self._next_particle = 0
#         self._acc_droplet_len = 0.0
#         # Foam state
#         self._foam_enabled = False
#         self._foam_cfg = None
#         self._foam_step = 0
#         self._foam_time = 0.0
#         if self._carriers is not None:
#             for k in self._next_particle_map:
#                 self._next_particle_map[k] = 0
#             self._bucket_age = [0.0 for _ in self._carriers]
#             self._bucket_last_pos = [None for _ in self._carriers]
#             self._bucket_is_hardened = [False for _ in self._carriers]

#     # ──────────────────────────────────────────────────────────────────────
#     # Legacy emit — now outputs (B, N, 3) to kernels
#     # ──────────────────────────────────────────────────────────────────────
#     def emit(
#         self,
#         droplet_shape,
#         droplet_size,
#         droplet_length=None,
#         pos=(0.5, 0.5, 1.0),
#         direction=(0, 0, -1),
#         theta=0.0,
#         speed=1.0,
#         p_size=None,
#     ):
#         assert self._entity is not None

#         if droplet_shape in ["circle", "sphere", "square"]:
#             assert isinstance(droplet_size, (int, float))
#         elif droplet_shape == "rectangle":
#             assert isinstance(droplet_size, (tuple, list)) and len(droplet_size) == 2
#         else:
#             gs.raise_exception(f"Unsupported nozzle shape: {droplet_shape}.")

#         direction = np.asarray(direction, dtype=gs.np_float)
#         if np.linalg.norm(direction) < gs.EPS:
#             gs.raise_exception("Zero-length direction.")
#         else:
#             direction = gu.normalize(direction)

#         p_size = self._solver.particle_size if p_size is None else p_size
#         pos = np.asarray(pos, dtype=gs.np_float)

#         if droplet_length is None:
#             droplet_length = (
#                 speed * self._solver.substep_dt * self._sim.substeps + self._acc_droplet_len
#             )
#             if droplet_length < p_size:  # too short; accumulate for later
#                 self._acc_droplet_len = droplet_length
#                 droplet_length = 0.0
#             else:
#                 self._acc_droplet_len = 0.0

#         if droplet_length > 0.0:
#             if droplet_shape == "circle":
#                 positions = pu.cylinder_to_particles(
#                     p_size=p_size,
#                     radius=droplet_size / 2,
#                     height=droplet_length,
#                     sampler=self._entity.sampler,
#                 )
#             elif droplet_shape == "sphere":  # sphere droplet ignores droplet_length
#                 positions = pu.sphere_to_particles(
#                     p_size=p_size,
#                     radius=droplet_size / 2,
#                     sampler=self._entity.sampler,
#                 )
#             elif droplet_shape == "square":
#                 positions = pu.box_to_particles(
#                     p_size=p_size,
#                     size=np.array([droplet_size, droplet_size, droplet_length]),
#                     sampler=self._entity.sampler,
#                 )
#             elif droplet_shape == "rectangle":
#                 positions = pu.box_to_particles(
#                     p_size=p_size,
#                     size=np.array([droplet_size[0], droplet_size[1], droplet_length]),
#                     sampler=self._entity.sampler,
#                 )
#             else:
#                 gs.raise_exception()

#             # (N,3) in local → (N,3) in world
#             positions = gu.transform_by_T(
#                 positions,
#                 gu.trans_R_to_T(
#                     pos,
#                     gu.z_to_R(direction) @ gu.axis_angle_to_R(np.array([0, 0, 1]), theta),
#                 ),
#             ).astype(gs.np_float)

#             # tile to (B,N,3)
#             B = self._sim._B
#             positions_B = np.tile(positions[None], (B, 1, 1))  # (B,N,3)
#             if not self._solver.boundary.is_inside(positions_B):
#                 gs.raise_exception("Emitted particles are outside the boundary.")

#             n_particles = len(positions)
#             v_single = (direction * speed).astype(gs.np_float, copy=False)
#             vels = np.tile(v_single, (n_particles, 1))          # (N,3)
#             vels_B = np.tile(vels[None], (B, 1, 1))             # (B,N,3)

#             if n_particles > self._entity.n_particles:
#                 gs.logger.warning(
#                     f"Number of particles to emit ({n_particles}) at the current step "
#                     f"is larger than the maximum number of particles ({self._entity.n_particles})."
#                 )

#             start_abs, _local_idx = self._get_next_pointer(self._entity, n_particles)

#             self._solver._kernel_set_particles_pos(
#                 self._sim.cur_substep_local, start_abs, n_particles, positions_B
#             )
#             self._solver._kernel_set_particles_vel(
#                 self._sim.cur_substep_local, start_abs, n_particles, vels_B
#             )
#             self._solver._kernel_set_particles_active(
#                 self._sim.cur_substep_local, start_abs, n_particles, gs.ACTIVE
#             )

#             self._advance_pointer(self._entity, n_particles)
#             gs.logger.debug(
#                 f"Emitted {n_particles} particles. Next local pointer: "
#                 f"{self._next_particle if self._carriers is None else self._next_particle_map[id(self._entity)]}."
#             )
#         else:
#             gs.logger.debug("Droplet length is too short for current step. Skipping to next step.")

#     # ──────────────────────────────────────────────────────────────────────
#     # Omni emit — now outputs (B, N, 3)
#     # ──────────────────────────────────────────────────────────────────────
#     def emit_omni(self, source_radius=0.1, pos=(0.5, 0.5, 1.0), speed=1.0, particle_size=None):
#         """
#         Emit particles from a spherical shell, in all directions.
#         """
#         assert self._entity is not None

#         pos = np.asarray(pos, dtype=gs.np_float)
#         particle_size = self._solver.particle_size if particle_size is None else particle_size

#         positions_local = pu.shell_to_particles(
#             p_size=particle_size,
#             outer_radius=source_radius,
#             inner_radius=source_radius * 0.4,
#             sampler=self._entity.sampler,
#         )
#         # handle zero radius samples before transform
#         dists = np.linalg.norm(positions_local, axis=1, keepdims=True)
#         zero_idx = np.where(dists < gs.EPS)[0]
#         if len(zero_idx) > 0:
#             positions_local[zero_idx] = np.array([gs.EPS, gs.EPS, gs.EPS])

#         positions = gu.transform_by_T(positions_local, gu.trans_to_T(pos)).astype(gs.np_float)  # (N,3)
#         B = self._sim._B
#         positions_B = np.tile(positions[None], (B, 1, 1))  # (B,N,3)
#         if not self._solver.boundary.is_inside(positions_B):
#             gs.raise_exception("Emitted particles are outside the boundary.")

#         n_particles = len(positions)
#         vels = (positions_local / dists * speed).astype(gs.np_float)  # (N,3)
#         vels_B = np.tile(vels[None], (B, 1, 1))                       # (B,N,3)

#         if n_particles > self._entity.n_particles:
#             gs.logger.warning(
#                 f"Number of particles to emit ({n_particles}) at the current step "
#                 f"is larger than the maximum number of particles ({self._entity.n_particles})."
#             )

#         start_abs, _local_idx = self._get_next_pointer(self._entity, n_particles)

#         self._solver._kernel_set_particles_pos(
#             self._sim.cur_substep_local, start_abs, n_particles, positions_B
#         )
#         self._solver._kernel_set_particles_vel(
#             self._sim.cur_substep_local, start_abs, n_particles, vels_B
#         )
#         self._solver._kernel_set_particles_active(
#             self._sim.cur_substep_local, start_abs, n_particles, gs.ACTIVE
#         )

#         self._advance_pointer(self._entity, n_particles)
#         gs.logger.debug(
#             f"Emitted {n_particles} particles (omni). Next local pointer: "
#             f"{self._next_particle if self._carriers is None else self._next_particle_map[id(self._entity)]}."
#         )

#     # ──────────────────────────────────────────────────────────────────────
#     # NEW: Foam emitter (liquid → expansion → hardening) — uses (B,N,3)
#     # ──────────────────────────────────────────────────────────────────────
#     def emit_foam(
#         self,
#         # geometry / kinematics (similar to emit)
#         droplet_shape="sphere",
#         droplet_size=0.02,
#         pos=(0.5, 0.5, 1.0),
#         direction=(0, 0, -1),
#         theta=0.0,
#         speed=0.4,
#         p_size=None,
#         # schedule / curing profile (seconds)
#         liquid_duration=0.2,        # time window that stays "liquid-like"
#         expansion_duration=0.3,     # expansion (rise) window after liquid
#         cure_total=0.8,             # time until considered "hardened"
#         # expansion controls
#         expansion_ratio=1.5,        # target radius multiplier across expansion
#         expansion_rings=6,          # emit rings per step during expansion
#         expansion_shell_radius=0.015,
#         expansion_shell_speed=0.2,
#         expansion_particle_size=None,
#         # material transition
#         hardening_material=None,    # e.g., gs.materials.MPM.ElastoPlastic(...)
#         # buckets (phase separation without solver mods)
#         use_buckets=True,
#         bucket_stride_steps=6,      # how many steps to keep writing into same bucket
#     ):
#         """
#         Emit "foam" with a 3-phase curing schedule:
#           1) Liquid       [0, liquid_duration)
#           2) Expansion    [liquid_duration, liquid_duration + expansion_duration)
#           3) Hardened     [>= cure_total]  (we also flip material when reached)
#         """
#         assert self._entity is not None, "Call set_entity(...) or set_carriers([...]) first."

#         # Initialize foam config on first call or when parameters change
#         self._foam_enabled = True
#         self._bucket_stride_steps = int(max(1, bucket_stride_steps))

#         if self._foam_cfg is None:
#             self._foam_cfg = dict(
#                 liquid_duration=float(liquid_duration),
#                 expansion_duration=float(expansion_duration),
#                 cure_total=float(cure_total),
#                 expansion_ratio=float(max(1.0, expansion_ratio)),
#                 expansion_rings=int(max(0, expansion_rings)),
#                 expansion_shell_radius=float(expansion_shell_radius),
#                 expansion_shell_speed=float(expansion_shell_speed),
#                 hardening_material=hardening_material,
#             )
#             # ensure bucket state exists if user didn't call set_carriers
#             if self._carriers is None:
#                 self._carriers = [self._entity]
#                 self._next_particle_map = {id(self._entity): self._next_particle}
#                 self._bucket_age = [0.0]
#                 self._bucket_last_pos = [None]
#                 self._bucket_is_hardened = [False]

#         # Resolve dt for internal time tracking (one call ~ one frame)
#         dt = self._solver.substep_dt * self._sim.substeps
#         self._foam_time += dt
#         self._foam_step += 1

#         # choose current bucket to receive fresh droplet
#         K = len(self._carriers)
#         b = (self._foam_step // self._bucket_stride_steps) % K
#         carrier = self._carriers[b]
#         self._entity = carrier  # re-bind current target

#         # Geometry defaults
#         direction = np.asarray(direction, dtype=gs.np_float)
#         if np.linalg.norm(direction) < gs.EPS:
#             gs.raise_exception("Zero-length direction.")
#         else:
#             direction = gu.normalize(direction)
#         p_size_eff = self._solver.particle_size if p_size is None else p_size
#         pos = np.asarray(pos, dtype=gs.np_float)

#         # Emit the base droplet (liquid-ish newborn batch)
#         droplet_len = speed * dt + self._acc_droplet_len
#         if droplet_shape != "sphere" and droplet_len < p_size_eff:
#             self._acc_droplet_len = droplet_len
#         else:
#             self._acc_droplet_len = 0.0
#             if droplet_shape == "sphere":
#                 positions_local = pu.sphere_to_particles(
#                     p_size=p_size_eff, radius=droplet_size * 0.5, sampler=carrier.sampler
#                 )
#             elif droplet_shape == "circle":
#                 positions_local = pu.cylinder_to_particles(
#                     p_size=p_size_eff, radius=droplet_size * 0.5, height=droplet_len, sampler=carrier.sampler
#                 )
#             elif droplet_shape == "square":
#                 positions_local = pu.box_to_particles(
#                     p_size=p_size_eff,
#                     size=np.array([droplet_size, droplet_size, droplet_len]),
#                     sampler=carrier.sampler,
#                 )
#             elif droplet_shape == "rectangle":
#                 assert isinstance(droplet_size, (tuple, list)) and len(droplet_size) == 2
#                 positions_local = pu.box_to_particles(
#                     p_size=p_size_eff,
#                     size=np.array([droplet_size[0], droplet_size[1], droplet_len]),
#                     sampler=carrier.sampler,
#                 )
#             else:
#                 gs.raise_exception(f"Unsupported nozzle shape: {droplet_shape}.")

#             positions = gu.transform_by_T(
#                 positions_local,
#                 gu.trans_R_to_T(
#                     pos,
#                     gu.z_to_R(direction) @ gu.axis_angle_to_R(np.array([0, 0, 1]), theta),
#                 ),
#             ).astype(gs.np_float)  # (N,3)

#             B = self._sim._B
#             positions_B = np.tile(positions[None], (B, 1, 1))  # (B,N,3)
#             if not self._solver.boundary.is_inside(positions_B):
#                 gs.raise_exception("Emitted particles are outside the boundary.")

#             n_particles = len(positions)
#             v_single = (direction * speed).astype(gs.np_float, copy=False)
#             vels = np.tile(v_single, (n_particles, 1))          # (N,3)
#             vels_B = np.tile(vels[None], (B, 1, 1))             # (B,N,3)

#             start_abs, _local_idx = self._get_next_pointer(carrier, n_particles)
#             self._solver._kernel_set_particles_pos(self._sim.cur_substep_local, start_abs, n_particles, positions_B)
#             self._solver._kernel_set_particles_vel(self._sim.cur_substep_local, start_abs, n_particles, vels_B)
#             self._solver._kernel_set_particles_active(self._sim.cur_substep_local, start_abs, n_particles, gs.ACTIVE)
#             self._advance_pointer(carrier, n_particles)

#             self._bucket_last_pos[b] = tuple(pos.tolist())

#         # Update ages for all buckets and perform expansion/hardening actions
#         cfg = self._foam_cfg
#         T_liq = cfg["liquid_duration"]
#         T_exp = cfg["expansion_duration"]
#         T_cure = cfg["cure_total"]
#         target_ratio = cfg["expansion_ratio"]

#         rings = cfg["expansion_rings"]
#         shell_R = cfg["expansion_shell_radius"]
#         shell_speed = cfg["expansion_shell_speed"]
#         psize_shell = self._solver.particle_size if expansion_particle_size is None else expansion_particle_size

#         for k, ent in enumerate(self._carriers):
#             self._bucket_age[k] += dt
#             age = self._bucket_age[k]

#             # Phase 2: expansion window
#             if T_liq <= age < (T_liq + T_exp) and self._bucket_last_pos[k] is not None and rings > 0:
#                 prog = (age - T_liq) / max(gs.EPS, T_exp)  # 0..1
#                 ring_R_cur = shell_R * (1.0 + (target_ratio - 1.0) * prog)
#                 cx, cy, cz = self._bucket_last_pos[k]

#                 prev_ent = self._entity
#                 self._entity = ent
#                 for j in range(rings):
#                     ang = 2.0 * np.pi * j / rings
#                     dx, dy = ring_R_cur * np.cos(ang), ring_R_cur * np.sin(ang)
#                     self.emit(
#                         droplet_shape="sphere",
#                         droplet_size=max(0.5 * psize_shell, 0.7 * psize_shell),
#                         droplet_length=None,
#                         pos=(cx + dx, cy + dy, cz),
#                         direction=(0, 0, -1),
#                         theta=0.0,
#                         speed=shell_speed,
#                         p_size=psize_shell,
#                     )
#                 self._entity = prev_ent

#             # Phase 3: hardening trigger
#             if (age >= T_cure) and (not self._bucket_is_hardened[k]) and (cfg["hardening_material"] is not None):
#                 try:
#                     ent.material = cfg["hardening_material"]
#                     self._bucket_is_hardened[k] = True
#                     gs.logger.info(
#                         f"Foam bucket {k} reached cure_total={T_cure:.3f}s; "
#                         f"switched material to <{ent.material.__class__.__name__}>."
#                     )
#                 except Exception as ex:
#                     gs.logger.warning(f"Failed to switch material on bucket {k}: {ex}")

#         # Reset fresh bucket age (just emitted into it)
#         self._bucket_age[b] = 0.0

#     # ──────────────────────────────────────────────────────────────────────
#     # Properties
#     # ──────────────────────────────────────────────────────────────────────
#     @property
#     def uid(self):
#         return self._uid

#     @property
#     def entity(self):
#         return self._entity

#     @property
#     def max_particles(self):
#         return self._max_particles

#     @property
#     def solver(self):
#         return self._solver

#     @property
#     def next_particle(self):
#         # For multi-carrier, returns pointer for the "current" entity
#         if self._carriers is None:
#             return self._next_particle
#         return self._next_particle_map.get(id(self._entity), 0)