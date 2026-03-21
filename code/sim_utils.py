import pinocchio as pin
import hppfcl
import numpy as np
import tap
import simple
import matplotlib.pyplot as plt
import time
import meshcat
from pinocchio.visualize import MeshcatVisualizer

try:
    import mujoco
    from robot_descriptions.loaders.mujoco import load_robot_description as mlr

    mujoco_imported = True
except:
    mujoco_imported = False


class SimulationArgs(tap.Tap):
    horizon: int = 1000
    dt: float = 1e-3
    contact_solver: str = "ADMM"  # set to PGS or ADMM
    maxit: int = 100  # solver maxit
    tol: float = 1e-6  # absolute constraint solver tol
    tol_rel: float = 1e-6  # relative constraint solver tol
    warm_start: int = 1  # warm start the solver?
    mu_prox: float = 1e-4  # prox value for admm
    material: str = "metal"  # contact friction
    compliance: float = 0
    max_contacts_per_pair: int = 4
    mujoco: bool = False
    Kp: float = 0  # baumgarte proportional term
    Kd: float = 0  # baumgarte derivative term
    seed: int = 1234
    random_init_vel: bool = False
    display: bool = False
    display_traj: bool = False
    solve_ccp: bool = False
    debug: bool = False
    debug_step: int = -1
    dont_stop: bool = False
    admm_update_rule: str = "spectral"
    ratio_primal_dual: float = 10
    tau: float = 0.5
    rho: float = 10.0
    rho_power: float = 0.05
    rho_power_factor: float = 0.05
    lanczos_size: int = 3
    linear_update_rule_factor: float = 2

'''
#Αυτη ειναι η og συναρτηση
def setupSimulatorFromArgs(sim: simple.Simulator, args: SimulationArgs):
    sim.warm_start_contact_forces = args.warm_start
    sim.constraints_problem.setMaxNumberOfContactsPerCollisionPair(
        args.max_contacts_per_pair
    )
    sim.constraints_problem.Kp = args.Kp
    sim.constraints_problem.Kd = args.Kd
    sim.constraints_problem.is_ncp = not args.solve_ccp
    # PGS
    sim.pgs_constraint_solver_settings.stat_record = args.debug
    sim.pgs_constraint_solver_settings.max_iter = args.maxit
    sim.pgs_constraint_solver_settings.absolute_precision = args.tol
    sim.pgs_constraint_solver_settings.relative_precision = args.tol_rel
    # ADMM
    sim.admm_constraint_solver_settings.stat_record = args.debug
    sim.admm_constraint_solver_settings.max_iter = args.maxit
    sim.admm_constraint_solver_settings.absolute_precision = args.tol
    sim.admm_constraint_solver_settings.relative_precision = args.tol_rel
    sim.admm_constraint_solver_settings.ratio_primal_dual = args.ratio_primal_dual
    sim.admm_constraint_solver_settings.mu = args.mu_prox
    sim.admm_constraint_solver_settings.rho = args.rho
    sim.admm_constraint_solver_settings.tau = args.tau
    if args.admm_update_rule == "spectral":
        sim.admm_constraint_solver_settings.admm_update_rule = (
            pin.ADMMUpdateRule.SPECTRAL
        )
        sim.admm_constraint_solver_settings.rho_power_factor = args.rho_power_factor
        sim.admm_constraint_solver_settings.rho_power = args.rho_power
        sim.admm_constraint_solver_settings.lanczos_size = args.lanczos_size
    elif args.admm_update_rule == "linear":
        sim.admm_constraint_solver_settings.admm_update_rule = pin.ADMMUpdateRule.LINEAR
        sim.admm_constraint_solver_settings.linear_update_rule_factor = (
            args.linear_update_rule_factor
        )
    elif args.admm_update_rule == "constant":
        sim.admm_constraint_solver_settings.admm_update_rule = (
            pin.ADMMUpdateRule.CONSTANT
        )
'''

#Αυτη εινια η αλλαγμενη για να ταιριαζει με την αλλη

def setupSimulatorFromArgs(sim: simple.Simulator, geom_data: pin.GeometryData, args: SimulationArgs):
    # --- 1. ΡΥΘΜΙΣΗ ΤΟΥ "ΡΑΝΤΑΡ" ΣΥΓΚΡΟΥΣΕΩΝ (HPP-FCL) - ΑΥΤΟ ΠΟΥ ΕΛΕΙΠΕ! ---
    for col_req in geom_data.collisionRequests:
        col_req: hppfcl.CollisionRequest
        # Συγχρονίζουμε το ραντάρ με την αόρατη ασπίδα:
        col_req.security_margin = args.patch_tolerance #ποτε θεωρουμε οτι αρχιζει η συγκρουση  
        col_req.break_distance = args.patch_tolerance + 1e-4  #ποτε θεωρουμε οτι τελειωσε η συγκρουση γιαυτο προσθετουμε το +1e-3
        
        # Υψηλή ακρίβεια στα μαθηματικά της σύγκρουσης
        col_req.gjk_tolerance = 1e-6
        col_req.epa_tolerance = 1e-6
        col_req.gjk_initial_guess = hppfcl.GJKInitialGuess.CachedGuess
        col_req.gjk_variant = hppfcl.GJKVariant.DefaultGJK

    for patch_req in geom_data.contactPatchRequests:
        patch_req.setPatchTolerance(args.patch_tolerance) #εδω λεμε δημιουργησε μια αορατη ασπιδα τοσης ανοχης οσο patch tolerance στην οποια θα ψαξει για τα 4 σημεια που εχουμε ορισει με max_contacts_per_pair

    # --- 2. ΡΥΘΜΙΣΗ ΤΟΥ SIMULATOR ΚΑΙ ΤΩΝ ΕΛΑΤΗΡΙΩΝ (Ο παλιός σου κώδικας) ---
    sim.warm_start_contact_forces = args.warm_start
    sim.constraints_problem.setMaxNumberOfContactsPerCollisionPair(
        args.max_contacts_per_pair
    )
    sim.constraints_problem.Kp = args.Kp
    sim.constraints_problem.Kd = args.Kd
    sim.constraints_problem.is_ncp = not args.solve_ccp
    
    # PGS
    sim.pgs_constraint_solver_settings.stat_record = args.debug
    sim.pgs_constraint_solver_settings.max_iter = args.maxit
    sim.pgs_constraint_solver_settings.absolute_precision = args.tol
    sim.pgs_constraint_solver_settings.relative_precision = args.tol_rel
    
    # ADMM
    sim.admm_constraint_solver_settings.stat_record = args.debug
    sim.admm_constraint_solver_settings.max_iter = args.maxit
    sim.admm_constraint_solver_settings.absolute_precision = args.tol
    sim.admm_constraint_solver_settings.relative_precision = args.tol_rel
    sim.admm_constraint_solver_settings.ratio_primal_dual = args.ratio_primal_dual
    sim.admm_constraint_solver_settings.mu = args.mu_prox
    sim.admm_constraint_solver_settings.rho = args.rho
    sim.admm_constraint_solver_settings.tau = args.tau
    
    if args.admm_update_rule == "spectral":
        sim.admm_constraint_solver_settings.admm_update_rule = (
            pin.ADMMUpdateRule.SPECTRAL
        )
        sim.admm_constraint_solver_settings.rho_power_factor = args.rho_power_factor
        sim.admm_constraint_solver_settings.rho_power = args.rho_power
        sim.admm_constraint_solver_settings.lanczos_size = args.lanczos_size
    elif args.admm_update_rule == "linear":
        sim.admm_constraint_solver_settings.admm_update_rule = pin.ADMMUpdateRule.LINEAR
        sim.admm_constraint_solver_settings.linear_update_rule_factor = (
            args.linear_update_rule_factor
        )
    elif args.admm_update_rule == "constant":
        sim.admm_constraint_solver_settings.admm_update_rule = (
            pin.ADMMUpdateRule.CONSTANT
        )



def plotContactSolver(
    sim: simple.Simulator,
    args: SimulationArgs,
    t: int,
    q: np.ndarray,
    v: np.ndarray,
):
    if args.debug or t == args.debug_step:
        stats: pin.SolverStats = sim.admm_constraint_solver.getStats()
        if args.contact_solver == "ADMM":
            solver = sim.admm_constraint_solver
        if args.contact_solver == "PGS":
            solver = sim.pgs_constraint_solver
        stats = solver.getStats()
        abs_res = solver.getAbsoluteConvergenceResidual()
        rel_res = solver.getRelativeConvergenceResidual()
        it = solver.getIterationCount()
        if stats.size() > 0:
            plt.cla()
            title = (
                f"Step {t}, it = {it}, abs res = {abs_res:.2e}, rel res = {rel_res:.2e}"
            )
            if args.contact_solver == "ADMM":
                title += f", cholesky count: {stats.cholesky_update_count}"
            plt.title(title)
            plt.plot(stats.primal_feasibility, label="primal feas")
            plt.plot(stats.dual_feasibility, label="dual feas")
            plt.plot(stats.dual_feasibility_ncp, label="dual feas NCP")
            if args.contact_solver == "ADMM":
                plt.plot(stats.dual_feasibility_admm, label="dual feas ADMM")
                plt.plot(
                    stats.dual_feasibility_constraint, label="dual feas constraint"
                )
                plt.plot(stats.rho, label="rho")
            if args.contact_solver == "PGS":
                plt.plot(stats.complementarity, label="complementarity")
            plt.yscale("log")
            plt.legend()
            plt.ion()
            plt.show()
        print(f"{t=}")
        print(f"{q=}")
        print(f"{v=}")
        print(f"{sim.qnew=}")
        print(f"{sim.vnew=}")
        print(f"{sim.admm_constraint_solver.getIterationCount()=}")
        print(f"{sim.pgs_constraint_solver.getIterationCount()=}")
        print(f"{sim.constraints_problem.constraints_forces=}")
        print(f"{sim.constraints_problem.constraints_problem_size=}")
        print(f"{sim.constraints_problem.joint_friction_constraint_size=}")
        print(f"{sim.constraints_problem.joint_limit_constraint_size=}")
        print(f"{sim.constraints_problem.bilateral_constraints_size=}")
        print(f"{sim.constraints_problem.weld_constraints_size=}")
        print(f"{sim.constraints_problem.frictional_point_constraints_size=}")
        print("Constraint solver timings: ", sim.getConstraintSolverCPUTimes().user)
        input("[Press enter to continue]")


def subSample(xs, duration, fps):
    nb_frames = len(xs)
    nb_subframes = int(duration * fps)
    if nb_frames < nb_subframes:
        return xs
    else:
        step = nb_frames // nb_subframes
        xs_sub = [xs[i] for i in range(0, nb_frames, step)]
        return xs_sub


def addSystemCollisionPairs(model, geom_model, qref):
    """
    Add the right collision pairs of a model, given qref.
    qref is here as a `T-pose`. The function uses this pose to determine which objects are in collision
    in this ref pose. If objects are in collision, they are not added as collision pairs, as they are considered
    to always be in collision.
    """
    data = model.createData()
    geom_data = geom_model.createData()
    pin.updateGeometryPlacements(model, data, geom_model, geom_data, qref)
    geom_model.removeAllCollisionPairs()
    num_col_pairs = 0
    for i in range(len(geom_model.geometryObjects)):
        for j in range(i + 1, len(geom_model.geometryObjects)):
            # Don't add collision pair if same object
            if i != j:
                gobj_i: pin.GeometryObject = geom_model.geometryObjects[i]
                gobj_j: pin.GeometryObject = geom_model.geometryObjects[j]
                if gobj_i.name == "floor" or gobj_j.name == "floor":
                    num_col_pairs += 1
                    col_pair = pin.CollisionPair(i, j)
                    geom_model.addCollisionPair(col_pair)
                else:
                    if gobj_i.parentJoint != gobj_j.parentJoint:
                        # Compute collision between the geometries. Only add the collision pair if there is no collision.
                        M1 = geom_data.oMg[i]
                        M2 = geom_data.oMg[j]
                        colreq = hppfcl.CollisionRequest()
                        colreq.security_margin = 1e-2  # 1cm of clearance
                        colres = hppfcl.CollisionResult()
                        hppfcl.collide(
                            gobj_i.geometry, M1, gobj_j.geometry, M2, colreq, colres
                        )
                        if not colres.isCollision():
                            num_col_pairs += 1
                            col_pair = pin.CollisionPair(i, j)
                            geom_model.addCollisionPair(col_pair)


def addFloor(geom_model: pin.GeometryModel, visual_model: pin.GeometryModel):
    floor_collision_shape = hppfcl.Halfspace(0, 0, 1, 0)
    M = pin.SE3.Identity()
    floor_collision_object = pin.GeometryObject("floor", 0, 0, M, floor_collision_shape)
    floor_collision_object.meshColor = np.array([0.5, 0.5, 0.5, 0.5])
    geom_model.addGeometryObject(floor_collision_object)

    h = 0.01
    floor_visual_shape = hppfcl.Box(20, 20, h)
    Mvis = pin.SE3.Identity()
    Mvis.translation = np.array([0.0, 0.0, -h / 2])
    floor_visual_object = pin.GeometryObject("floor", 0, 0, Mvis, floor_visual_shape)
    floor_visual_object.meshColor = np.array([0.5, 0.5, 0.5, 0.4])
    visual_model.addGeometryObject(floor_visual_object)


def addMaterialAndCompliance(geom_model, material: str, compliance: float):
    for gobj in geom_model.geometryObjects:
        if material == "ice":
            gobj.physicsMaterial.materialType = pin.PhysicsMaterialType.ICE
        elif material == "plastic":
            gobj.physicsMaterial.materialType = pin.PhysicsMaterialType.PLASTIC
        elif material == "wood":
            gobj.physicsMaterial.materialType = pin.PhysicsMaterialType.WOOD
        elif material == "metal":
            gobj.physicsMaterial.materialType = pin.PhysicsMaterialType.METAL
        elif material == "concrete":
            gobj.physicsMaterial.materialType = pin.PhysicsMaterialType.CONCRETE
        else:
            raise Exception("Material unknown.")

        # Compliance
        gobj.physicsMaterial.compliance = compliance


def printSimulationPerfStats(step_timings: np.ndarray):
    print("============================================")
    print("SIMULATION")
    print("Time elapsed: ", np.sum(step_timings))
    print(
        "Mean timings time step: {mean:.2f} +/- {std:.2f} microseconds".format(
            mean=np.mean(step_timings) * 1e6, std=np.std(step_timings) * 1e6
        )
    )
    print(
        "Steps frequency: {freq:.2f} kHz".format(
            freq=(step_timings.size) / np.sum(step_timings) * 1e-3
        )
    )
    print("============================================")


def runMujocoXML(model_path: str, args: SimulationArgs):
    if not mujoco_imported:
        print("Can't run, need to install mujoco")
    if model_path.endswith(".xml"):
        m = mujoco.MjModel.from_xml_path(model_path)
    else:
        m = mlr(model_path)
    m.opt.cone = 1  # Elliptic
    m.opt.solver = 2  # Newton
    m.opt.timestep = args.dt
    m.opt.iterations = args.maxit
    m.opt.tolerance = args.tol
    m.opt.ls_iterations = 50
    m.opt.ls_tolerance = 1e-2
    d = mujoco.MjData(m)
    d.qpos = m.qpos0
    q0 = d.qpos.copy()
    v0 = d.qvel.copy()
    a0 = d.qacc.copy()

    print(f"{q0=}")
    print(f"{v0=}")

    step_timings = np.zeros(args.horizon)
    for t in range(args.horizon):
        start_time = time.time()
        mujoco.mj_step(m, d)
        end_time = time.time()
        step_timings[t] = end_time - start_time
    printSimulationPerfStats(step_timings)

    d.qpos = q0
    d.qvel = v0
    d.qacc = a0
    show_ui = False
    display_contacts = False
    with mujoco.viewer.launch_passive(
        m, d, show_left_ui=show_ui, show_right_ui=show_ui
    ) as viewer:
        input("[Press enter to display trajectory]")
        while True:
            d.qpos = q0
            d.qvel = v0
            d.qacc = a0
            for t in range(args.horizon):
                step_start = time.time()
                # mj_step can be replaced with code that also evaluates
                # a policy and applies a control signal before stepping the physics.
                mujoco.mj_step(m, d)

                # Example modification of a viewer option: toggle contact points every two seconds.
                if display_contacts:
                    with viewer.lock():
                        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONSTRAINT] = (
                            1  # int(d.time % 2)
                        )
                        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = (
                            1  # int(d.time % 2)
                        )
                        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = (
                            1  # int(d.time % 2)
                        )

                # Pick up changes to the physics state, apply perturbations, update options from GUI.
                viewer.sync()

                # Rudimentary time keeping, will drift relative to wall clock.
                time_until_next_step = m.opt.timestep - (time.time() - step_start)
                if args.debug:
                    print(d.qpos)
                    input(f"==== TIMESTEP {t} ====")
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)


def createVisualizer(
    model: pin.GeometryModel,
    geom_model: pin.GeometryModel,
    visual_model: pin.GeometryModel,
):
    viewer = meshcat.Visualizer(zmq_url="tcp://127.0.0.1:6000")
    viewer.delete()
    for obj in visual_model.geometryObjects:
        if obj.name != "floor":
            color = np.random.rand(4)
            color[3] = 1.0
            obj.meshColor = color
    vizer: MeshcatVisualizer = MeshcatVisualizer(model, geom_model, visual_model)
    vizer.initViewer(viewer=viewer, open=False, loadModel=True)
    return vizer, viewer
