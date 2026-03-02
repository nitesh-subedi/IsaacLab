from pxr import UsdPhysics, PhysxSchema
import omni.usd

stage = omni.usd.get_context().get_stage()

# The collision mesh is typically a child of the deformable prim
# e.g. /World/MyDeformable/collision_mesh
collision_mesh_prim = stage.GetPrimAtPath("/World/apple/collision_mesh")

# ✅ Correct: PhysxSchema.PhysxCollisionAPI (NOT UsdPhysics.PhysxCollisionAPI)
if not collision_mesh_prim.HasAPI(PhysxSchema.PhysxCollisionAPI):
    physx_collision_api = PhysxSchema.PhysxCollisionAPI.Apply(collision_mesh_prim)
else:
    physx_collision_api = PhysxSchema.PhysxCollisionAPI(collision_mesh_prim)

# Set contact and rest offsets (in meters)
physx_collision_api.GetContactOffsetAttr().Set(0.001)   # 1mm
physx_collision_api.GetRestOffsetAttr().Set(0.0005)     # 0.5mm


# Change mass properties of a deformable prim (e.g. for better stability or to simulate different materials)
from pxr import UsdPhysics

stage = omni.usd.get_context().get_stage()
prim = stage.GetPrimAtPath("/World/Sphere")

mass_api = UsdPhysics.MassAPI.Apply(prim)
mass_api.CreateDensityAttr().Set(0.0)  # or clear it if your workflow supports it
mass_api.CreateMassAttr().Set(0.2)

# Surface Deformables
from pxr import Usd, UsdGeom, Vt
import omni.usd
import numpy as np

stage = omni.usd.get_context().get_stage()
mesh_prim = stage.GetPrimAtPath("/root/stalk5_leaf6/plant_039")
mesh = UsdGeom.Mesh(mesh_prim)

# Check face vertex counts
face_counts = mesh.GetFaceVertexCountsAttr().Get()
print("Face vertex counts (unique):", set(face_counts))
# If you see anything other than 3, the mesh has quads/ngons

# Triangulate: split quads into triangles
face_vertices = list(mesh.GetFaceVertexIndicesAttr().Get())
new_face_counts = []
new_face_vertices = []

idx = 0
for count in face_counts:
    if count == 3:
        new_face_counts.append(3)
        new_face_vertices.extend(face_vertices[idx:idx+3])
    elif count == 4:
        # Split quad into 2 triangles
        v = face_vertices[idx:idx+4]
        new_face_counts.extend([3, 3])
        new_face_vertices.extend([v[0], v[1], v[2]])  # tri 1
        new_face_vertices.extend([v[0], v[2], v[3]])  # tri 2
    idx += count

mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray(new_face_counts))
mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray(new_face_vertices))
print(f"Triangulated: {len(new_face_counts)} triangles")


from pxr import PhysxSchema, Usd
import omni.usd

stage = omni.usd.get_context().get_stage()
leaf = stage.GetPrimAtPath("/root/stalk5_leaf6")

# Remove volume deformable APIs from parent — these cause the tetmesh error
schemas_to_remove = [
    "PhysxAutoDeformableBodyAPI",
    "PhysxAutoDeformableMeshSimplificationAPI",
    "OmniPhysicsDeformableBodyAPI",
    "OmniPhysicsBodyAPI",
]
for schema in schemas_to_remove:
    if schema in leaf.GetAppliedSchemas():
        leaf.RemoveAPI(Usd.SchemaRegistry.GetTypeFromName(schema))
        print(f"Removed: {schema}")
        
        
        
        
from isaacsim.core.experimental.prims import DeformablePrim
from isaacsim.core.experimental.materials import SurfaceDeformableMaterial

# Point to the XFORM parent, not the mesh itself
# Isaac Sim will find the child mesh automatically
deformable = DeformablePrim(
    paths="/root/stalk5_leaf6",       # ← Xform, not plant_041
    deformable_type="surface",        # ← explicitly surface
)

# Apply your material
material = SurfaceDeformableMaterial("/root/Surface_material")




import omni.usd
from omni.physx.scripts import deformableUtils
from omni.physx import get_physx_cooking_interface
from pxr import UsdGeom, UsdShade, Sdf

from isaacsim.core.experimental.materials import VolumeDeformableMaterial
from isaacsim.core.experimental.materials import SurfaceDeformableMaterial  # <-- NEW

stage = omni.usd.get_context().get_stage()

# ---------------------------
# CONFIG (EDIT IF NEEDED)
# ---------------------------
ROOT = "/root"
GROUND = f"{ROOT}/GroundPlane"  # change if needed

# Deformable params
VOLUME_RESOLUTION = 30
SOLVER_POS_ITERS = 255
REMESH_STALK_VOLUME = 0
REMESH_LEAF_SURFACE = 5  # <-- you asked: all surface remeshingResolution = 5

# Material paths (two separate materials)
VOL_MAT_PATH = "/PhysicsMaterialVolume"
SURF_MAT_PATH = "/PhysicsMaterialSurface"

# Material params (edit if you want different leaf vs stalk)
DENSITY = 1.0
YOUNGS = 2e30
POISSON = 0.45
MU_D = 0.25
MU_S = 0.5


# ---------------------------
# Basic helpers
# ---------------------------
def get_prim(path: str):
    prim = stage.GetPrimAtPath(path)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"Invalid prim: {path}")
    return prim

def prim_exists(path: str) -> bool:
    p = stage.GetPrimAtPath(path)
    return bool(p and p.IsValid())

def remove_prim_if_exists(path: str):
    if prim_exists(path):
        stage.RemovePrim(Sdf.Path(path))

def cleanup_attachment(root_path: str):
    remove_prim_if_exists(f"{root_path}/attachment")
    remove_prim_if_exists(f"{root_path}/attachment_02")

def cleanup_physx_visualization():
    remove_prim_if_exists("/PhysxProxiesVisualization")

def set_int_attr(prim, name: str, value: int):
    prim.CreateAttribute(name, Sdf.ValueTypeNames.Int).Set(int(value))

def set_cooking_source_mesh(root_prim, cook_src_path: str):
    rel = root_prim.CreateRelationship("physxDeformableBody:cookingSourceMesh")
    rel.SetTargets([Sdf.Path(cook_src_path)])

def bind_material(usd_mat: UsdShade.Material, paths: list[str]):
    for p in paths:
        if not prim_exists(p):
            continue
        prim = stage.GetPrimAtPath(p)
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(usd_mat)

def find_first_mesh_under(root_path: str) -> str:
    """Return path of the first UsdGeom.Mesh found under root_path (DFS)."""
    root = get_prim(root_path)
    stack = list(root.GetChildren())
    while stack:
        prim = stack.pop()
        if prim.IsA(UsdGeom.Mesh):
            return prim.GetPath().pathString
        stack.extend(list(prim.GetChildren()))
    raise RuntimeError(f"No UsdGeom.Mesh found under {root_path}.")

# ---------------------------
# Discover stalks/leaves directly under /root
# ---------------------------
def discover_names_under_root(root_path: str):
    root = get_prim(root_path)

    root_stalk = f"{root_path}/stalk"
    child_stalks = [f"{root_path}/stalk{i}" for i in range(1, 7)]

    leaves = []
    for child in root.GetChildren():
        if not child.IsA(UsdGeom.Xform):
            continue
        name = child.GetName()
        if name.startswith("stalk") and "_leaf" in name:
            leaves.append(child.GetPath().pathString)

    root_stalk = root_stalk if prim_exists(root_stalk) else None
    child_stalks = [p for p in child_stalks if prim_exists(p)]
    leaves = sorted([p for p in leaves if prim_exists(p)])
    return root_stalk, child_stalks, leaves

def stalk_for_leaf(leaf_path: str) -> str | None:
    leaf_name = leaf_path.split("/")[-1]  # stalk3_leaf4
    if "_leaf" not in leaf_name:
        return None
    stalk_name = leaf_name.split("_leaf")[0]  # stalk3
    stalk_path = "/".join(leaf_path.split("/")[:-1] + [stalk_name])
    return stalk_path if prim_exists(stalk_path) else None

# ---------------------------
# Surface hierarchy function (handle naming differences across builds)
# ---------------------------
def get_surface_hierarchy_fn():
    if hasattr(deformableUtils, "create_auto_surface_deformable_hierarchy"):
        return deformableUtils.create_auto_surface_deformable_hierarchy
    candidates = [
        x for x in dir(deformableUtils)
        if "surface" in x.lower() and "hierarchy" in x.lower() and callable(getattr(deformableUtils, x))
    ]
    if candidates:
        return getattr(deformableUtils, candidates[0])
    raise RuntimeError("No surface deformable hierarchy creator found in deformableUtils.")

CREATE_SURFACE_HIERARCHY = get_surface_hierarchy_fn()

# ---------------------------
# Deformable creation
# ---------------------------
def make_volume_deformable(root_path: str, remesh: int):
    sim_path = f"{root_path}/simulation_mesh"
    col_path = f"{root_path}/collision_mesh"
    cook_src = find_first_mesh_under(root_path)

    ok = deformableUtils.create_auto_volume_deformable_hierarchy(
        stage=stage,
        root_prim_path=root_path,
        simulation_tetmesh_path=sim_path,
        collision_tetmesh_path=col_path,
        cooking_src_mesh_path=cook_src,
        simulation_hex_mesh_enabled=True,
        cooking_src_simplification_enabled=True,
        set_visibility_with_guide_purpose=True,
    )
    if not ok:
        raise RuntimeError(f"create_auto_volume_deformable_hierarchy failed for {root_path}")

    root_prim = get_prim(root_path)
    set_int_attr(root_prim, "physxDeformableBody:resolution", VOLUME_RESOLUTION)
    set_int_attr(root_prim, "physxDeformableBody:solverPositionIterationCount", SOLVER_POS_ITERS)
    set_int_attr(root_prim, "physxDeformableBody:remeshingResolution", remesh)
    set_cooking_source_mesh(root_prim, cook_src)

    get_physx_cooking_interface().cook_auto_deformable_body(root_path)
    return sim_path, col_path, cook_src

import inspect

def make_surface_deformable(root_path: str, remesh: int):
    sim_path = f"{root_path}/simulation_mesh"
    col_path = f"{root_path}/collision_mesh"  # keep for binding if it gets created
    cook_src = find_first_mesh_under(root_path)

    fn = CREATE_SURFACE_HIERARCHY
    params = set(inspect.signature(fn).parameters.keys())

    # Build kwargs using only what this Isaac build supports
    kwargs = {}
    if "stage" in params:
        kwargs["stage"] = stage
    if "root_prim_path" in params:
        kwargs["root_prim_path"] = root_path

    # simulation mesh arg name differs across builds
    if "simulation_trimesh_path" in params:
        kwargs["simulation_trimesh_path"] = sim_path
    elif "simulation_mesh_path" in params:
        kwargs["simulation_mesh_path"] = sim_path
    else:
        raise RuntimeError(f"{fn.__name__} has no supported simulation mesh path argument. "
                           f"Signature params: {sorted(params)}")

    # collision args: many builds do NOT take any collision path for surface deformables
    if "collision_trimesh_path" in params:
        kwargs["collision_trimesh_path"] = col_path
    elif "collision_mesh_path" in params:
        kwargs["collision_mesh_path"] = col_path
    # else: omit collision args entirely

    if "cooking_src_mesh_path" in params:
        kwargs["cooking_src_mesh_path"] = cook_src
    if "cooking_src_simplification_enabled" in params:
        kwargs["cooking_src_simplification_enabled"] = True
    if "set_visibility_with_guide_purpose" in params:
        kwargs["set_visibility_with_guide_purpose"] = True

    ok = fn(**kwargs)
    if not ok:
        raise RuntimeError(f"{fn.__name__} failed for {root_path}")

    root_prim = get_prim(root_path)
    set_int_attr(root_prim, "physxDeformableBody:solverPositionIterationCount", SOLVER_POS_ITERS)
    set_int_attr(root_prim, "physxDeformableBody:remeshingResolution", remesh)  # <- 5
    set_cooking_source_mesh(root_prim, cook_src)

    get_physx_cooking_interface().cook_auto_deformable_body(root_path)
    return sim_path, col_path, cook_src

# ---------------------------
# Attachments
# ---------------------------
def make_attachment(attachment_path: str, deformable_actor: str, target_actor: str, overlap: float):
    deformableUtils.create_auto_deformable_attachment(stage, attachment_path, deformable_actor, target_actor)
    prim = get_prim(attachment_path)
    p = "physxAutoDeformableAttachment:"
    prim.GetAttribute(p + "enableDeformableVertexAttachments").Set(True)
    prim.GetAttribute(p + "deformableVertexOverlapOffset").Set(float(overlap))


# ===========================
# MAIN
# ===========================
root_stalk, child_stalks, leaves = discover_names_under_root(ROOT)
if root_stalk is None:
    raise RuntimeError(f"Expected {ROOT}/stalk. Update ROOT or the prim name.")

print("\n=== DISCOVERED ===")
print("ROOT_STALK:", root_stalk)
print("CHILD_STALKS:", child_stalks)
print("LEAVES:")
for l in leaves:
    print("  ", l)

# Cleanup
cleanup_physx_visualization()
cleanup_attachment(root_stalk)
for s in child_stalks:
    cleanup_attachment(s)
for l in leaves:
    cleanup_attachment(l)

# ---------------------------
# Create BOTH materials
# ---------------------------
# Volume deformable material for stalks
VolumeDeformableMaterial(
    paths=VOL_MAT_PATH,
    densities=DENSITY,
    youngs_moduli=YOUNGS,
    poissons_ratios=POISSON,
    dynamic_frictions=MU_D,
    static_frictions=MU_S,
)
usd_mat_vol = UsdShade.Material(get_prim(VOL_MAT_PATH))

# Surface deformable material for leaves
SurfaceDeformableMaterial(
    paths=SURF_MAT_PATH,
    densities=DENSITY,
    youngs_moduli=YOUNGS,
    poissons_ratios=POISSON,
    dynamic_frictions=MU_D,
    static_frictions=MU_S,
)
usd_mat_surf = UsdShade.Material(get_prim(SURF_MAT_PATH))

# ---------------------------
# Cook deformables + bind correct material
# ---------------------------
print("\n=== COOKING DEFORMABLES ===")

stalk_infos = []
for s in [root_stalk] + child_stalks:
    sim, col, cook = make_volume_deformable(s, remesh=REMESH_STALK_VOLUME)
    bind_material(usd_mat_vol, [s, cook, sim, col])   # <-- VOLUME mat
    stalk_infos.append((s, sim, col, cook))
    print("VOLUME:", s, "| cook_src:", cook)

leaf_infos = []
for l in leaves:
    sim, col, cook = make_surface_deformable(l, remesh=REMESH_LEAF_SURFACE)  # <-- remesh=5
    bind_material(usd_mat_surf, [l, cook, sim, col])  # <-- SURFACE mat
    leaf_infos.append((l, sim, col, cook))
    print("SURFACE:", l, "| cook_src:", cook, "| remesh:", REMESH_LEAF_SURFACE)

# ---------------------------
# Attachments (your requested structure)
# ---------------------------
print("\n=== ATTACHMENTS ===")

# 1) /root/stalk -> ground
if prim_exists(GROUND):
    make_attachment(f"{root_stalk}/attachment", root_stalk, GROUND, overlap=0.3)
    print("ATTACH:", root_stalk, "->", GROUND)
else:
    print("WARNING: Ground not found at", GROUND)

# 2) stalk1..6 -> stalk
for s in child_stalks:
    make_attachment(f"{s}/attachment", s, root_stalk, overlap=0.2)
    print("ATTACH:", s, "->", root_stalk)

# 3) leaf -> respective stalkX
for (leaf, _, _, _) in leaf_infos:
    target = stalk_for_leaf(leaf) or root_stalk
    make_attachment(f"{leaf}/attachment", leaf, target, overlap=0.1)
    print("ATTACH:", leaf, "->", target)

print("\nDone.")



def set_physx_offsets_on_mesh(mesh_path: str, contact: float, rest: float):
    prim = stage.GetPrimAtPath(mesh_path)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"Invalid prim: {mesh_path}")

    # Apply PhysxCollisionAPI to the mesh prim
    if not prim.HasAPI(PhysxSchema.PhysxCollisionAPI):
        api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    else:
        api = PhysxSchema.PhysxCollisionAPI(prim)

    api.GetContactOffsetAttr().Set(float(contact))
    api.GetRestOffsetAttr().Set(float(rest))
    
    
CONTACT_OFFSET = 0.001
REST_OFFSET = 0.0005

for (leaf, leaf_sim, _leaf_col, _leaf_cook) in leaf_infos:
    # For surface deformables, leaf_sim is the collision geometry
    set_physx_offsets_on_mesh(leaf_sim, CONTACT_OFFSET, REST_OFFSET)





import omni.usd
import inspect
from omni.physx.scripts import deformableUtils
from omni.physx import get_physx_cooking_interface
from pxr import UsdGeom, UsdShade, Sdf, PhysxSchema

from isaacsim.core.experimental.materials import VolumeDeformableMaterial
from isaacsim.core.experimental.materials import SurfaceDeformableMaterial

stage = omni.usd.get_context().get_stage()

# ---------------------------
# PATHS
# ---------------------------
ROOT = "/root"
GROUND = f"{ROOT}/GroundPlane"

ROOT_STALK = f"{ROOT}/stalk"
CHILD_STALKS = [f"{ROOT}/stalk{i}" for i in range(1, 7)]
LEAVES = [f"{ROOT}/stalk{i}_leaf{j}" for i in range(1, 7) for j in range(1, 7)]

# ---------------------------
# DEFORMABLE PARAMS
# ---------------------------
VOLUME_RESOLUTION = 30
SOLVER_POS_ITERS = 255
REMESH_STALK_VOLUME = 0
REMESH_LEAF_SURFACE = 5

# ---------------------------
# COLLISION OFFSETS (meters)
# ---------------------------
CONTACT_OFFSET = 0.001   # 1 mm
REST_OFFSET = 0.0005     # 0.5 mm

# ---------------------------
# MATERIALS
# ---------------------------
VOL_MAT_PATH = "/PhysicsMaterialVolume"
SURF_MAT_PATH = "/PhysicsMaterialSurface"

# Volume material params
DENSITY = 1.0
YOUNGS = 2e30
POISSON = 0.45
MU_D = 0.25
MU_S = 0.5

# Surface material params (from your UI)
SURF_DENSITY = 1.0
SURF_YOUNGS = 2000.0
SURF_POISSON = 0.45
SURF_MU_D = 0.25
SURF_MU_S = 0.5
SURF_BEND = 2.0
SURF_SHEAR = 2.0
SURF_STRETCH = 2.0
SURF_THICK = 0.001

# ---------------------------
# Helpers
# ---------------------------
def prim_exists(path: str) -> bool:
    p = stage.GetPrimAtPath(path)
    return bool(p and p.IsValid())

def get_prim(path: str):
    prim = stage.GetPrimAtPath(path)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"Invalid prim: {path}")
    return prim

def remove_prim_if_exists(path: str):
    if prim_exists(path):
        stage.RemovePrim(Sdf.Path(path))

def cleanup_attachment(root_path: str):
    remove_prim_if_exists(f"{root_path}/attachment")
    remove_prim_if_exists(f"{root_path}/attachment_02")

def set_int_attr(prim, name: str, value: int):
    prim.CreateAttribute(name, Sdf.ValueTypeNames.Int).Set(int(value))

def set_float_attr(prim, name: str, value: float):
    prim.CreateAttribute(name, Sdf.ValueTypeNames.Float).Set(float(value))

def set_cooking_source_mesh(root_prim, cook_src_path: str):
    rel = root_prim.CreateRelationship("physxDeformableBody:cookingSourceMesh")
    rel.SetTargets([Sdf.Path(cook_src_path)])

def bind_material(usd_mat: UsdShade.Material, paths: list[str]):
    for p in paths:
        if not prim_exists(p):
            continue
        prim = stage.GetPrimAtPath(p)
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(usd_mat)

def find_first_mesh_under(root_path: str) -> str:
    root = get_prim(root_path)
    stack = list(root.GetChildren())
    while stack:
        prim = stack.pop()
        if prim.IsA(UsdGeom.Mesh):
            return prim.GetPath().pathString
        stack.extend(list(prim.GetChildren()))
    raise RuntimeError(f"No UsdGeom.Mesh found under {root_path}")

def make_attachment(attachment_path: str, deformable_actor: str, target_actor: str, overlap: float):
    deformableUtils.create_auto_deformable_attachment(stage, attachment_path, deformable_actor, target_actor)
    prim = get_prim(attachment_path)
    p = "physxAutoDeformableAttachment:"
    prim.GetAttribute(p + "enableDeformableVertexAttachments").Set(True)
    prim.GetAttribute(p + "deformableVertexOverlapOffset").Set(float(overlap))

def set_physx_offsets_on_prim(prim_path: str, contact: float, rest: float):
    prim = get_prim(prim_path)
    if not prim.HasAPI(PhysxSchema.PhysxCollisionAPI):
        api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    else:
        api = PhysxSchema.PhysxCollisionAPI(prim)
    api.GetContactOffsetAttr().Set(float(contact))
    api.GetRestOffsetAttr().Set(float(rest))

def set_offsets_recursive(root_path: str, contact: float, rest: float):
    root = get_prim(root_path)
    stack = [root]
    while stack:
        p = stack.pop()
        if p.IsA(UsdGeom.Mesh):
            if not p.HasAPI(PhysxSchema.PhysxCollisionAPI):
                api = PhysxSchema.PhysxCollisionAPI.Apply(p)
            else:
                api = PhysxSchema.PhysxCollisionAPI(p)
            api.GetContactOffsetAttr().Set(float(contact))
            api.GetRestOffsetAttr().Set(float(rest))
        stack.extend(list(p.GetChildren()))

def set_mass(root_path: str, value: float):
    prim = get_prim(root_path)
    set_float_attr(prim, "omniphysics:mass", float(value))

# surface hierarchy fn (varies by build)
def get_surface_hierarchy_fn():
    if hasattr(deformableUtils, "create_auto_surface_deformable_hierarchy"):
        return deformableUtils.create_auto_surface_deformable_hierarchy
    candidates = [
        x for x in dir(deformableUtils)
        if "surface" in x.lower() and "hierarchy" in x.lower() and callable(getattr(deformableUtils, x))
    ]
    if candidates:
        return getattr(deformableUtils, candidates[0])
    raise RuntimeError("No surface deformable hierarchy creator found in deformableUtils.")

CREATE_SURFACE_HIERARCHY = get_surface_hierarchy_fn()

# ---------------------------
# Deformable creation
# ---------------------------
def make_volume_deformable(root_path: str, remesh: int):
    sim_path = f"{root_path}/simulation_mesh"
    col_path = f"{root_path}/collision_mesh"
    cook_src = find_first_mesh_under(root_path)

    ok = deformableUtils.create_auto_volume_deformable_hierarchy(
        stage=stage,
        root_prim_path=root_path,
        simulation_tetmesh_path=sim_path,
        collision_tetmesh_path=col_path,
        cooking_src_mesh_path=cook_src,
        simulation_hex_mesh_enabled=True,
        cooking_src_simplification_enabled=True,
        set_visibility_with_guide_purpose=True,
    )
    if not ok:
        raise RuntimeError(f"create_auto_volume_deformable_hierarchy failed for {root_path}")

    root_prim = get_prim(root_path)
    set_int_attr(root_prim, "physxDeformableBody:resolution", VOLUME_RESOLUTION)
    set_int_attr(root_prim, "physxDeformableBody:solverPositionIterationCount", SOLVER_POS_ITERS)
    set_int_attr(root_prim, "physxDeformableBody:remeshingResolution", remesh)
    set_cooking_source_mesh(root_prim, cook_src)

    get_physx_cooking_interface().cook_auto_deformable_body(root_path)
    return sim_path, col_path, cook_src

def make_surface_deformable(root_path: str, remesh: int):
    sim_path = f"{root_path}/simulation_mesh"
    col_path = f"{root_path}/collision_mesh"  # may not exist for surface
    cook_src = find_first_mesh_under(root_path)

    fn = CREATE_SURFACE_HIERARCHY
    params = set(inspect.signature(fn).parameters.keys())

    kwargs = {}
    if "stage" in params: kwargs["stage"] = stage
    if "root_prim_path" in params: kwargs["root_prim_path"] = root_path

    if "simulation_trimesh_path" in params:
        kwargs["simulation_trimesh_path"] = sim_path
    elif "simulation_mesh_path" in params:
        kwargs["simulation_mesh_path"] = sim_path
    else:
        raise RuntimeError(f"{fn.__name__} missing simulation mesh arg; params={sorted(params)}")

    if "collision_trimesh_path" in params:
        kwargs["collision_trimesh_path"] = col_path
    elif "collision_mesh_path" in params:
        kwargs["collision_mesh_path"] = col_path

    if "cooking_src_mesh_path" in params: kwargs["cooking_src_mesh_path"] = cook_src
    if "cooking_src_simplification_enabled" in params: kwargs["cooking_src_simplification_enabled"] = True
    if "set_visibility_with_guide_purpose" in params: kwargs["set_visibility_with_guide_purpose"] = True

    ok = fn(**kwargs)
    if not ok:
        raise RuntimeError(f"{fn.__name__} failed for {root_path}")

    root_prim = get_prim(root_path)
    set_int_attr(root_prim, "physxDeformableBody:solverPositionIterationCount", SOLVER_POS_ITERS)
    set_int_attr(root_prim, "physxDeformableBody:remeshingResolution", remesh)  # <- 5
    set_cooking_source_mesh(root_prim, cook_src)

    get_physx_cooking_interface().cook_auto_deformable_body(root_path)
    return sim_path, col_path, cook_src


# ===========================
# MAIN (ALL STALKS + LEAVES)
# ===========================
ALL_STALKS = [ROOT_STALK] + CHILD_STALKS
ALL_PARTS = ALL_STALKS + LEAVES + [GROUND]

missing = [p for p in ALL_PARTS if not prim_exists(p)]
if missing:
    # allow missing leaves if some don't exist, but fail on stalks
    missing_stalks = [p for p in ALL_STALKS if not prim_exists(p)]
    if missing_stalks:
        raise RuntimeError(f"Missing stalk prims: {missing_stalks}")
    print("Warning: some leaves/ground are missing, will skip those:\n", "\n".join(missing))

# Cleanup attachments for all parts that exist
for s in ALL_STALKS:
    if prim_exists(s):
        cleanup_attachment(s)
for l in LEAVES:
    if prim_exists(l):
        cleanup_attachment(l)

# Create materials
VolumeDeformableMaterial(
    paths=VOL_MAT_PATH,
    densities=DENSITY,
    youngs_moduli=YOUNGS,
    poissons_ratios=POISSON,
    dynamic_frictions=MU_D,
    static_frictions=MU_S,
)
usd_mat_vol = UsdShade.Material(get_prim(VOL_MAT_PATH))

SurfaceDeformableMaterial(
    paths=SURF_MAT_PATH,
    densities=SURF_DENSITY,
    youngs_moduli=SURF_YOUNGS,
    poissons_ratios=SURF_POISSON,
    dynamic_frictions=SURF_MU_D,
    static_frictions=SURF_MU_S,
)
surf_mat_prim = get_prim(SURF_MAT_PATH)
set_float_attr(surf_mat_prim, "omniphysics:surfaceBendStiffness", SURF_BEND)
set_float_attr(surf_mat_prim, "omniphysics:surfaceShearStiffness", SURF_SHEAR)
set_float_attr(surf_mat_prim, "omniphysics:surfaceStretchStiffness", SURF_STRETCH)
set_float_attr(surf_mat_prim, "omniphysics:surfaceThickness", SURF_THICK)
usd_mat_surf = UsdShade.Material(surf_mat_prim)

# Cook stalks (volume) + set masses + bind volume material
stalk_infos = []
for s in ALL_STALKS:
    if not prim_exists(s):
        continue
    sim, col, cook = make_volume_deformable(s, remesh=REMESH_STALK_VOLUME)
    bind_material(usd_mat_vol, [s, cook, sim, col])
    stalk_infos.append((s, sim, col, cook))

# Mass: main stalk = 5.0, other stalks = 1.0
set_mass(ROOT_STALK, 5.0)
for s in CHILD_STALKS:
    if prim_exists(s):
        set_mass(s, 1.0)

# Cook leaves (surface) + bind surface material
leaf_infos = []
for leaf in LEAVES:
    if not prim_exists(leaf):
        continue
    sim, col, cook = make_surface_deformable(leaf, remesh=REMESH_LEAF_SURFACE)
    bind_material(usd_mat_surf, [leaf, cook, sim, col])
    leaf_infos.append((leaf, sim, col, cook))

# Attachments:
# - /root/stalk -> ground
if prim_exists(GROUND):
    make_attachment(f"{ROOT_STALK}/attachment", ROOT_STALK, GROUND, overlap=0.01)

# - /root/stalk1..6 -> /root/stalk
for s in CHILD_STALKS:
    if prim_exists(s) and prim_exists(ROOT_STALK):
        make_attachment(f"{s}/attachment", s, ROOT_STALK, overlap=0.001)

# - /root/stalkX_leafY -> /root/stalkX
for (leaf, _, _, _) in leaf_infos:
    leaf_name = leaf.split("/")[-1]         # stalk3_leaf2
    stalk_name = leaf_name.split("_leaf")[0]  # stalk3
    target = f"{ROOT}/{stalk_name}"
    if not prim_exists(target):
        target = ROOT_STALK
    make_attachment(f"{leaf}/attachment", leaf, target, overlap=0.001)

# Collision offsets:
# - Volume deformables: collision_mesh prim if present, else sim subtree
for (s, sim, col, _cook) in stalk_infos:
    if prim_exists(col):
        set_physx_offsets_on_prim(col, CONTACT_OFFSET, REST_OFFSET)
    else:
        set_offsets_recursive(sim, CONTACT_OFFSET, REST_OFFSET)

# - Surface deformables: collision is sim mesh (apply recursively)
for (_leaf, leaf_sim, _leaf_col, _leaf_cook) in leaf_infos:
    set_offsets_recursive(leaf_sim, CONTACT_OFFSET, REST_OFFSET)

print("Done ALL stalks + leaves.")
print("Stalks cooked:", [x[0] for x in stalk_infos])
print("Leaves cooked:", [x[0] for x in leaf_infos])
print("Masses: stalk=5.0, stalk1-6=1.0")
print("Offsets:", "contact=", CONTACT_OFFSET, "rest=", REST_OFFSET)
print("Surface remesh:", REMESH_LEAF_SURFACE)



from pxr import UsdPhysics, Sdf

LEAF_GROUP_PATH = "/CollisionGroup_Leaves"  # any path you want

def ensure_collision_group_no_self(group_path: str):
    # Define a collision group prim (same as GUI: Create > Physics > Collision Group)
    group = UsdPhysics.CollisionGroup.Define(stage, Sdf.Path(group_path))

    # Filter against itself => members of this group do NOT collide with each other
    group.CreateFilteredGroupsRel().SetTargets([Sdf.Path(group_path)])
    return group

def add_prim_to_collision_group(group_path: str, prim_path: str):
    """
    Adds a prim to the group by writing the collection relationship directly:
      collection:colliders:includes -> [prim_path]
    """
    group_prim = stage.GetPrimAtPath(group_path)
    if not group_prim or not group_prim.IsValid():
        raise RuntimeError(f"CollisionGroup does not exist: {group_path}")

    # This is the key: author the collection includes relationship directly
    includes_rel = group_prim.CreateRelationship("collection:colliders:includes", custom=False)
    existing = includes_rel.GetTargets() or []
    target = Sdf.Path(prim_path)

    if target not in existing:
        includes_rel.SetTargets(existing + [target])

def disable_leaf_leaf_collisions(leaf_infos, group_path=LEAF_GROUP_PATH):
    ensure_collision_group_no_self(group_path)

    # Add all leaf collision prims into the group.
    # For surface deformables, collision is typically the SIM mesh subtree.
    for (leaf, leaf_sim, _leaf_col, _leaf_cook) in leaf_infos:
        if not prim_exists(leaf_sim):
            continue

        # Add the sim root; PhysX will treat meshes beneath as colliders.
        # If you want to be extra-safe, add each Mesh child instead (see below).
        add_prim_to_collision_group(group_path, leaf_sim)

    print(f"[CollisionGroup] Leaf-leaf collisions disabled via {group_path}")


disable_leaf_leaf_collisions(leaf_infos)

