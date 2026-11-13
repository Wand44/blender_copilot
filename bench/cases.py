"""
Blender copilot benchmark cases.

Each case:
  id        unique name
  tier      easy | medium | hard | fix (fix = broken code / legacy API)
  prompt    what you send to the model
  setup     code run in an EMPTY scene BEFORE the model's code (may be "")
  reference known-good answer (also used to sanity-check the checks themselves)
  check     code run AFTER the model's code; use assert. Passing = no exception.

Runner contract (run each case in a fresh process):
  blender -b --factory-startup --python runner.py
  1. bpy.ops.wm.read_factory_settings(use_empty=True)   # no default cube/camera/light
  2. exec(setup)
  3. exec(model_code)      # strip ``` fences first; catch exceptions -> "crashed"
  4. exec(check)           # AssertionError -> "ran but wrong"
Score per case: crashed=0, ran-but-wrong=0 (track separately), passed=1.
Validate the harness first: every `reference` must pass its own `check`.
"""

CASES = [
    # ---------------------------------------------------------------- EASY
    dict(
        id="cube_origin", tier="easy",
        prompt="Make a blender script that when ran creates a cube in the middle of the scene",
        setup="",
        reference="""import bpy
bpy.ops.mesh.primitive_cube_add(location=(0, 0, 0))
""",
        check="""
objs = [o for o in bpy.data.objects if o.type == 'MESH']
assert len(objs) == 1, len(objs)
o = objs[0]
assert len(o.data.vertices) == 8
assert o.location.length < 1e-4
""",
    ),
    dict(
        id="sphere_at_position", tier="easy",
        prompt="Write a Blender script that adds a UV sphere with radius 0.5 at x=2, y=0, z=0.",
        setup="",
        reference="""import bpy
bpy.ops.mesh.primitive_uv_sphere_add(radius=0.5, location=(2, 0, 0))
""",
        check="""
objs = [o for o in bpy.data.objects if o.type == 'MESH']
assert len(objs) == 1
o = objs[0]
assert abs(o.location.x - 2) < 1e-4 and abs(o.location.y) < 1e-4 and abs(o.location.z) < 1e-4
assert abs(max(o.dimensions) - 1.0) < 0.01   # diameter 1.0 (radius 0.5 baked in or scaled)
""",
    ),
    dict(
        id="delete_all", tier="easy",
        prompt="Write a Blender script that deletes every object in the scene.",
        setup="""import bpy
for i in range(4):
    bpy.ops.mesh.primitive_cube_add(location=(i*3, 0, 0))
bpy.ops.object.light_add(type='POINT')
""",
        reference="""import bpy
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()
""",
        check="""
assert len(bpy.context.scene.objects) == 0
""",
    ),
    dict(
        id="rename_active", tier="easy",
        prompt="Write a Blender script that renames the active object to 'Hero'.",
        setup="""import bpy
bpy.ops.mesh.primitive_cube_add()
""",
        reference="""import bpy
bpy.context.active_object.name = "Hero"
""",
        check="""
assert "Hero" in bpy.data.objects
""",
    ),
    dict(
        id="render_settings", tier="easy",
        prompt="Write a Blender script that sets the render resolution to 1920x1080 at 50% scale and the output format to PNG.",
        setup="",
        reference="""import bpy
r = bpy.context.scene.render
r.resolution_x = 1920
r.resolution_y = 1080
r.resolution_percentage = 50
r.image_settings.file_format = 'PNG'
""",
        check="""
r = bpy.context.scene.render
assert (r.resolution_x, r.resolution_y, r.resolution_percentage) == (1920, 1080, 50)
assert r.image_settings.file_format == 'PNG'
""",
    ),

    # -------------------------------------------------------------- MEDIUM
    dict(
        id="red_material", tier="medium",
        prompt="Write a Blender script that creates a pure red material called 'Red' and assigns it to the active object.",
        setup="""import bpy
bpy.ops.mesh.primitive_cube_add()
""",
        reference="""import bpy
mat = bpy.data.materials.new(name="Red")
mat.use_nodes = True
bsdf = mat.node_tree.nodes["Principled BSDF"]
bsdf.inputs["Base Color"].default_value = (1, 0, 0, 1)
mat.diffuse_color = (1, 0, 0, 1)
obj = bpy.context.active_object
obj.data.materials.append(mat)
""",
        check="""
obj = bpy.context.active_object
m = obj.active_material
assert m is not None and m.name.startswith("Red")
col = tuple(m.diffuse_color)
if m.use_nodes:
    b = next((n for n in m.node_tree.nodes if n.type == 'BSDF_PRINCIPLED'), None)
    if b: col = tuple(b.inputs["Base Color"].default_value)
assert col[0] > 0.95 and col[1] < 0.05 and col[2] < 0.05, col
""",
    ),
    dict(
        id="grid_of_cubes", tier="medium",
        prompt="Write a Blender script that creates a 5x5 grid of cubes in the XY plane, spaced 3 units apart.",
        setup="",
        reference="""import bpy
for x in range(5):
    for y in range(5):
        bpy.ops.mesh.primitive_cube_add(location=(x * 3, y * 3, 0))
""",
        check="""
objs = [o for o in bpy.data.objects if o.type == 'MESH']
assert len(objs) == 25, len(objs)
locs = {(round(o.location.x, 3), round(o.location.y, 3)) for o in objs}
assert len(locs) == 25
xs = sorted({l[0] for l in locs}); ys = sorted({l[1] for l in locs})
assert len(xs) == 5 and len(ys) == 5
assert all(abs((xs[i+1]-xs[i]) - 3) < 1e-3 for i in range(4))
assert all(abs((ys[i+1]-ys[i]) - 3) < 1e-3 for i in range(4))
""",
    ),
    dict(
        id="subsurf_smooth", tier="medium",
        prompt="Write a Blender script that adds a Subdivision Surface modifier with 2 viewport levels to the active object and shades it smooth.",
        setup="""import bpy
bpy.ops.mesh.primitive_cube_add()
""",
        reference="""import bpy
obj = bpy.context.active_object
mod = obj.modifiers.new(name="Subsurf", type='SUBSURF')
mod.levels = 2
bpy.ops.object.shade_smooth()
""",
        check="""
obj = bpy.context.active_object
mods = [m for m in obj.modifiers if m.type == 'SUBSURF']
assert len(mods) == 1 and mods[0].levels == 2
assert all(p.use_smooth for p in obj.data.polygons)
""",
    ),
    dict(
        id="point_light_data_api", tier="medium",
        prompt="Using bpy.data (not bpy.ops), create a point light named 'KeyLight' at (4, 1, 6) with 1000 W energy and link it to the scene.",
        setup="",
        reference="""import bpy
light_data = bpy.data.lights.new(name="KeyLight", type='POINT')
light_data.energy = 1000
light_obj = bpy.data.objects.new(name="KeyLight", object_data=light_data)
light_obj.location = (4, 1, 6)
bpy.context.scene.collection.objects.link(light_obj)
""",
        check="""
o = bpy.data.objects.get("KeyLight")
assert o is not None and o.type == 'LIGHT'
assert o.data.type == 'POINT' and abs(o.data.energy - 1000) < 1e-3
assert (o.location - __import__('mathutils').Vector((4, 1, 6))).length < 1e-4
assert o.name in bpy.context.scene.objects      # actually linked
""",
    ),
    dict(
        id="keyframe_z", tier="medium",
        prompt="Write a Blender script that animates the active object's Z location from 0 at frame 1 to 5 at frame 60.",
        setup="""import bpy
bpy.ops.mesh.primitive_cube_add()
""",
        reference="""import bpy
obj = bpy.context.active_object
obj.location.z = 0
obj.keyframe_insert(data_path="location", index=2, frame=1)
obj.location.z = 5
obj.keyframe_insert(data_path="location", index=2, frame=60)
""",
        # version-agnostic: sample the animation instead of poking at fcurves
        check="""
obj = bpy.context.active_object
sc = bpy.context.scene
sc.frame_set(1);  z1 = obj.location.z
sc.frame_set(60); z60 = obj.location.z
assert abs(z1) < 1e-3 and abs(z60 - 5) < 1e-3, (z1, z60)
""",
    ),
    dict(
        id="mesh_from_pydata", tier="medium",
        prompt="Write a Blender script that builds a single triangle mesh from raw vertex data (no primitive operators) and links it to the scene as an object named 'Tri'.",
        setup="",
        reference="""import bpy
mesh = bpy.data.meshes.new("Tri")
mesh.from_pydata([(0, 0, 0), (1, 0, 0), (0, 1, 0)], [], [(0, 1, 2)])
mesh.update()
obj = bpy.data.objects.new("Tri", mesh)
bpy.context.scene.collection.objects.link(obj)
""",
        check="""
o = bpy.data.objects.get("Tri")
assert o is not None and o.type == 'MESH'
assert len(o.data.vertices) == 3 and len(o.data.polygons) == 1
assert o.name in bpy.context.scene.objects
""",
    ),

    # ---------------------------------------------------------------- HARD
    dict(
        id="addon_operator_panel", tier="hard",
        prompt=(
            "Write a complete Blender addon script that, when run, registers: "
            "(1) an operator with bl_idname 'object.add_random_cubes' and an integer property 'count' (default 5) "
            "that adds that many cubes at random locations within -5..5 on each axis, and "
            "(2) a panel in the 3D Viewport sidebar (N-panel) under a tab called 'Tools' with a button that runs the operator. "
            "Include register() and unregister() and call register() at the end."
        ),
        setup="",
        reference="""import bpy
import random

class OBJECT_OT_add_random_cubes(bpy.types.Operator):
    bl_idname = "object.add_random_cubes"
    bl_label = "Add Random Cubes"
    bl_options = {'REGISTER', 'UNDO'}

    count: bpy.props.IntProperty(name="Count", default=5, min=1, max=100)

    def execute(self, context):
        for _ in range(self.count):
            loc = (random.uniform(-5, 5), random.uniform(-5, 5), random.uniform(-5, 5))
            bpy.ops.mesh.primitive_cube_add(location=loc)
        return {'FINISHED'}

class VIEW3D_PT_random_cubes(bpy.types.Panel):
    bl_label = "Random Cubes"
    bl_idname = "VIEW3D_PT_random_cubes"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Tools"

    def draw(self, context):
        self.layout.operator("object.add_random_cubes")

classes = (OBJECT_OT_add_random_cubes, VIEW3D_PT_random_cubes)

def register():
    for c in classes:
        bpy.utils.register_class(c)

def unregister():
    for c in reversed(classes):
        bpy.utils.unregister_class(c)

register()
""",
        check="""
assert hasattr(bpy.ops.object, "add_random_cubes")
before = len(bpy.data.objects)
bpy.ops.object.add_random_cubes(count=7)
assert len(bpy.data.objects) - before == 7
for o in list(bpy.data.objects)[before:]:
    assert all(-5.0001 <= c <= 5.0001 for c in o.location)
panels = [p for p in bpy.types.Panel.__subclasses__()
          if getattr(p, "bl_space_type", "") == 'VIEW_3D'
          and getattr(p, "bl_region_type", "") == 'UI'
          and getattr(p, "bl_category", "") == "Tools"]
assert panels, "no N-panel in 'Tools' tab"
""",
    ),
    dict(
        id="bmesh_wave_grid", tier="hard",
        prompt="Using bmesh, create a flat grid mesh of at least 20x20 vertices, then set each vertex's Z to sin(x) * 0.5. Put it in an object named 'Wave' linked to the scene.",
        setup="",
        reference="""import bpy, bmesh, math

mesh = bpy.data.meshes.new("Wave")
obj = bpy.data.objects.new("Wave", mesh)
bpy.context.scene.collection.objects.link(obj)

bm = bmesh.new()
bmesh.ops.create_grid(bm, x_segments=20, y_segments=20, size=5)
for v in bm.verts:
    v.co.z = math.sin(v.co.x) * 0.5
bm.to_mesh(mesh)
bm.free()
""",
        check="""
import math
o = bpy.data.objects.get("Wave")
assert o is not None and o.type == 'MESH' and o.name in bpy.context.scene.objects
vs = o.data.vertices
assert len(vs) >= 400, len(vs)
for v in vs:
    assert abs(v.co.z - math.sin(v.co.x) * 0.5) < 1e-3
assert max(v.co.z for v in vs) > 0.2          # not flat
""",
    ),
    dict(
        id="noise_shader_nodes", tier="hard",
        prompt="Write a Blender script that creates a material 'Noisy' where a Noise Texture node drives the Base Color of the Principled BSDF, and assigns it to the active object.",
        setup="""import bpy
bpy.ops.mesh.primitive_cube_add()
""",
        reference="""import bpy
mat = bpy.data.materials.new("Noisy")
mat.use_nodes = True
nodes, links = mat.node_tree.nodes, mat.node_tree.links
bsdf = nodes["Principled BSDF"]
noise = nodes.new("ShaderNodeTexNoise")
noise.location = (-300, 0)
links.new(noise.outputs["Color"], bsdf.inputs["Base Color"])
bpy.context.active_object.data.materials.append(mat)
""",
        check="""
m = bpy.context.active_object.active_material
assert m is not None and m.use_nodes
ok = any(l.from_node.type == 'TEX_NOISE'
         and l.to_node.type == 'BSDF_PRINCIPLED'
         and l.to_socket.name == 'Base Color'
         for l in m.node_tree.links)
assert ok, "no Noise -> Principled Base Color link"
""",
    ),
    dict(
        id="cycles_gpu_128", tier="hard",
        prompt="Write a Blender script that sets the render engine to Cycles, 128 samples, GPU compute, and enables the denoiser.",
        setup="",
        reference="""import bpy
s = bpy.context.scene
s.render.engine = 'CYCLES'
s.cycles.samples = 128
s.cycles.device = 'GPU'
s.cycles.use_denoising = True
""",
        check="""
s = bpy.context.scene
assert s.render.engine == 'CYCLES'
assert s.cycles.samples == 128
assert s.cycles.device == 'GPU'
assert s.cycles.use_denoising is True
""",
    ),

    # ----------------------------------------------- FIX / LEGACY-API TRAPS
    dict(
        id="fix_scale_assignment", tier="fix",
        prompt=(
            "This Blender script fails. Fix it.\n\n"
            "```python\nimport bpy\nbpy.ops.mesh.primitive_cube_add(location=(0, 0, 0))\n"
            "bpy.context.object.scale = 2\n```\n\n"
            "Error: TypeError: bpy_struct: item.attr = val: sequence expected at dimension 1, not 'int'"
        ),
        setup="",
        reference="""import bpy
bpy.ops.mesh.primitive_cube_add(location=(0, 0, 0))
bpy.context.object.scale = (2, 2, 2)
""",
        check="""
o = bpy.context.active_object
assert all(abs(s - 2) < 1e-4 for s in o.scale)
""",
    ),
    dict(
        id="fix_legacy_28_api", tier="fix",
        prompt=(
            "This script was written for Blender 2.7x and fails in Blender 4.x. Update it.\n\n"
            "```python\nimport bpy\nme = bpy.data.meshes.new('Quad')\n"
            "me.from_pydata([(0,0,0),(1,0,0),(1,1,0),(0,1,0)], [], [(0,1,2,3)])\n"
            "me.update()\nob = bpy.data.objects.new('Quad', me)\n"
            "bpy.context.scene.objects.link(ob)\n```\n\n"
            "Error: AttributeError: 'Scene' object has no attribute 'objects.link' / 'bpy_prop_collection' object has no attribute 'link'"
        ),
        setup="",
        reference="""import bpy
me = bpy.data.meshes.new('Quad')
me.from_pydata([(0,0,0),(1,0,0),(1,1,0),(0,1,0)], [], [(0,1,2,3)])
me.update()
ob = bpy.data.objects.new('Quad', me)
bpy.context.scene.collection.objects.link(ob)
""",
        check="""
o = bpy.data.objects.get("Quad")
assert o is not None and o.name in bpy.context.scene.objects
assert len(o.data.polygons) == 1
""",
    ),
    dict(
        id="fix_context_override_trap", tier="fix",
        prompt=(
            "This script crashes when run from a script, not the UI. Fix it so it works.\n\n"
            "```python\nimport bpy\nbpy.ops.mesh.primitive_cube_add()\n"
            "bpy.ops.object.mode_set(mode='EDIT')\nbpy.ops.mesh.subdivide(number_cuts=2)\n"
            "bpy.ops.object.mode_set(mode='OBJECT')\n```\n\n"
            "Goal: the cube should end up subdivided (more than 8 vertices) and back in Object Mode."
        ),
        setup="",
        reference="""import bpy
bpy.ops.mesh.primitive_cube_add()
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_all(action='SELECT')
bpy.ops.mesh.subdivide(number_cuts=2)
bpy.ops.object.mode_set(mode='OBJECT')
""",
        check="""
o = bpy.context.active_object
assert o.mode == 'OBJECT'
assert len(o.data.vertices) > 8
""",
    ),
]