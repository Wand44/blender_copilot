#!/usr/bin/env python3
"""Build a static, version-tagged Blender API cache from a real Blender install.

Run inside Blender headless:
    blender -b --python build_api_cache.py -- <out_path>

The cache is the single source of truth for the synthetic-data pipeline:
    - bpy.types attributes (RNA properties + methods + dir())
    - bpy.ops signatures and per-operator kwarg/enum tables
    - context / data / props / utils / app roots
    - extra modules: bmesh, mathutils, bpy_extras, gpu, aud, freestyle, idprop, imbuf
    - namespace tags for diversity capping
    - a small deprecated/removed table for the 5.x line
"""
import json
import os
import sys
import bpy
import config


def enum_values(cls, prop):
    try:
        return [e.identifier for e in cls.bl_rna.properties[prop].enum_items]
    except Exception:
        return []


def _mod_attrs(obj):
    try:
        return sorted(a for a in dir(obj) if not a.startswith("_"))
    except Exception:
        return []


def collect_types():
    types = {}
    namespaces = {}
    for name in dir(bpy.types):
        try:
            cls = getattr(bpy.types, name)
        except Exception:
            continue
        attrs = set(dir(cls))
        try:
            attrs.update(cls.bl_rna.properties.keys())
        except Exception:
            pass
        try:
            attrs.update(cls.bl_rna.functions.keys())
        except Exception:
            pass
        types[name] = sorted(attrs)
        # namespace guess from module
        mod = getattr(cls, "__module__", "").split(".")[-1]
        namespaces[name] = f"types.{mod}" if mod else "types.other"
    return types, namespaces


def collect_ops():
    ops = set()
    ops_props = {}
    ops_signatures = {}
    namespaces = {}
    for cat in dir(bpy.ops):
        if cat.startswith("_"):
            continue
        try:
            cat_obj = getattr(bpy.ops, cat)
        except Exception:
            continue
        for op in _mod_attrs(cat_obj):
            path = f"{cat}.{op}"
            ops.add(path)
            namespaces[path] = f"ops.{cat}"
            try:
                rna = getattr(cat_obj, op).get_rna_type()
                props = {}
                sig_parts = []
                for pname, p in rna.properties.items():
                    if pname == "rna_type":
                        continue
                    enums = [i.identifier for i in getattr(p, "enum_items", [])]
                    props[pname] = enums or None
                    sig_parts.append(pname)
                ops_props[path] = props
                ops_signatures[path] = f"bpy.ops.{path}({', '.join(sig_parts)})"
            except Exception:
                ops_props[path] = {}
                ops_signatures[path] = f"bpy.ops.{path}(...)"
    return sorted(ops), ops_props, ops_signatures, namespaces


def collect_roots():
    roots = {}
    for root in ("props", "utils", "app"):
        roots[root] = _mod_attrs(getattr(bpy, root))
    roots["context"] = _mod_attrs(bpy.context)
    roots["data"] = _mod_attrs(bpy.data)
    return roots


def collect_modules():
    modules = {}
    namespaces = {}
    handled = {"types", "ops", "props", "utils", "app", "context", "data"}
    for name in dir(bpy):
        if name.startswith("_") or name in handled:
            continue
        obj = getattr(bpy, name)
        if hasattr(obj, "__dict__") or hasattr(obj, "__path__"):
            modules[name] = _mod_attrs(obj)
            namespaces[name] = f"bpy.{name}"
    # commonly-used non-bpy top-level modules shipped with Blender
    for mod_name in ("bmesh", "mathutils", "bpy_extras", "gpu", "aud",
                     "freestyle", "idprop", "imbuf"):
        try:
            mod = __import__(mod_name)
            modules[mod_name] = _mod_attrs(mod)
            namespaces[mod_name] = mod_name
            # sub-modules of mathutils
            if mod_name == "mathutils":
                for sub in ("bvhtree", "geometry", "interpolate", "kdtree", "noise"):
                    try:
                        sub_mod = getattr(mod, sub)
                        modules[f"mathutils.{sub}"] = _mod_attrs(sub_mod)
                        namespaces[f"mathutils.{sub}"] = "mathutils"
                    except Exception:
                        pass
        except Exception:
            pass
    return modules, namespaces


def collect_deprecated():
    """Known renames/removals in the 4.x/5.x line. Keys are dotted attribute
    chains; values are the preferred replacement or None if fully removed."""
    return {
        "bgl": None,
        "bpy.types.Mesh.use_auto_smooth": None,
        "bpy.ops.mesh.calc_normals": None,
        "bpy.ops.mesh.edges_sharp_to_custom_normals": None,
        "bpy.types.NodeTree.inputs.new": "interface.new_socket",
        "bpy.types.NodeTree.outputs.new": "interface.new_socket",
        "bpy.types.NodeTree.inputs": "interface",
        "bpy.types.NodeTree.outputs": "interface",
        "bpy.types.SequenceEditor.sequences": "strips",
        "bpy.types.SequencesTopLevel.sequences": "strips",
        "bpy.context.scene.render.engine": "BLENDER_EEVEE_NEXT",  # present in 4.2+
        "bpy.types.SceneEEVEE": "SceneEEVEE",  # still exists, engine id changed
        "bpy.types.Bone.layers": None,
        "bpy.types.Object.layers": None,
        "bpy.types.Scene.update": None,
        "bpy.types.Object.hide": None,
        "bpy.types.Object.hide_render": "visible_render",
        "bpy.types.Object.hide_viewport": "visible_viewport",
        "bpy.ops.view3d.layers": None,
    }


def main():
    out = sys.argv[-1] if sys.argv[-1].endswith(".json") else str(config.CACHE_PATH)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    print("CACHE building...", bpy.app.version_string)
    types, type_ns = collect_types()
    ops, ops_props, ops_sigs, ops_ns = collect_ops()
    roots = collect_roots()
    modules, mod_ns = collect_modules()

    # bl_panel enums
    try:
        regions = enum_values(bpy.types.Panel, "bl_region_type")
        spaces = enum_values(bpy.types.Panel, "bl_space_type")
    except Exception:
        regions, spaces = [], []

    namespaces = {}
    namespaces.update({f"bpy.types.{k}": v for k, v in type_ns.items()})
    namespaces.update({f"bpy.ops.{k}": v for k, v in ops_ns.items()})
    namespaces.update(mod_ns)

    dump = {
        "blender_version": bpy.app.version_string,
        "blender_version_tuple": bpy.app.version,
        "target_version": config.TARGET_BLENDER,
        "blender_bin": config.BLENDER_BIN,
        "types": types,
        "ops": ops,
        "ops_props": ops_props,
        "ops_signatures": ops_sigs,
        "roots": roots,
        "modules": modules,
        "bl_region_type": regions,
        "bl_space_type": spaces,
        "deprecated": collect_deprecated(),
        "namespaces": namespaces,
    }

    with open(out, "w") as f:
        json.dump(dump, f, indent=2)
    print("CACHE_OK", out, len(types), len(ops))


if __name__ == "__main__":
    main()
