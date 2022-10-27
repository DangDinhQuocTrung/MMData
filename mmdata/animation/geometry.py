import numpy as np
from mmdata.animation.skeleton import Bone


class Geometry:
    """
    Geometry of a PMX model: vertices, normals, faces, bones, grants, etc.
    """
    def __init__(self, pmx):
        # read core attributes
        self.vertices = np.stack([[v.position.x, v.position.y, -v.position.z] for v in pmx.vertices], axis=0)
        self.normals = np.stack([[v.normal.x, v.normal.y, v.normal.z] for v in pmx.vertices], axis=0)
        self.uvs = np.stack([[v.uv.x, v.uv.y] for v in pmx.vertices], axis=0)
        self.faces = np.array([pmx.indices[i:(i + 3)] for i in range(0, len(pmx.indices), 3)])

        # bones
        self.bones = []
        self.bone_type_table = dict()

        for body in pmx.rigidbodies:
            value = body.mode
            if body.bone_index in self.bone_type_table:
                value = max(body.mode, self.bone_type_table.get(body.bone_index))
            self.bone_type_table[body.bone_index] = value

        for bone_index, bone in enumerate(pmx.bones):
            out_bone = {
                "index": bone_index,
                "transformation_class": bone.layer,
                "parent": bone.parent_index,
                "name": bone.name,
                "pos": np.array([bone.position.x, bone.position.y, -bone.position.z]),
                "rot": np.array([0, 0, 0, 1], dtype=np.float32),
                "scl": np.array([1, 1, 1]),
                "rigid_body_type": self.bone_type_table.get(bone_index, -1),
            }
            if bone.parent_index != -1:
                parent_bone = pmx.bones[bone.parent_index]
                parent_pos = np.array([parent_bone.position.x, parent_bone.position.y, -parent_bone.position.z])
                out_bone["pos"] -= parent_pos
            self.bones.append(out_bone)

        # ik
        self.iks = []

        for bone_index, bone in enumerate(pmx.bones):
            if bone.ik is None:
                continue

            ik_param = {
                "target": bone_index,
                "effector": bone.ik.target_index,
                "iteration": bone.ik.loop,
                "max_angle": bone.ik.limit_radian,
                "links": [],
            }

            for link in bone.ik.link:
                link_param = {
                    "index": link.bone_index,
                    "enabled": True,
                    "limit_rotation": link.limit_angle == 1,
                }
                if link.limit_angle == 1:
                    link_param["rotation_max"] = np.array([-link.limit_min[0], -link.limit_min[1], -link.limit_min[2]])
                    link_param["rotation_min"] = np.array([-link.limit_max[0], -link.limit_max[1], -link.limit_max[2]])
                ik_param["links"].append(link_param)

            self.iks.append(ik_param)
            self.bones[bone_index]["ik"] = ik_param

        # grant
        self.grant_entry_map = {}
        self.grants = []
        root_entry = {"parent": None, "children": [], "grant_param": None, "visited": False}

        for bone_index, bone in enumerate(pmx.bones):
            if bone.getExternalRotationFlag() or bone.getExternalTranslationFlag():
                grant_param = {
                    "index": bone_index,
                    "parent_index": bone.effect_index,
                    "ratio": bone.effect_factor,
                    "is_local": bone.hasFlag(0x0080),
                    "affect_rotation": bone.getExternalRotationFlag(),
                    "affect_position": bone.getExternalTranslationFlag(),
                    "transform_class": bone.layer,
                }
                self.grant_entry_map[bone_index] = {
                    "parent": None,
                    "children": [],
                    "grant_param": grant_param,
                    "visited": False,
                }

        # build a tree of grant hierarchy
        for bone_index, grant_entry in self.grant_entry_map.items():
            parent_index = grant_entry["grant_param"]["parent_index"]
            if parent_index in self.grant_entry_map:
                parent_grant_entry = self.grant_entry_map[parent_index]
            else:
                parent_grant_entry = root_entry
            grant_entry["parent"] = parent_grant_entry
            parent_grant_entry["children"].append(grant_entry)

        # sort grants from parent to children
        self.__traverse_grant(root_entry)

        # morph
        self.morph_targets = []
        self.morph_positions = []
        self.morph_target_influences = []
        self.morph_target_dict = dict()

        for morph in pmx.morphs:
            params = {"name": morph.name}
            attribute = {
                "array": np.zeros([len(pmx.vertices), 3], dtype=np.float32),
                "name": morph.name,
            }

            for i in range(0, self.vertices.shape[0]):
                attribute["array"][i, :] = self.vertices[i, :]

            if morph.morph_type == 0:
                for offset in morph.offsets:
                    morph2 = pmx.morphs[offset.morph_index]
                    if morph2.morph_type == 1:
                        for offset2 in morph2.offsets:
                            i = offset2.vertex_index
                            position_offset = np.array([offset2.position_offset[j] for j in range(0, 3)])
                            attribute["array"][i, :] += position_offset * offset.value

            elif morph.morph_type == 1:
                for offset in morph.offsets:
                    i = offset.vertex_index
                    position_offset = np.array([offset.position_offset[j] for j in range(0, 3)])
                    attribute["array"][i, :] += position_offset

            self.morph_targets.append(params)
            self.morph_positions.append(attribute)
            self.morph_target_influences.append(0)
            self.morph_target_dict[morph.name] = len(self.morph_positions) - 1

        self.morph_target_influences = np.array(self.morph_target_influences, dtype=np.float32)

    def __traverse_grant(self, entry):
        if entry["grant_param"] is not None:
            self.grants.append(entry["grant_param"])
            self.bones[entry["grant_param"]["index"]]["grant"] = entry["grant_param"]
        entry["visited"] = True

        for child in entry["children"]:
            if not child["visited"]:
                self.__traverse_grant(child)
        return

    def get_bone_hierarchy(self) -> ([Bone], [Bone]):
        bones = []
        top_most = []

        # first, create array of 'Bone' objects from geometry data
        for geo_bone in self.bones:
            bones.append(Bone(geo_bone["name"], geo_bone["pos"], geo_bone["rot"], geo_bone["scl"]))

        # second, create bone hierarchy
        for bone_index, geo_bone in enumerate(self.bones):
            if (geo_bone["parent"] != -1) and (geo_bone["parent"] < len(bones)):
                bones[geo_bone["parent"]].add(bones[bone_index])
            else:
                top_most.append(bones[bone_index])

        for top_bone in top_most:
            top_bone.update_matrix_world()
        return bones, top_most
