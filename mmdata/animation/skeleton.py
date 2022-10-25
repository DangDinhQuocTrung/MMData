from __future__ import annotations

import numpy as np
import mmdata.utils.quaternion_utils as quaternion_utils
from mmdata.configs.bone_dictionary import bone_jp_to_eng_converter


class Bone:
    def __init__(self, name, position, quaternion, scale):
        self.name = name
        self.position = position
        self.quaternion = quaternion
        self.scale = scale
        self.rest_position = position
        self.rest_quaternion = quaternion
        self.rest_scale = scale

        self.matrix = self.compute_local_matrix()
        self.matrix_world = self.matrix.copy()
        self.parent = None
        self.children = []

    def rest_pose(self):
        self.position = self.rest_position
        self.quaternion = self.rest_quaternion
        self.scale = self.rest_scale
        self.matrix = self.compute_local_matrix()

    def compute_local_matrix(self):
        p = self.position
        q = self.quaternion
        s = self.scale
        qx2, qy2, qz2 = (q[0] + q[0]), (q[1] + q[1]), (q[2] + q[2])
        qxx, qxy, qxz = (q[0] * qx2), (q[0] * qy2), (q[0] * qz2)
        qyy, qyz, qzz = (q[1] * qy2), (q[1] * qz2), (q[2] * qz2)
        qwx, qwy, qwz = (q[3] * qx2), (q[3] * qy2), (q[3] * qz2)

        matrix_world = np.array([
            [(1.0 - (qyy + qzz)) * s[0], (qxy - qwz) * s[1], (qxz + qwy) * s[2], p[0]],
            [(qxy + qwz) * s[0], (1.0 - (qxx + qzz)) * s[1], (qyz - qwx) * s[2], p[1]],
            [(qxz - qwy) * s[0], (qyz + qwx) * s[1], (1.0 - (qxx + qyy)) * s[2], p[2]],
            [0.0, 0.0, 0.0, 1.0],
        ])
        return matrix_world

    def decompose_matrix_world(self):
        mw = self.matrix_world
        position = self.matrix_world[:3, 3].copy()
        scale = np.array([
            np.linalg.norm(np.array([mw[0, 0], mw[1, 0], mw[2, 0]])),
            np.linalg.norm(np.array([mw[0, 1], mw[1, 1], mw[2, 1]])),
            np.linalg.norm(np.array([mw[0, 2], mw[1, 2], mw[2, 2]])),
        ])

        if np.linalg.det(mw) < 0.0:
            scale[0] = -scale[0]

        temp_m = self.matrix_world.copy()
        temp_m[:, 0] *= (1.0 / scale[0])
        temp_m[:, 1] *= (1.0 / scale[1])
        temp_m[:, 2] *= (1.0 / scale[2])
        q = quaternion_utils.quaternion_from_rotation_matrix(temp_m)
        return position, q, scale

    def add(self, child: Bone):
        self.children.append(child)
        child.parent = self

    def update_matrix_world(self):
        self.matrix = self.compute_local_matrix()
        if self.parent is None:
            self.matrix_world = self.matrix.copy()
        else:
            self.matrix_world = np.matmul(self.parent.matrix_world, self.matrix)
        for child in self.children:
            child.update_matrix_world()
        return self.matrix_world


class Skeleton:
    def __init__(self, bones: [Bone], top_most: [Bone]):
        self.bones = bones
        self.top_most = top_most
        self.bone_inverses = None
        self.bone_matrices = None
        self.bone_by_name = dict()
        self.rest_pose()

    def rest_pose(self):
        self.bone_matrices = np.zeros([len(self.bones), 4, 4], dtype=np.float32)
        bone_inverses = []

        for i in range(0, len(self.bones)):
            self.bones[i].rest_pose()

        for top_bone in self.top_most:
            top_bone.update_matrix_world()

        for i in range(0, len(self.bones)):
            inverse = np.linalg.inv(self.bones[i].matrix_world)
            bone_inverses.append(inverse)
        self.bone_inverses = np.stack(bone_inverses, axis=0)

        for bone in self.bones:
            self.bone_by_name[bone.name] = bone
        return

    def update_matrix_world(self):
        for bone in self.top_most:
            bone.update_matrix_world()
        return

    def update_bone_matrices(self):
        for bone_index, bone in enumerate(self.bones):
            offset_matrix = np.matmul(bone.matrix_world, self.bone_inverses[bone_index])
            self.bone_matrices[bone_index, ...] = offset_matrix
        return

    def get_bone_vertices(self):
        bone_vertices = []

        for bone in self.bones:
            if bone.name in bone_jp_to_eng_converter:
                position, _, _ = bone.decompose_matrix_world()
                position = [position[0], position[1], position[2]]
                bone_vertices.append({
                    "name": bone_jp_to_eng_converter.get(bone.name, ""),
                    "position": position,
                })
        return bone_vertices
