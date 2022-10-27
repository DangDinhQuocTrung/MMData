import os
import json
import numpy as np
import pymeshio.pmx.reader
import pymeshio.vmd.reader
import mmdata.animation.solvers as solvers
import mmdata.utils.pmx_utils as pmx_utils

from PIL import Image, ImageOps
from mmdata.animation.geometry import Geometry
from mmdata.animation.skeleton import Skeleton
from mmdata.animation.animation_clip import AnimationClipBuilder
from mmdata.utils import quaternion_utils


def get_default_weight(deform, j):
    if j == 0:
        return 1.0
    elif j == 1 and hasattr(deform, "weight0"):
        return 1.0 - deform.weight0
    return 0.0


class Animator:
    def __init__(self, pmx_path, vmd_path):
        self.character_dir = os.path.dirname(pmx_path)
        self.character_name = os.path.basename(self.character_dir).replace(" ", "_")
        # build skeleton
        self.pmx = pymeshio.pmx.reader.read_from_file(pmx_path)
        self.geometry = Geometry(self.pmx)
        self.skeleton = Skeleton(*self.geometry.get_bone_hierarchy())

        # read animation clip
        vmd = pymeshio.vmd.reader.read_from_file(vmd_path)
        self.animation = AnimationClipBuilder().from_vmd_and_skeleton(vmd, self.geometry, self.skeleton)

    def __pose_skeleton_in_frame(self, pose_data, accumulative=False):
        bone_name_dict = dict()
        self.skeleton.rest_pose()

        for bone_index, bone in enumerate(self.skeleton.bones):
            bone_name_dict[bone.name] = bone_index

        for pose_bone_name, pose in pose_data.poses.items():
            if isinstance(pose_bone_name, int):
                self.geometry.morph_target_influences[pose_bone_name] = pose["influence"]
            else:
                bone_index = bone_name_dict.get(pose_bone_name, None)
                if bone_index is not None:
                    bone = self.skeleton.bones[bone_index]
                    if accumulative:
                        bone.position = bone.position + pose["pos"]
                        bone.quaternion = quaternion_utils.multiply_quaternions(bone.quaternion, pose["q"])
                    else:
                        bone.position = pose["pos"]
                        bone.quaternion = pose["q"]

        # AnimationMixer has looping properties
        # If t > max_t, t = t - max_t
        # However, even without looping, somehow q.x = -q.x
        self.skeleton.update_matrix_world()
        self.skeleton.update_bone_matrices()

        ik_solver = solvers.IkSolver(self.skeleton, self.geometry.iks)
        ik_solver.update()
        grant_solver = solvers.GrantSolver(self.skeleton, self.geometry.grants)
        grant_solver.update()

        self.skeleton.update_matrix_world()
        self.skeleton.update_bone_matrices()

    def __pose_vertices_with_skeleton(self, vertices):
        bone_matrices = self.skeleton.bone_matrices
        morph_target_influences = self.geometry.morph_target_influences
        morph_positions = self.geometry.morph_positions

        for i in range(0, vertices.shape[0]):
            vertex = vertices[i, :]
            vertex = np.concatenate([vertex, [1.0]], axis=0).reshape([1, 4])
            deform = self.pmx.vertices[i].deform

            # bone deformation
            deform_indices = [getattr(deform, f"index{j}", 0.0) for j in range(0, 4)]
            deform_weights = [getattr(deform, f"weight{j}", get_default_weight(deform, j)) for j in range(0, 4)]
            skinned_vertex = np.zeros_like(vertex)

            for j in range(0, 4):
                si, sw = int(deform_indices[j]), deform_weights[j]
                bone_matrix = bone_matrices[si].transpose()
                warped_vertex = np.matmul(vertex, bone_matrix) * sw
                skinned_vertex += warped_vertex

            # morph deformation
            morphed_vertex = np.zeros_like(vertices[i, :])

            for morph_index, morph_target in enumerate(morph_positions):
                influence = morph_target_influences[morph_index]

                if influence > 0.0:
                    target = morph_target["array"][i, :]
                    morphed_vertex += (target - vertices[i, :]) * influence

            # output
            skinned_vertex[0, :3] += morphed_vertex
            vertices[i, :] = skinned_vertex[0, :3]

        vertices[:, 1] = vertices[:, 1] - 10.0
        return vertices

    def copy_textures(self, texture_names, output_dir):
        visited_textures = set()
        for texture_name in texture_names:
            texture_path = os.path.join(self.character_dir, texture_name)
            texture_basename = os.path.basename(texture_name)

            if os.path.exists(texture_path) and (texture_path not in visited_textures):
                visited_textures.add(texture_path)
                texture_image = Image.open(texture_path)
                texture_image = ImageOps.flip(texture_image)
                new_texture_path = os.path.join(output_dir, texture_basename.replace(" ", "_"))
                texture_image.save(new_texture_path, quality=100)
        return

    def animate(self, timestamp: float, output_dir: str):
        vertices = self.geometry.vertices.copy()
        # capture model in animation
        if timestamp > 0.0:
            frame_pose_data = self.animation.get_frame_pose_data(timestamp)
            self.__pose_skeleton_in_frame(frame_pose_data, accumulative=False)
            vertices = self.__pose_vertices_with_skeleton(vertices)

        # write object mesh
        obj_content = pmx_utils.pmx_to_obj(self.pmx, self.geometry, vertices)
        with open(os.path.join(output_dir, f"{self.character_name}.obj"), "w+") as file:
            file.write(obj_content)

        # write bone
        bone_vertices = self.skeleton.get_bone_vertices()
        json.dump(bone_vertices, open(os.path.join(output_dir, "bone_vertices.json"), "w+", encoding="utf-8"), indent=4, ensure_ascii=False)

        # write materials
        mat_dict, mtl_output, texture_names = pmx_utils.pmx_to_mtl(self.pmx)
        with open(os.path.join(output_dir, "material.mtl"), "w+") as file:
            file.write(mtl_output)
        self.copy_textures(texture_names, output_dir)
        json.dump(mat_dict, open(os.path.join(output_dir, "material.json"), "w+", encoding="utf-8"), indent=4, ensure_ascii=False)
