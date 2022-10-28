import math
import numpy as np
import mmdata.utils.quaternion_utils as quaternion_utils


class GrantSolver:
    def __init__(self, skeleton, grants):
        self.skeleton = skeleton
        self.grants = grants

    def update(self):
        for grant in self.grants:
            self.__update_one(grant)
        return

    def __update_one(self, grant):
        bone = self.skeleton.bones[grant["index"]]
        parent_bone = self.skeleton.bones[grant["parent_index"]]

        if grant["is_local"]:
            # TODO: no implementation
            pass
        else:
            if grant["affect_position"]:
                # TODO: no implementation
                pass
            if grant["affect_rotation"]:
                self.__add_grant_rotation(bone, parent_bone.quaternion, grant["ratio"])
        return

    def __add_grant_rotation(self, bone, q, ratio):
        temp_q = np.array([0.0, 0.0, 0.0, 1.0])
        temp_q = quaternion_utils.slerp(temp_q, q, ratio)
        bone.quaternion = quaternion_utils.multiply_quaternions(bone.quaternion, temp_q)


class IkSolver:
    def __init__(self, skeleton, iks):
        self.skeleton = skeleton
        self.iks = iks

    def update(self):
        for ik in self.iks:
            self.__update_one(ik)
        return

    def __update_one(self, ik):
        effector = self.skeleton.bones[ik["effector"]]
        target = self.skeleton.bones[ik["target"]]

        target_pos, _, _ = target.decompose_matrix_world()
        links = ik["links"]
        iteration = ik["iteration"]

        for i in range(0, iteration):
            rotated = False

            for link in links:
                if not link["enabled"]:
                    continue

                link_bone = self.skeleton.bones[link["index"]]
                link_pos, link_q, link_scale = link_bone.decompose_matrix_world()
                inv_link_q = quaternion_utils.invert_quaternion(link_q)

                effector_pos, _, _ = effector.decompose_matrix_world()

                # work in link world
                effector_vec = effector_pos - link_pos
                effector_vec = quaternion_utils.vector_apply_quaternion(effector_vec, inv_link_q)
                effector_vec = effector_vec / np.linalg.norm(effector_vec)

                target_vec = target_pos - link_pos
                target_vec = quaternion_utils.vector_apply_quaternion(target_vec, inv_link_q)
                target_vec = target_vec / np.linalg.norm(target_vec)

                angle = float(np.dot(target_vec, effector_vec))
                angle = min(max(angle, 0.0), 1.0)
                angle = math.acos(angle)

                if angle < 1e-5:
                    continue
                if ik.get("max_angle", None):
                    angle = min(angle, ik["max_angle"])

                axis = np.cross(effector_vec, target_vec)
                axis = axis / np.linalg.norm(axis)
                _q = quaternion_utils.set_quaternion_from_axis_angle(axis, angle)
                link_bone.quaternion = quaternion_utils.multiply_quaternions(link_bone.quaternion, _q)

                if link["limit_rotation"]:
                    # TODO: fix Object3D.rotation property, conversion between Euler and Quaternion
                    link_rotation = quaternion_utils.quaternion_to_euler(link_bone.quaternion)
                    link_rotation = np.clip(link_rotation, link["rotation_min"], link["rotation_max"])
                    link_bone.quaternion = quaternion_utils.euler_to_quaternion(link_rotation)

                link_bone.update_matrix_world()
                rotated = True

            if not rotated:
                break
        return
