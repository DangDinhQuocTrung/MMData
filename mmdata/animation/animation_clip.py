import numpy as np
from typing import List
from mmdata.animation.animation_track import AnimationTrack, SkeletalTrack, MorphTrack
from mmdata.animation.interpolation import BezierInterpolationMethod
from mmdata.animation.geometry import Geometry
from mmdata.animation.skeleton import Skeleton


class FramePoseData:
    """
    FramePoseData consists of pose data corresponding to one timestamp of VMD.
    """
    def __init__(self, frame_interpolation: dict):
        """
        :param frame_interpolation: {track_name: value}
        """
        self.poses = dict()

        for track_name, value in frame_interpolation.items():
            start_index = track_name.find("[") + 1
            end_index = track_name.find("]")
            bone_name = track_name[start_index:end_index]
            bone_name = int(bone_name) if bone_name.isdigit() else bone_name

            if bone_name not in self.poses:
                self.poses[bone_name] = dict()

            if isinstance(bone_name, int):
                self.poses[bone_name]["influence"] = value
            elif "position" in track_name:
                self.poses[bone_name]["pos"] = value
            elif "quaternion" in track_name:
                self.poses[bone_name]["q"] = value
            else:
                raise ValueError("Invalid bone name!!")
        return


class AnimationClip:
    """
    Animation sequence that a VMD file describes.
    """
    def __init__(self, tracks: List[AnimationTrack]):
        self.tracks = tracks

    def get_frame_pose_data(self, t: float) -> FramePoseData:
        """
        :param t: timestamp
        :return: current pose
        """
        frame_interpolation = dict()

        for track in self.tracks:
            times_bigger_than_t = track.times > t
            bigger_index = np.where(times_bigger_than_t)[0]
            bigger_index = -1 if bigger_index.shape[0] == 0 else bigger_index[0]

            if bigger_index == -1:
                # get last element since frame timestamp is bigger than the whole track
                # no interpolation needed
                frame_interpolation[track.name] = track.values[bigger_index]
                continue
            frame_interpolation[track.name] = track.get_value(bigger_index - 1, bigger_index, t)

        # convert to VPD
        return FramePoseData(frame_interpolation)


class AnimationClipBuilder:
    def __init__(self):
        # only Bezier is supported at the moment
        self.interpolation_method = BezierInterpolationMethod()
        self.tracks: [AnimationTrack] = []

    def reset(self):
        self.tracks = []

    def __build_skeletal_animation(self, vmd, skeleton: Skeleton):
        """
        :param vmd:
        :param skeleton:
        """
        bones = skeleton.bones
        bone_dict_name = dict()
        motions = dict()
        tracks = []

        # list existing bones
        for bone in bones:
            bone_dict_name[bone.name] = True

        # group by name
        for motion in vmd.motions:
            if bone_dict_name.get(motion.name, False):
                if motion.name not in motions:
                    motions[motion.name] = []
                motions[motion.name].append(motion)

        # loop by name
        for bone_name, bone_motions in motions.items():
            bone_motions = sorted(bone_motions, key=lambda x: x.frame)
            base_position = skeleton.bone_by_name[bone_name].position

            times = []
            positions = []
            rotations = []
            p_interpolations = []
            r_interpolations = []

            for step_bone_motion in bone_motions:
                pos, q = step_bone_motion.pos, step_bone_motion.q
                interpolation = step_bone_motion.complement
                times.append(step_bone_motion.frame / 30)
                positions.append(np.array([pos.x, pos.y, -pos.z]) + base_position)
                rotations.append(np.array([q.x, -q.y, q.z, q.w]))

                for int_index in range(0, 4):
                    if int_index < 3:
                        p_interpolations.append(np.array([
                            interpolation[int_index + 0] / 127, interpolation[int_index + 8] / 127,
                            interpolation[int_index + 4] / 127, interpolation[int_index + 12] / 127,
                        ]))
                    else:
                        r_interpolations.append(np.array([
                            interpolation[int_index + 0] / 127, interpolation[int_index + 8] / 127,
                            interpolation[int_index + 4] / 127, interpolation[int_index + 12] / 127,
                        ]))

            times = np.array(times)
            positions = np.stack(positions, axis=0)
            rotations = np.stack(rotations, axis=0)
            p_interpolations = np.stack(p_interpolations, axis=0)
            r_interpolations = np.stack(r_interpolations, axis=0)

            # example name: ".bones[センター].quaternion"
            key_name = f".bones[{bone_name}]"
            tracks.append(SkeletalTrack(f"{key_name}.position", times, positions, p_interpolations, self.interpolation_method))
            tracks.append(SkeletalTrack(f"{key_name}.quaternion", times, rotations, r_interpolations, self.interpolation_method))
        self.tracks += tracks

    def __build_morph_animation(self, vmd, geometry: Geometry):
        """
        :param vmd:
        :param geometry:
        """
        tracks = []
        morphs = dict()

        # group by name
        for vmd_morph in vmd.shapes:
            if vmd_morph.name in geometry.morph_target_dict:
                if vmd_morph.name not in morphs:
                    morphs[vmd_morph.name] = []
                morphs[vmd_morph.name].append(vmd_morph)

        # loop by name
        for morph_name, morph_motions in morphs.items():
            morph_motions = sorted(morph_motions, key=lambda x: x.frame)

            times = []
            values = []

            for step_morph_motion in morph_motions:
                times.append(step_morph_motion.frame / 30)
                values.append(step_morph_motion.ratio)

            times = np.array(times)
            values = np.array(values)

            tracks.append(MorphTrack(
                f".morphTargetInfluences[{geometry.morph_target_dict[morph_name]}]",
                times, values))
        self.tracks += tracks

    def from_vmd_and_skeleton(self, vmd, geometry: Geometry, skeleton: Skeleton):
        """
        :param vmd:
        :param geometry:
        :param skeleton:
        :return: animation clip corresponding to VMD file
        """
        self.reset()
        self.__build_skeletal_animation(vmd, skeleton)
        self.__build_morph_animation(vmd, geometry)
        return AnimationClip(self.tracks)
