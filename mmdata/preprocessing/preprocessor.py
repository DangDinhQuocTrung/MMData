import os
import math
import pickle
import numpy as np
import trimesh
import mmdata.utils.mesh_utils as mesh_utils
import mmdata.utils.prt_utils as prt_utils


class Preprocessor:
    def __init__(self, n=40, order=2):
        self.n = n
        self.order = order

    def compute_prt(self, mesh_path):
        mesh = trimesh.load(mesh_path, file_type="obj", split_object=True)
        prt_dict, face_dict, out_dict = {}, {}, {}

        # sample parameters
        vectors_orig, phi, theta = prt_utils.sample_spherical_directions(self.n)
        sh_orig = prt_utils.get_sh_coeffs(self.order, phi, theta)
        w = 4.0 * math.pi / (self.n * self.n)

        for key, geometry in mesh_utils.get_mesh_geometry(mesh).items():
            origins = geometry.vertices
            normals = geometry.vertex_normals.copy()
            n_v = origins.shape[0]

            normals = normals * -1.0
            origins = np.repeat(origins[:, None], self.n, axis=1).reshape(-1, 3)
            normals = np.repeat(normals[:, None], self.n, axis=1).reshape(-1, 3)
            prt_all = None

            for i in range(0, self.n):
                sh = np.repeat(sh_orig[None, (i * self.n):((i + 1) * self.n)], n_v, axis=0).reshape(-1, sh_orig.shape[1])
                vectors = np.repeat(vectors_orig[None, (i * self.n):((i + 1) * self.n)], n_v, axis=0).reshape(-1, 3)

                dots = (vectors * normals).sum(1)
                front = (dots > 0.0)

                delta = 1e-3 * min(geometry.bounding_box.extents)
                hits = geometry.ray.intersects_any(origins + delta * normals, vectors)
                no_hits = np.logical_and(front, np.logical_not(hits))
                prt = (no_hits.astype(np.float32) * dots)[:, None] * sh

                if prt_all is not None:
                    prt_all += (prt.reshape(-1, self.n, sh.shape[1]).sum(1))
                else:
                    prt_all = (prt.reshape(-1, self.n, sh.shape[1]).sum(1))

            prt = w * prt_all
            prt_dict[key] = prt
            face_dict[key] = geometry.faces

        # NOTE: trimesh sometimes break the original vertex order, but topology will not change.
        # when loading PRT in other program, use the triangle list from trimesh.
        for key, prt in prt_dict.items():
            face = face_dict[key]
            out_dict[key] = {"bounce0": prt, "face": face}
        return out_dict

    def __save_normalized_mesh(self, mesh_path, scale, offset):
        mesh_lines = []
        new_mesh_lines = []

        with open(mesh_path) as file:
            for line in file:
                mesh_lines.append(line.strip())

        for line in mesh_lines:
            if line.startswith("v "):
                values = line.replace("  ", " ").split(" ")
                values = [float(v) for v in values[1:]]
                values[0] = (values[0] - offset[0]) * scale
                values[1] = (values[1] - offset[1]) * scale
                values[2] = (values[2] - offset[2]) * scale
                new_line = f"v {values[0]} {values[1]} {values[2]}"
                new_mesh_lines.append(new_line)
            else:
                new_mesh_lines.append(line)

        with open(mesh_path, "w") as file:
            for line in new_mesh_lines:
                file.write("%s\n" % line)
        return

    def normalize(self, mesh_path):
        mesh = trimesh.load(mesh_path, file_type="obj", split_object=True)
        mesh_geometry_parts = []

        # get all vertices
        for key, geometry in mesh_utils.get_mesh_geometry(mesh).items():
            mesh_geometry_parts.append(geometry.vertices)
        mesh_v = np.concatenate(mesh_geometry_parts, axis=0)

        # compute bounding box of all vertices
        min_xyz = np.min(mesh_v, axis=0, keepdims=True)
        max_xyz = np.max(mesh_v, axis=0, keepdims=True)
        offset = (min_xyz + max_xyz) * 0.5

        # compute scale
        scale_inv = np.max(max_xyz - min_xyz)
        scale = 1.0 / scale_inv * (0.75 + 0.5 * 0.15)
        self.__save_normalized_mesh(mesh_path, scale, offset[0])

    def process(self, mesh_path):
        self.normalize(mesh_path)
        out_dict = self.compute_prt(mesh_path)

        bounce_dir = os.path.join(os.path.dirname(mesh_path), "bounce")
        os.makedirs(bounce_dir, exist_ok=True)
        pickle.dump(
            out_dict,
            open(os.path.join(bounce_dir, "prt_data.pkl"), "wb+"))
