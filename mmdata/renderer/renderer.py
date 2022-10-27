import os
import glob
import math
import json
import pickle
import numpy as np
import cv2
import scipy.io
import tqdm
import pyexr
import mmdata.utils.mesh_utils as mesh_utils

from PIL import Image
from mmdata.renderer.gl.init_gl import initialize_GL_context
from mmdata.renderer.gl.prt_render import PRTRender
from mmdata.renderer.camera import Camera


class Renderer:
    """
    Render mesh to images with OpenGL.
    """
    def __init__(self, config: dict, output_dir: str):
        self.config = config
        self.output_dir = output_dir

        # load lighting params
        initialize_GL_context(
            width=config["image_size"], height=config["image_size"],
            egl=config["egl"])
        self.render_uv_params = {"width": config["image_size"], "height": config["image_size"], "egl": config["egl"]}

        # initialize renderer
        self.render = PRTRender(width=config["image_size"], height=config["image_size"], ms_rate=1.0, egl=config["egl"])
        self.cam = Camera(
            width=config["image_size"], height=config["image_size"],
            focal=config["cam_f"], near=config["cam_near"], far=config["cam_far"])
        self.cam.sanity_check()

    def __generate_cameras(self):
        cams = []
        target = [0, 0, 0]
        up = [0, 1, 0]
        dist = self.config["cam_dist"]
        angles = [(math.pi * 2 / self.config["view_number"]) * view_idx for view_idx in range(0, self.config["view_number"])]

        for angle in angles:
            eye = np.asarray([dist * math.sin(angle), 0, dist * math.cos(angle)])

            fwd = np.asarray(target, np.float64) - eye
            fwd /= np.linalg.norm(fwd)
            right = np.cross(fwd, up)
            right /= np.linalg.norm(right)
            down = np.cross(fwd, right)

            # eye is camera distance
            # fwd is camera translation
            # right and down is camera rotation
            cams.append(
                {
                    "center": eye,
                    "direction": fwd,
                    "right": right,
                    "up": -down,
                }
            )
        return cams

    def render_mesh(self, input_dir: str):
        """
        Render OBJ mesh to images.
        :param input_dir: contains OBJ and texture files.
        """
        input_name = os.path.basename(input_dir)
        output_dir = os.path.join(self.output_dir, input_name)

        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "color"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "color_uv"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "mask"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "meta"), exist_ok=True)

        # load JSON
        material_map = json.load(open(os.path.join(input_dir, "material.json")))

        # read data
        mesh_filename = sorted(glob.glob(os.path.join(input_dir, "*.obj")))[0]
        prt_data = pickle.load(open(os.path.join(input_dir, "bounce/prt_data.pkl"), "rb"))

        # texture rendering
        # while scene renderer can allocate regions for different texture images, UV renderer cannot
        # every UV texture image is written into a single image
        # the texture that comes first appears on the top, while following images are beneath
        render_uv_dict = dict()

        for key_name, text_name in material_map.items():
            if key_name not in prt_data:
                continue

            prt, face_prt = prt_data[key_name]["bounce0"], prt_data[key_name]["face"]
            text_file = os.path.join(input_dir, material_map[key_name])
            vertices, faces, normals, faces_normals, textures, face_textures = mesh_utils.load_obj_mesh(
                mesh_filename, key_name, with_normal=True, with_texture=True)

            if not os.path.exists(text_file):
                continue
            elif text_file.endswith(".tga"):
                texture_image = np.array(Image.open(text_file).convert("RGB"))
            elif any([text_file.endswith(ext) for ext in [".jpg", ".png", ".jpeg"]]):
                texture_image = cv2.imread(text_file)
                texture_image = cv2.cvtColor(texture_image, cv2.COLOR_BGR2RGB)
            else:
                continue

            self.render.set_norm_mat(1.0, 0.0)
            tan, bitan = mesh_utils.compute_tangent(vertices, faces, normals, textures, face_textures)
            self.render.set_mesh(vertices, faces, normals, faces_normals, textures, face_textures, prt, face_prt, tan, bitan, mat_name=key_name)
            self.render.set_albedo(texture_image, mat_name=key_name)

            # seems that dict-by-key_name does not add new info, so go with dict-by-text_name
            if text_name not in render_uv_dict:
                render_uv_dict[text_name] = PRTRender(
                    width=self.render_uv_params["width"], height=self.render_uv_params["height"],
                    uv_mode=True, egl=self.render_uv_params["egl"])

            render_uv_dict[text_name].set_mesh(
                vertices, faces, normals, faces_normals, textures, face_textures,
                prt, face_prt, tan, bitan, mat_name=key_name)
            render_uv_dict[text_name].set_albedo(texture_image, mat_name=key_name)

        cam_params = self.__generate_cameras()
        sh_list = []
        uv_pos_dict = dict()

        for ci, cam_param in enumerate(tqdm.tqdm(cam_params, ascii=True)):
            self.cam.center = cam_param["center"]
            self.cam.right = cam_param["right"]
            self.cam.up = cam_param["up"]
            self.cam.direction = cam_param["direction"]
            self.cam.sanity_check()
            self.render.set_camera(self.cam)

            # sh is related to lighting
            sh = np.full([9, 3], 0.0)
            sh[0, :] = 0.5
            sh_list.append(sh)

            self.render.set_sh(sh)
            self.render.analytic = False
            self.render.use_inverse_depth = False
            self.render.display()

            out_all_f = self.render.get_color(0)
            out_all_f = cv2.cvtColor(out_all_f, cv2.COLOR_RGBA2BGR)
            out_all_f = np.uint8(out_all_f * 255)
            out_all_f[np.where((out_all_f == [0, 0, 0]).all(axis=-1))] = [255, 255, 255]
            cv2.imwrite(os.path.join(output_dir, "color", f"{ci:04d}.png"), out_all_f)

            out_mask = self.render.get_color(4)
            cv2.imwrite(os.path.join(output_dir, "mask", f"{ci:04d}.png"), np.uint8(out_mask * 255))

            for text_name, render_uv in render_uv_dict.items():
                text_name = text_name.replace(".png", "").replace(".jpg", "")
                text_name = text_name.replace(".tga", "")

                render_uv.set_camera(self.cam)
                render_uv.set_sh(sh)
                render_uv.analytic = False
                render_uv.use_inverse_depth = False
                render_uv.display()

                uv_color = render_uv.get_color(0)
                uv_color = cv2.cvtColor(uv_color, cv2.COLOR_RGBA2BGR)
                cv2.imwrite(
                    os.path.join(output_dir, "color_uv", f"{ci:04d}_{text_name}.png"),
                    np.uint8(uv_color * 255))

                if ci == 0:
                    uv_pos = render_uv.get_color(1)
                    uv_pos_dict[text_name] = uv_pos
                    uv_mask = uv_pos[:, :, 3]
                    cv2.imwrite(os.path.join(output_dir, "meta", f"uv_mask_{text_name}.png"), np.uint8(uv_mask * 255))

                    data = {"default": uv_pos[:, :, :3]}
                    # default is a reserved name
                    pyexr.write(os.path.join(output_dir, "meta", f"uv_pos_{text_name}.exr"), data)

                    uv_nml = render_uv.get_color(2)
                    uv_nml = cv2.cvtColor(uv_nml, cv2.COLOR_RGBA2BGR)
                    cv2.imwrite(os.path.join(output_dir, "meta", f"uv_nml_{text_name}.png"), np.uint8(uv_nml * 255))

        scipy.io.savemat(
            os.path.join(output_dir, "meta", "cam_data.mat"),
            {"cam": cam_params})
        scipy.io.savemat(
            os.path.join(output_dir, "meta", "sh_data.mat"),
            {"sh": sh_list})
        self.render.cleanup()
        for text_name, render_uv in render_uv_dict.items():
            render_uv.cleanup()
        return
