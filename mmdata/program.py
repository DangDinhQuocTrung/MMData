import os
import glob
from mmdata.animation.animator import Animator
from mmdata.preprocessing.preprocessor import Preprocessor
from mmdata.renderer.renderer import Renderer
from mmdata.configs.configs import render_config


def main():
    pmx_dir = "/home/tyler/work/data/mmdata/test_data"
    vmd_path = "/home/tyler/work/data/mmdata/wavefile_v2.vmd"
    mesh_output_dir = "/home/tyler/work/data/mmdata/mesh_output"
    image_output_dir = "/home/tyler/work/data/mmdata/image_output"
    step = 50

    model_dirs = glob.glob(os.path.join(pmx_dir, "*"))
    renderer = Renderer(render_config, image_output_dir)
    preprocessor = Preprocessor()

    for model_dir in model_dirs:
        pmx_path = glob.glob(os.path.join(model_dir, "*.pmx"))[0]
        animator = Animator(pmx_path, vmd_path)
        model_name = os.path.basename(model_dir)
        model_pose_name = f"{model_name}_step_{step:02d}"

        model_pose_dir = os.path.join(mesh_output_dir, model_pose_name)
        os.makedirs(model_pose_dir, exist_ok=True)

        animator.animate(step, model_pose_dir)
        preprocessor.process(os.path.join(model_pose_dir, f"{model_name}.obj"))
        renderer.render_mesh(model_pose_dir)
    return


if __name__ == "__main__":
    main()
