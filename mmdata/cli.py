import sys
import os
import glob
import argparse
import logging
import trimesh
import mmdata.utils.mesh_utils as mesh_utils
from natsort import natsorted
from mmdata.animation.animator import Animator
from mmdata.preprocessing.preprocessor import Preprocessor
from mmdata.renderer.renderer import Renderer
from mmdata.configs.configs import render_config


logging.basicConfig(
    format="[%(name)s %(asctime)s %(levelname)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.ERROR)


def parse_arguments(argv):
    desc = """
MMData is a tool to manipulate PMX (MikuMikuDance) models with VMD (Vocaloid Motion Data) files.
This can be used for visual purposes, or to generate a huge amount of data for training 3D mesh reconstruction models.
Examples of mesh reconstruction models are PIFu and PaMIR.
For more information, please visit https://github.com/DangDinhQuocTrung/MMData.

To pose a PMX model with a VMD file, use:
    mmdata pose -p ./model.pmx -v ./motion.vmd -t 50.0 -o ./mesh_output
To generate mesh reconstruction training data, use:
    mmdata gen -p ./pmx_input -v ./motion.vmd -t 50.0 -m ./mesh_output -i ./image_output
    """
    parser = argparse.ArgumentParser(prog="mmdata", description=desc, formatter_class=argparse.RawTextHelpFormatter)
    subparsers = parser.add_subparsers(help="commands", dest="command")

    pose_parser = subparsers.add_parser("pose")
    pose_parser.add_argument("--pmx", "-p", required=True, type=str, help="Path to the PMX model file.")
    pose_parser.add_argument("--vmd", "-v", required=True, type=str, help="Path to the VMD motion file.")
    pose_parser.add_argument("--timestamp", "-t", required=True, type=float, help="The timestamp in VMD file.")
    pose_parser.add_argument("--output_dir", "-o", type=str, default=".", help="Path to output directory.")
    pose_parser.add_argument("--no_display", action="store_true", help="Flag to prevent displaying the pose output.")

    gen_parser = subparsers.add_parser("gen")
    gen_parser.add_argument("--pmx_dir", "-p", required=True, type=str, help="Path to the PMX directories.")
    gen_parser.add_argument("--vmd", "-v", required=True, type=str, help="Path to the VMD motion file.")
    gen_parser.add_argument("--mesh_dir", "-m", required=True, type=str, help="Path to the OBJ directories.")
    gen_parser.add_argument("--image_dir", "-i", required=True, type=str, help="Path to the directory of output images.")
    gen_parser.add_argument("--timestamp", "-t", required=True, type=float, help="The timestamp in VMD file.")

    return parser.parse_args(argv)


def pose_pmx_model(args):
    logger = logging.getLogger("mmdata")

    # init the Animator
    try:
        animator = Animator(args.pmx, args.vmd)
    except FileNotFoundError:
        logger.error("PMX/VMD file does not exist!")
        return
    except Exception as e:
        logger.error(f"Reading files failed: {e}")
        return

    model_name = os.path.splitext(os.path.basename(args.pmx))[0]
    # make output directory
    dirname = f"{model_name}_step_{args.timestamp:.2f}_model"
    output_dir = os.path.join(args.output_dir, dirname)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    else:
        logger.warning("The output directory is not empty.")

    # Animate the model
    try:
        animator.animate(args.timestamp, output_dir)
    except Exception as e:
        logger.error(f"Animation failed: {e}")
        return

    # Show the output
    if not args.no_display:
        mesh = trimesh.load(os.path.join(output_dir, f"{model_name}.obj"))
        mesh_utils.display_mesh(mesh)
    return


def generate_data(args):
    logger = logging.getLogger("mmdata")

    # init processors
    model_dirs = natsorted(glob.glob(os.path.join(args.pmx_dir, "*")))
    preprocessor = Preprocessor()
    # init OpenGL
    try:
        renderer = Renderer(render_config, args.image_dir)
    except Exception as e:
        logger.error(f"Init of OpenGL failed: {e}")
        return

    for model_dir in model_dirs:
        pmx_path = glob.glob(os.path.join(model_dir, "*.pmx"))[0]
        # init the Animator
        try:
            animator = Animator(pmx_path, args.vmd)
        except FileNotFoundError:
            logger.error("PMX/VMD file does not exist!")
            return
        except Exception as e:
            logger.error(f"Reading files failed: {e}")
            return

        model_name = os.path.basename(model_dir)
        model_pose_name = f"{model_name}_step_{args.timestamp:.2f}_model"

        model_pose_dir = os.path.join(args.mesh_dir, model_pose_name)
        os.makedirs(model_pose_dir, exist_ok=True)

        # Animate the model
        try:
            animator.animate(args.timestamp, model_pose_dir)
        except Exception as e:
            logger.error(f"Animation failed: {e}")
            return

        # Preprocessing: normalization and computing PRT
        try:
            preprocessor.process(os.path.join(model_pose_dir, f"{model_name}.obj"))
        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            return

        # Render 3D mesh to 2D images
        try:
            renderer.render_mesh(model_pose_dir)
        except Exception as e:
            logger.error(f"Rendering failed: {e}")
            return
    return


def main(args):
    if args.command == "pose":
        pose_pmx_model(args)
    elif args.command == "gen":
        generate_data(args)
    else:
        raise ValueError("Invalid command!")
    return


if __name__ == "__main__":
    # document
    main(parse_arguments(sys.argv[1:]))
