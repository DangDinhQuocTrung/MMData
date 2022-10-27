import sys
import os
import glob
import argparse
import trimesh
import mmdata.utils.mesh_utils as mesh_utils
from natsort import natsorted
from mmdata.animation.animator import Animator
from mmdata.preprocessing.preprocessor import Preprocessor
from mmdata.renderer.renderer import Renderer
from mmdata.configs.configs import render_config


def parse_arguments(argv):
    desc = """
    """
    parser = argparse.ArgumentParser(prog="mmdata", description=desc, formatter_class=argparse.RawTextHelpFormatter)
    subparsers = parser.add_subparsers(help="commands", dest="command")

    pose_parser = subparsers.add_parser("pose")
    pose_parser.add_argument("--pmx", "-p", required=True, type=str, help="Path to the PMX model file.")
    pose_parser.add_argument("--vmd", "-v", required=True, type=str, help="Path to the VMD motion file.")
    pose_parser.add_argument("--timestamp", "-t", required=True, type=float, help="The timestamp in VMD file.")
    pose_parser.add_argument("--output_dir", "-o", type=str, default=".", help="Path to output directory.")
    pose_parser.add_argument("--no_display", action="store_true", help="Flag to display the pose output.")

    gen_parser = subparsers.add_parser("gen")
    gen_parser.add_argument("--pmx_dir", "-p", required=True, type=str, help="Path to the PMX directories.")
    gen_parser.add_argument("--vmd", "-v", required=True, type=str, help="Path to the VMD motion file.")
    gen_parser.add_argument("--mesh_dir", "-m", required=True, type=str, help="Path to the OBJ directories.")
    gen_parser.add_argument("--image_dir", "-i", required=True, type=str, help="Path to the directory of output images.")
    gen_parser.add_argument("--timestamp", "-t", required=True, type=float, help="The timestamp in VMD file.")

    return parser.parse_args(argv)


def pose_pmx_model(args):
    animator = Animator(args.pmx, args.vmd)
    model_name = os.path.splitext(os.path.basename(args.pmx))[0]

    # make output directory
    dirname = f"{model_name}_step_{args.timestamp:.2f}_model"
    output_dir = os.path.join(args.output_dir, dirname)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    animator.animate(args.timestamp, output_dir)
    if not args.no_display:
        mesh = trimesh.load(os.path.join(output_dir, f"{model_name}.obj"))
        mesh_utils.display_mesh(mesh)
    return


def generate_data(args):
    model_dirs = natsorted(glob.glob(os.path.join(args.pmx_dir, "*")))
    renderer = Renderer(render_config, args.image_dir)
    preprocessor = Preprocessor()

    for model_dir in model_dirs:
        pmx_path = glob.glob(os.path.join(model_dir, "*.pmx"))[0]
        animator = Animator(pmx_path, args.vmd)
        model_name = os.path.basename(model_dir)
        model_pose_name = f"{model_name}_step_{args.timestamp:.2f}_model"

        model_pose_dir = os.path.join(args.mesh_dir, model_pose_name)
        os.makedirs(model_pose_dir, exist_ok=True)

        animator.animate(args.timestamp, model_pose_dir)
        preprocessor.process(os.path.join(model_pose_dir, f"{model_name}.obj"))
        renderer.render_mesh(model_pose_dir)
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
    # comment, try-catch, type-hint
    # document
    main(parse_arguments(sys.argv[1:]))
