import pytest
import pathlib
from mmdata.animation.animator import Animator


ASSETS_DIR = pathlib.Path(__file__).parent.parent.joinpath("assets")


def test_animator():
    pmx_path = ASSETS_DIR.joinpath("pmx_data/A/A.pmx")
    vmd_path = ASSETS_DIR.joinpath("walking.vmd")
    gt_obj_path = ASSETS_DIR.joinpath("mesh_data/A.obj")
    output_dir = ASSETS_DIR.joinpath("output")
    timestamp = 10.0

    output_dir.mkdir(parents=True, exist_ok=True)
    animator = Animator(pmx_path, vmd_path)
    animator.animate(timestamp, output_dir)

    gt_obj = open(gt_obj_path).read()
    out_obj = open(output_dir.joinpath("A.obj")).read()
    assert out_obj == gt_obj


if __name__ == "__main__":
    pytest.main()
