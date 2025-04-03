import os
import logging
import os.path as osp
import pycolmap

from pathlib import Path


def run_bundle_adjuster(deep_sfm_dir, ba_dir):
    logging.info("Running the bundle adjuster.")

    deep_sfm_model_dir = osp.join(deep_sfm_dir, 'model')
    reconstruction = pycolmap.Reconstruction(deep_sfm_model_dir)
    reconstruction = pycolmap.bundle_adjustment(reconstruction)
    reconstruction.write(ba_dir)


def main(deep_sfm_dir, ba_dir):
    assert Path(deep_sfm_dir).exists(), deep_sfm_dir
              
    Path(ba_dir).mkdir(parents=True, exist_ok=True)
    run_bundle_adjuster(deep_sfm_dir, ba_dir)