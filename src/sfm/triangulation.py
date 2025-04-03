import os
import h5py
import logging
import tqdm
import subprocess
import os.path as osp
import numpy as np
import open3d as o3d
import pycolmap
import sqlite3

from pathlib import Path
from src.utils.colmap.read_write_model import CAMERA_MODEL_NAMES, Image, read_cameras_binary, read_images_binary
from src.utils.colmap.database import COLMAPDatabase


def names_to_pair(name0, name1):
    return '_'.join((name0.replace('/', '-'), name1.replace('/', '-')))


def geometric_verification(database_path, pairs_path):
    """ Geometric verfication """
    logging.info('Performing geometric verification of the matches...')
    try:
        with open(pairs_path, 'r') as f:
            pairs = [line.strip().split() for line in f.readlines()]
    except FileNotFoundError:
        logging.warning(f'Could not find match list at: {pairs_path}')
        return

    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()

    # Map image names to image IDs
    cursor.execute("SELECT image_id, name FROM images")
    image_name_to_id = {name: image_id for image_id, name in cursor.fetchall()}

    inserted = 0
    for name1, name2 in pairs:
        id1 = image_name_to_id.get(name1)
        id2 = image_name_to_id.get(name2)

        if id1 is None or id2 is None:
            logging.warning(f"Image not found in DB: {name1}, {name2}")
            continue

        i1, i2 = sorted((id1, id2))
        pair_id = i1 * 2**32 + i2

        cursor.execute("""
            INSERT OR REPLACE INTO two_view_geometries
            (pair_id, rows, cols, data, config)
            VALUES (?, ?, ?, ?, ?)
        """, (pair_id, 0, 0, sqlite3.Binary(b''), 0))
        inserted += 1

    conn.commit()
    conn.close()


def create_db_from_model(empty_model, database_path):
    """ Create COLMAP database file from empty COLMAP binary file. """
    if database_path.exists():
        logging.warning('Database already exists.')
    
    cameras = read_cameras_binary(str(empty_model / 'cameras.bin'))
    images = read_images_binary(str(empty_model / 'images.bin'))

    db = COLMAPDatabase.connect(database_path)
    db.create_tables()
    
    for i, camera in cameras.items():
        model_id = CAMERA_MODEL_NAMES[camera.model].model_id
        db.add_camera(model_id, camera.width, camera.height, camera.params,
                      camera_id=i, prior_focal_length=True)
    
    for i, image in images.items():
        db.add_image(image.name, image.camera_id, image_id=i)
    
    db.commit()
    db.close()
    return {image.name: i for i, image in images.items()}


def import_features(image_ids, database_path, feature_path):
    """ Import keypoints info into COLMAP database. """
    logging.info("Importing features into the database...")
    feature_file = h5py.File(str(feature_path), 'r')
    db = COLMAPDatabase.connect(database_path)

    for image_name, image_id in tqdm.tqdm(image_ids.items()):
        keypoints = feature_file[image_name]['keypoints'].__array__()
        keypoints += 0.5
        db.add_keypoints(image_id, keypoints)
    
    feature_file.close()
    db.commit()
    db.close()


def import_matches(image_ids, database_path, pairs_path, matches_path, feature_path,
                   min_match_score=None, skip_geometric_verification=False):
    """ Import matches info into COLMAP database. """
    logging.info("Importing matches into the database...")

    with open(str(pairs_path), 'r') as f:
        pairs = [p.split(' ') for p in f.read().split('\n')]
    
    match_file = h5py.File(str(matches_path), 'r')
    db = COLMAPDatabase.connect(database_path)
    
    matched = set()
    for name0, name1 in tqdm.tqdm(pairs):
        id0, id1 = image_ids[name0], image_ids[name1]
        if len({(id0, id1), (id1, id0)} & matched) > 0:
            continue
    
        pair = names_to_pair(name0, name1)
        if pair not in match_file:
            raise ValueError(
                f'Could not find pair {(name0, name1)}... '
                'Maybe you matched with a different list of pairs? '
                f'Reverse in file: {names_to_pair(name0, name1) in match_file}.'
            )
        
        matches = match_file[pair]['matches0'].__array__()
        valid = matches > -1
        if min_match_score:
            scores = match_file[pair]['matching_scores0'].__array__()
            valid = valid & (scores > min_match_score)

        matches = np.stack([np.where(valid)[0], matches[valid]], -1)

        db.add_matches(id0, id1, matches)
        matched |= {(id0, id1), (id1, id0)}

        if skip_geometric_verification:
            db.add_two_view_geometry(id0, id1, matches)
    
    match_file.close()
    db.commit()
    db.close()


def run_triangulation(model_path, database_path, image_dir, empty_model):
    """ run triangulation on given database """
    logging.info('Running the triangulation...')

    recon = pycolmap.Reconstruction(empty_model)

    options = pycolmap.IncrementalMapperOptions()
    options.ba_refine_focal_length = False
    options.ba_refine_principal_point = False
    options.ba_refine_extra_params = False

    recon = pycolmap.triangulate_points(
        reconstruction=recon,
        database_path=str(database_path),
        image_path=str(image_dir),
        output_path=str(model_path),
        options=options
    )

    # Analyze the result
    stats = {
        'num_reg_images': len(recon.images),
        'num_sparse_points': len(recon.points3D),
        'num_observations': sum(len(p.track.elements) for p in recon.points3D.values()),
        'mean_track_length': sum(len(p.track.elements) for p in recon.points3D.values()) / len(recon.points3D) if recon.points3D else 0,
        'num_observations_per_image': (
            sum(len(p.track.elements) for p in recon.points3D.values()) / len(recon.images)
            if recon.images else 0
        )
    }

    print("Triangulation finished.")
    print(stats)
    return stats

def main(sfm_dir, empty_sfm_model, outputs_dir, pairs, features, matches, \
        skip_geometric_verification=True, min_match_score=None, image_dir=None):
    """ 
        Import keypoints, matches.
        Given keypoints and matches, reconstruct sparse model from given camera poses.
    """
    assert Path(empty_sfm_model).exists(), empty_sfm_model
    assert Path(features).exists(), features
    assert Path(pairs).exists(), pairs
    assert Path(matches).exists(), matches 

    Path(sfm_dir).mkdir(parents=True, exist_ok=True)
    database = osp.join(sfm_dir, 'database.db')
    model = osp.join(sfm_dir, 'model')
    Path(model).mkdir(exist_ok=True)

    image_ids = create_db_from_model(Path(empty_sfm_model), Path(database))
    import_features(image_ids, database, features)
    import_matches(image_ids, database, pairs, matches, features,
                   min_match_score, skip_geometric_verification)
    
    if not skip_geometric_verification:
        geometric_verification(database, pairs)
    
    if not image_dir:
        image_dir = '/'
    stats = run_triangulation(model, database, image_dir, empty_sfm_model)
    reconstruction = pycolmap.Reconstruction(model)
    save_reconstruction_to_ply(reconstruction, f'{outputs_dir}/model.ply')

def save_reconstruction_to_ply(recon, ply_path):
    points = []
    colors = []

    for point in recon.points3D.values():
        xyz = point.xyz
        rgb = [int(c) for c in point.color]  # Already in 0-255
        points.append(xyz)
        colors.append([c / 255.0 for c in rgb])  # Normalize to 0-1 for Open3D

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(ply_path, pcd)