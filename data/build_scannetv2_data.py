import argparse
import logging
import math
import glob
import os
import re
import shutil
import tarfile

from typing import List, Tuple, Optional

import numpy as np
import tensorflow as tf
from PIL import Image

from deeplab2.data import data_utils

logging.basicConfig(level=logging.INFO)


_PANOPTIC_LABEL_FORMAT = "raw"
_TF_RECORD_PATTERN = "%s-%05d-of-%05d.tfrecord"
_IMAGES_DIR_NAME = "color"
_PANOPTIC_MAPS_DIR_NAME = "panoptic"
_SCANS_SEARCH_PATTERN = "scene*"
_NUM_FRAMES_SEARCH_PATTERN = r"numColorFrames\s*=\s*(?P<num_frames>\w*)"


def _load_image(image_file_path: str):
    with open(image_file_path, "rb") as f:
        image_data = f.read()
    return image_data


def _load_panoptic_map(panoptic_map_path: str) -> Optional[str]:
    """Decodes the panoptic map from encoded image file.

    Args:
      panoptic_map_path: Path to the panoptic map image file.

    Returns:
      Panoptic map as an encoded int32 numpy array bytes or None if not existing.
    """
    with open(panoptic_map_path, "rb") as f:
        panoptic_map = np.array(Image.open(f)).astype(np.int32)
    return panoptic_map.tobytes()


def _extract_tar_archive(tar_archive_path: str):
    tar_archive = tarfile.open(tar_archive_path, "r:gz")
    extract_dir_path = os.path.dirname(tar_archive_path)
    tar_archive.extractall(
        path=extract_dir_path,
    )


def _load_scan_ids_from_file(scan_ids_file_path: str) -> List[str]:
    with open(scan_ids_file_path, "r") as f:
        scan_ids = f.readlines()
    return [s.rstrip('\n') for s in scan_ids]


def _find_scans(scans_root_dir: str) -> List[str]:
    scan_dirs = glob.glob(os.path.join(scans_root_dir, _SCANS_SEARCH_PATTERN))
    # TODO: use a regex to match the dirname with pattern scene[0-9]{4})_[0-9]{2}
    return [
        os.path.basename(scan_dir)
        for scan_dir in scan_dirs
        if tf.io.gfile.isdir(scan_dir)
    ]


def _get_image_info_from_path(image_path: str) -> Tuple[str, str]:
    """Gets image info including sequence id and image id.

    Image path is in the format of '.../scan_id/color/image_id.png',
    where `scan_id` refers to the id of the video sequence, and `image_id` is
    the id of the image in the video sequence.

    Args:
      image_path: Absolute path of the image.

    Returns:
      sequence_id, and image_id as strings.
    """
    scan_id = os.path.basename(os.path.dirname(os.path.dirname(image_path)))
    image_id = os.path.splitext(os.path.basename(image_path))[0]
    return scan_id, image_id


def _remove_dirs(dir_paths: List[str]):
    for dir_path in dir_paths:
        shutil.rmtree(dir_path)


def _compute_total_number_of_frames(
    scans_root_dir_path: str,
    scan_ids: int,
) -> int:
    cnt = 0
    for scan_id in scan_ids:
        scan_info_file_path = os.path.join(
            scans_root_dir_path, scan_id, f"{scan_id}.txt"
        )
        try:
            with open(scan_info_file_path, "r") as f:
                matches = re.findall(_NUM_FRAMES_SEARCH_PATTERN, f.read(), re.MULTILINE)
                cnt += int(matches[0])
        except FileNotFoundError:
            logging.error(f"Scan info file missing in {scan_id}!")
            continue
    return cnt


def _get_color_and_panoptic_per_shard(
    scans_root_dir_path: str,
    scan_ids: List[str],
    num_shards: int,
):
    num_frames = _compute_total_number_of_frames(scans_root_dir_path, scan_ids)

    num_examples_per_shard = math.ceil(math.ceil(num_frames / num_shards))

    color_and_panoptic_per_shard = []
    dirs_to_remove = []
    for i, scan_id in enumerate(scan_ids):
        scan_dir_path = os.path.join(scans_root_dir_path, scan_id)
        images_archive_path = os.path.join(scan_dir_path, f"{_IMAGES_DIR_NAME}.tar.gz")
        try:
            _extract_tar_archive(images_archive_path)
        except FileNotFoundError:
            logging.warning(f"{images_archive_path} not found. {scan_id} skipped.")
            continue
        images_dir_path = os.path.join(scan_dir_path, _IMAGES_DIR_NAME)

        panoptic_maps_archive_path = os.path.join(
            scan_dir_path, f"{_PANOPTIC_MAPS_DIR_NAME}.tar.gz"
        )
        try:
            _extract_tar_archive(panoptic_maps_archive_path)
        except FileNotFoundError:
            logging.warning(
                f"{panoptic_maps_archive_path} not found. {scan_id} skipped."
            )
            shutil.rmtree(str(images_dir_path))
            continue
        panoptic_maps_dir_path = os.path.join(scan_dir_path, _PANOPTIC_MAPS_DIR_NAME)

        image_file_paths = list(glob.glob(os.path.join(images_dir_path, "*.jpg")))
        for j, image_file_path in enumerate(image_file_paths):
            image_file_name = os.path.splitext(os.path.basename(image_file_path))[0]
            panoptic_map_file_path = os.path.join(
                panoptic_maps_dir_path,
                (re.sub(r"0+(.+)", r"\1", image_file_name) + ".png"),
            )

            color_and_panoptic_per_shard.append(
                (image_file_path, panoptic_map_file_path)
            )

            shard_data = len(
                color_and_panoptic_per_shard
            ) == num_examples_per_shard or (
                # Last image of the last scan in the list
                j == len(image_file_paths)
                and i == len(scan_ids)
            )
            if shard_data:
                yield color_and_panoptic_per_shard
                color_and_panoptic_per_shard = []
                # Remove all the directories that can be removed
                _remove_dirs(dirs_to_remove)
                dirs_to_remove = []

        dirs_to_remove.append(images_dir_path)
        dirs_to_remove.append(panoptic_maps_dir_path)

    # Clean up the last dirs
    _remove_dirs(dirs_to_remove)


def _create_panoptic_tfexample(
    image_path: str,
    panoptic_map_path: str,
) -> tf.train.Example:
    """Creates a TF example for each image.

    Args:
      image_path: Path to the image.
      panoptic_map_path: Path to the panoptic map (as an image file).

    Returns:
      TF example proto.
    """
    image_data = _load_image(image_path)
    label_data = _load_panoptic_map(panoptic_map_path)
    image_name = os.path.basename(image_path)
    image_format = os.path.splitext(image_name)[1].lstrip(".").lower()
    sequence_id, frame_id = _get_image_info_from_path(image_path)
    return data_utils.create_video_tfexample(
        image_data,
        image_format,
        image_name,
        label_format=_PANOPTIC_LABEL_FORMAT,
        sequence_id=sequence_id,
        image_id=frame_id,
        label_data=label_data,
        prev_image_data=None,
        prev_label_data=None,
    )


def _create_tf_record_dataset(
    scans_root_dir_path: str,
    dataset_tag: str,
    output_dir_path: str,
    num_shards: int,
    scan_ids_file_path: Optional[str],
):
    assert tf.io.gfile.isdir(scans_root_dir_path)

    tf.io.gfile.makedirs(output_dir_path)

    scan_ids = _find_scans(scans_root_dir_path)
    if scan_ids_file_path is not None:
        scan_ids = list(
            set(scan_ids) & set(_load_scan_ids_from_file(scan_ids_file_path))
        )

    if len(scan_ids) == 0:
        logging.error("No scans found!")
        exit(1)

    color_and_panoptic_per_shard = _get_color_and_panoptic_per_shard(
        scans_root_dir_path=scans_root_dir_path,
        scan_ids=scan_ids,
        num_shards=num_shards,
    )

    for shard_id, example_list in enumerate(color_and_panoptic_per_shard):
        shard_filename = _TF_RECORD_PATTERN % (dataset_tag, shard_id, num_shards)
        shard_file_path = os.path.join(output_dir_path, shard_filename)
        with tf.io.TFRecordWriter(shard_file_path) as tfrecord_writer:
            for image_path, panoptic_map_path in example_list:
                try:
                    example = _create_panoptic_tfexample(
                        image_path,
                        panoptic_map_path,
                    )
                except FileNotFoundError:
                    continue
                tfrecord_writer.write(example.SerializeToString())


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Converts scans from the ScanNetV2 dataset to TFRecord",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-sd",
        "--scans_root_dir_path",
        type=str,
        required=True,
        help="Scans root directory.",
    )

    parser.add_argument(
        "-o",
        "--output_dir_path",
        type=str,
        required=True,
        help="Path to save converted TFRecord of TensorFlow examples.",
    )

    parser.add_argument(
        "-t",
        "--dataset_tag",
        type=str,
        required=True,
        help="Dataset tag. All the shards will be named as ...",
    )

    parser.add_argument(
        "-ids",
        "--scan_ids_file_path",
        type=str,
        help="Path to a text file with the ids of the scans to consider.",
    )

    parser.add_argument(
        "-ns",
        "--num_shards",
        type=int,
        default=1000,
        help="Number of shards.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    _create_tf_record_dataset(
        scans_root_dir_path=args.scans_root_dir_path,
        dataset_tag=args.dataset_tag,
        output_dir_path=args.output_dir_path,
        scan_ids_file_path=args.scan_ids_file_path,
        num_shards=args.num_shards,
    )
