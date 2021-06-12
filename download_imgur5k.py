'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
IMGUR5K is shared as a set of image urls with annotations. This code downloads
th images and verifies the hash to the image to avoid data contamination.

Usage:
      python downloaad_imgur5k.py --dataset_info_dir <dir_with_annotaion_and_hashes> --output_dir <path_to_store_images>

Output:
     Images dowloaded to output_dir
     data_annotations.json : json file with image annotation mappings -> dowloaded to dataset_info_dir
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

import argparse
import hashlib
import json
import multiprocessing as mp
import numpy as np
import os
import requests

from concurrent import futures
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description="Processing imgur5K dataset download...")
    parser.add_argument(
        "--dataset_info_dir",
        type=str,
        default="dataset_info",
        required=False,
        help="Directory with dataset information",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="images",
        required=False,
        help="Directory path to download the image",
    )
    args = parser.parse_args()
    return args


# Image hash computed for image using md5..
def compute_image_hash(img_path):
    return hashlib.md5(open(img_path, 'rb').read()).hexdigest()

# Create a sub json based on split idx
def _create_split_json(anno_json, _split_idx):

    split_json = {}

    split_json['index_id'] = {}
    split_json['index_to_ann_map'] = {}
    split_json['ann_id'] = {}

    for _idx in _split_idx:
        # Check if the idx is not bad
        if _idx not in anno_json['index_id']:
            continue

        split_json['index_id'][_idx] = anno_json['index_id'][_idx]
        split_json['index_to_ann_map'][_idx] = anno_json['index_to_ann_map'][_idx]

        for ann_id in split_json['index_to_ann_map'][_idx]:
            split_json['ann_id'][ann_id] = anno_json['ann_id'][ann_id]

    return split_json


def partial_func(index, output_dir):
    """
    Partial function for retrieving and saving image content.

    Image content is comparing against the expected hash.
    If image is invalid, or hash does not match, invalid url will be returned.
    :param index: index of image to download
    :param output_dir: directory to write successful images
    :return: invalid_url or None
    """
    image_url = f'https://i.imgur.com/{index}.jpg'
    img_data = requests.get(image_url).content
    if len(img_data) < 100:
        return image_url

    with open(f'{output_dir}/{index}.jpg', 'wb') as handler:
        handler.write(img_data)

    compute_image_hash(f'{output_dir}/{index}.jpg')
    if hash_dict[index] != compute_image_hash(f'{output_dir}/{index}.jpg'):
        print(
            f"For IMG: {index}, ref hash: {hash_dict[index]} != cur hash: {compute_image_hash(f'{output_dir}/{index}.jpg')}")
        os.remove(f'{args.output_dir}/{index}.jpg')
        return image_url
    else:
        return None

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Create a hash dictionary with image index and its correspond gt hash
    with open(f"{args.dataset_info_dir}/imgur5k_hashes.lst", "r", encoding="utf-8") as _H:
        hashes = _H.readlines()
        hash_dict = {}

        for hash in hashes:
            hash_dict[f"{hash.split()[0]}"] = f"{hash.split()[1]}"


    tot_evals = 0
    num_match = 0
    invalid_urls = []

    executor = futures.ThreadPoolExecutor(max(mp.cpu_count() - 1, 1))
    # Download the urls and save only the ones with valid hash o ensure underlying image has not changed
    invalid_urls = list(executor.map(partial_func, [(key, args.output_dir) for key in list(hash_dict.keys())]))

    for result in invalid_urls:
        tot_evals +=1
        if result is None:
            num_match += 1



    # Generate the final annotations file
    # Format: { "index_id" : {indexes}, "index_to_annotation_map" : { annotations ids for an index}, "annotation_id": { each annotation's info } }
    # Bounding boxes with '.' mean the annotations were not done for various reasons

    _F = np.loadtxt(f'{args.dataset_info_dir}/imgur5k_data.lst', delimiter="\t", dtype=np.str)
    anno_json = {}

    anno_json['index_id'] = {}
    anno_json['index_to_ann_map'] = {}
    anno_json['ann_id'] = {}

    cur_index = ''
    for cnt, image_url in enumerate(_F[:,0]):
        if image_url in invalid_urls:
            continue

        index = image_url.split('/')[-1][:-4]
        if index != cur_index:
            anno_json['index_id'][index] = {'image_url': image_url, 'image_path': f'{args.output_dir}/{index}.jpg', 'image_hash': hash_dict[index]}
            anno_json['index_to_ann_map'][index] = []

        ann_id = f"{index}_{len(anno_json['index_to_ann_map'][index])}"
        anno_json['index_to_ann_map'][index].append(ann_id)
        anno_json['ann_id'][ann_id] = {'word': _F[cnt,2], 'bounding_box': _F[cnt,1]}

        cur_index = index

    json.dump(anno_json, open(f'{args.dataset_info_dir}/imgur5k_annotations.json', 'w'), indent=4)

    # Now split the annotations json in train, validation and test jsons
    splits = ['train', 'val', 'test']
    for split in splits:
        _split_idx = np.loadtxt(f'{args.dataset_info_dir}/{split}_index_ids.lst', delimiter="\n", dtype=np.str)
        split_json = _create_split_json(anno_json, _split_idx)
        json.dump(split_json, open(f'{args.dataset_info_dir}/imgur5k_annotations_{split}.json', 'w'), indent=4)

    print(f"MATCHES: {num_match}/{tot_evals}\n")

if __name__ == '__main__':
    main()
