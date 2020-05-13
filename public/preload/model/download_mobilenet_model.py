'''
Tools to download tensorflow.js models

Authot : iascchen
'''

import os
import requests
import json

BASE_FOLDER = "."

# MOBILENET
folders = [
    "mobilenet/mobilenet_v1_0.25_224",
    "mobilenet/mobilenet_v1_0.50_224",
    "mobilenet/mobilenet_v1_0.75_224",
    "mobilenet/mobilenet_v1_1.0_224",
]

models_json = [
    "https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224",
    "https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.50_224",
    "https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.75_224",
    "https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_1.0_224",
]

models_fn = "model.json"


def get_remote_target(folder, file_name):
    return "%s/%s" % (folder, file_name)


def get_target(folder, file_name=None):
    if file_name is None:
        return "%s/%s" % (BASE_FOLDER, folder)
    else:
        return "%s/%s" % (folder, file_name)


def download_model(remote_target, target):
    target_folder = get_target(target)
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    target_manifest = get_target(target_folder, models_fn)
    manifest_url = get_remote_target(remote_target, models_fn)
    download_file(manifest_url, target_manifest)
    file_list = parse_model_json(target_manifest)
    for file_name in file_list:
        target_file = get_target(target_folder, file_name)
        remote_url = get_remote_target(remote_target, file_name)
        download_file(remote_url, target_file)


def parse_model_json(josn_file):
    files = []

    print("Parsing ==>", josn_file)
    json_data = open(josn_file).read()
    data = json.loads(json_data)

    weights_manifest = data["weightsManifest"]

    for value in weights_manifest:
        # d.itervalues: an iterator over the values of d
        print("paths ==> ", value["paths"])
        # files.append(value["paths"])
        files = files + value["paths"]

    return files


def download_file(url, target):
    print("Downloading ==>", url, target)
    r = requests.get(url)
    f = open(target, 'wb')
    for chunk in r.iter_content(chunk_size=512 * 1024):
        if chunk:  # filter out keep-alive new chunks
            f.write(chunk)
    f.close()
    return


for idx, val in enumerate(models_json):
    print("Begin ==>", val)
    download_model(val, folders[idx])
    print("End ==>")
