from pyexpat import features

import spatialdata as sd
import numpy as np
import pandas as pd
import cv2 as cv
from huggingface_hub import login
from skimage.segmentation import expand_labels
import skimage
import torch
import timm
from torchvision import transforms
import pickle
from tqdm import tqdm


def extract_patches_from_cell_nucleus(HE_image, HE_nuc_image, patch_size=(224, 224)):
    HE_nuc_expand = expand_labels(HE_nuc_image, distance=10)

    props = skimage.measure.regionprops(HE_nuc_expand, intensity_image=HE_image)
    patch_height, patch_width = patch_size
    patches, labels = [], []

    for prop in tqdm(props):

        label = prop.label
        loc_x, loc_y = prop.centroid

        center_x, center_y = np.rint(loc_x).astype(np.int32), np.rint(loc_y).astype(np.int32)

        half_height = patch_height // 2
        half_width = patch_width // 2

        if center_x - half_width < 0:
            start_x = 0
            end_x = patch_width
        elif center_x + half_width > image.shape[1]:
            start_x = image.shape[1] - patch_width
            end_x = image.shape[1]
        else:
            start_x = center_x - half_width
            end_x = center_x + half_width

        if center_y - half_height < 0:
            start_y = 0
            end_y = patch_height
        elif center_y + half_height > image.shape[0]:
            start_y = image.shape[0] - patch_height
            end_y = image.shape[0]
        else:
            start_y = center_y - half_height
            end_y = center_y + half_height

        # Extract the patch
        patch = HE_image[start_y:end_y, start_x:end_x]

        patches.append(patch)
        labels.append(label)

    return patches, labels


def extract_patches_emb_from_cell_nucleus(HE_image, HE_nuc_image, patch_size=(224, 224)):
    HE_nuc_expand = expand_labels(HE_nuc_image, distance=10)
    props = skimage.measure.regionprops(HE_nuc_expand, intensity_image=HE_image)
    patch_height, patch_width = patch_size

    features = torch.empty((0, 1536))
    labels = []

    device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
    model = timm.create_model(
        "hf-hub:bioptimus/H-optimus-0", pretrained=True, init_values=1e-5, dynamic_img_size=False
    )
    model.to(device)
    model.eval()

    batch_size = 1024
    transform = transforms.Compose([
        # transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.707223, 0.578729, 0.703617),
            std=(0.211883, 0.230117, 0.177517)
        ),
    ])

    # We recommend using mixed precision for faster prediction.
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        with torch.inference_mode():

            patches = []

            for i, prop in tqdm(enumerate(props)):

                label = prop.label
                loc_x, loc_y = prop.centroid

                center_x, center_y = np.rint(loc_x).astype(np.int32), np.rint(loc_y).astype(np.int32)

                half_height = patch_height // 2
                half_width = patch_width // 2

                if center_x - half_width < 0:
                    start_x = 0
                    end_x = patch_width
                elif center_x + half_width > image.shape[1]:
                    start_x = image.shape[1] - patch_width
                    end_x = image.shape[1]
                else:
                    start_x = center_x - half_width
                    end_x = center_x + half_width

                if center_y - half_height < 0:
                    start_y = 0
                    end_y = patch_height
                elif center_y + half_height > image.shape[0]:
                    start_y = image.shape[0] - patch_height
                    end_y = image.shape[0]
                else:
                    start_y = center_y - half_height
                    end_y = center_y + half_height

                # Extract the patch
                patch = HE_image[start_y:end_y, start_x:end_x]
                patches.append(patch)
                labels.append(label)

                if (i + 1) % batch_size == 0 or (i + batch_size) // len(props) > 0:
                    input = torch.tensor(np.array(patches), dtype=torch.float16).permute(0, 3, 1, 2)
                    feature = model(transform(input).to(device))
                    features = torch.cat([features, feature.to("cpu")], dim=0)
                    patches = []

    return features, labels

def extract_patches_emb_from_cell_centroid(HE_image, centroid, patch_size=(224, 224)):

    patch_height, patch_width = patch_size

    features = torch.empty((0, 1536))

    device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
    model = timm.create_model(
        "hf-hub:bioptimus/H-optimus-0", pretrained=True, init_values=1e-5, dynamic_img_size=False
    )
    model.to(device)
    model.eval()

    batch_size = 1024
    transform = transforms.Compose([
        # transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.707223, 0.578729, 0.703617),
            std=(0.211883, 0.230117, 0.177517)
        ),
    ])

    # We recommend using mixed precision for faster prediction.
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        with torch.inference_mode():

            patches = []

            for i, in tqdm(range(centroid.shape[0])):

                loc_x, loc_y = centroid[i, 0], centroid[i, 1]

                center_x, center_y = np.rint(loc_x).astype(np.int32), np.rint(loc_y).astype(np.int32)

                half_height = patch_height // 2
                half_width = patch_width // 2

                if center_x - half_width < 0:
                    start_x = 0
                    end_x = patch_width
                elif center_x + half_width > image.shape[1]:
                    start_x = image.shape[1] - patch_width
                    end_x = image.shape[1]
                else:
                    start_x = center_x - half_width
                    end_x = center_x + half_width

                if center_y - half_height < 0:
                    start_y = 0
                    end_y = patch_height
                elif center_y + half_height > image.shape[0]:
                    start_y = image.shape[0] - patch_height
                    end_y = image.shape[0]
                else:
                    start_y = center_y - half_height
                    end_y = center_y + half_height

                # Extract the patch
                patch = HE_image[start_y:end_y, start_x:end_x]
                patches.append(patch)

                if (i + 1) % batch_size == 0 or (i + batch_size) // centroid.shape[0] > 0:
                    input = torch.tensor(np.array(patches), dtype=torch.float16).permute(0, 3, 1, 2)
                    feature = model(transform(input).to(device))
                    features = torch.cat([features, feature.to("cpu")], dim=0)
                    patches = []

    return features

if __name__ == '__main__':

    samples = ['UC1_Infl', 'UC6_Infl', 'UC7_Infl', 'DC1',
              'DC5', 'UC1_NonInfl', 'UC6_NonInfl', 'UC9_Infl']

    for sample in samples:
        sdata = sd.read_zarr(f"../data/{sample}-train-final.zarr")
        print(f"Processing dataset {sample}")

        nuc_image = sdata['HE_nuc_original'].to_numpy()[0]
        image = sdata['HE_original'].to_numpy().transpose(1, 2, 0)

        cell_id_max = np.max(sdata['cell_id-group'].obs['cell_id'])

        print(f"Cell number: {cell_id_max}")

        # tensor [cell_num, 224, 224]
        features, labels = extract_patches_emb_from_cell_nucleus(image, nuc_image, (224, 224))

        with open(f'../data/feature_{sample}.pkl', 'wb') as file:
            pickle.dump(features, file)

        with open(f'../data/label_{sample}.pkl', 'wb') as file:
            pickle.dump(labels, file)

        # with open(f'../data/image_+{sample}.pkl', 'wb') as file:
        #     pickle.dump(images, file)

