"""
Test a trained vae
"""
import argparse
import os

import cv2
import numpy as np
from stable_baselines.common import set_global_seeds

from vae.data_loader import get_image_augmenter
from vae.controller import VAEController

from config import ROI, SIZE_Z, PATH_MODEL, PATH_MODEL_BEST

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--folder', help='Log folder', type=str, default='vae/data/rgb/') # wyb
parser.add_argument('-vae', '--vae-path', help='Path to saved VAE', type=str, default=PATH_MODEL_BEST)
parser.add_argument('-n', '--n-samples', help='Max number of samples', type=int, default=50)
parser.add_argument('--seed', help='Random generator seed', type=int, default=0)
parser.add_argument('-augment', '--augment', action='store_true', default=False,
                    help='Use image augmenter')
args = parser.parse_args()

set_global_seeds(args.seed)

if not args.folder.endswith('/'):
    args.folder += '/'

vae = VAEController()
vae.load(args.vae_path)

images = [im for im in os.listdir(args.folder) if im.endswith('.png')] # wyb
images = np.array(images)
n_samples = len(images)

augmenter = None
if args.augment:
    augmenter = get_image_augmenter()

for i in range(args.n_samples):
    # Load test image
    image_idx = np.random.randint(n_samples)
    image_path = args.folder + images[image_idx]
    image = cv2.imread(image_path)

    if augmenter is not None:
        input_image = augmenter.augment_image(image)
    else:
        input_image = image

    encoded = vae.encode_from_raw_image(input_image)
    reconstructed_image = vae.decode(encoded)[0]
    # Plot reconstruction
    cv2.imshow("Original", image)

    if augmenter is not None:
        cv2.imshow("Augmented", input_image)

    cv2.imshow("Reconstruction", reconstructed_image)
    cv2.waitKey(0)
