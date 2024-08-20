"""
    This script is use dwtDctSvd/rivaGan to decode clean images to compute FPR.
"""
import sys, os
dir_path = os.path.abspath(".")
sys.path.append(dir_path)
dir_path = os.path.abspath("..")
sys.path.append(dir_path)
import os, argparse, torch
import numpy as np
import pandas as pd
from general import rgb2bgr, save_image_bgr, set_random_seeds, \
    watermark_np_to_str, watermark_str_to_numpy
from prepare_dataset_encode import get_parser
import utils
from PIL import Image
import decode


def main(args):
    # === Some dummt configs ===
    device = torch.device("cuda")
    params = args
    set_random_seeds(args.random_seed)

    # === Set the dataset paths ===
    dataset_input_path = os.path.join(
        args.clean_data_root, args.dataset
    )
    # === Scan all images in the dataset (clean) ===
    img_files = [f for f in os.listdir(dataset_input_path) if ".png" in f]
    print("Total number of images: [{}]".format(len(img_files)))

    output_root_path = os.path.join(
        ".", "dataset", "Clean_Watermark_Evasion", args.watermarker, args.dataset
    )
    os.makedirs(output_root_path, exist_ok=True)

    # === GT waternark ===
    print("Watermarker: ", args.watermarker)
    watermarked_file = os.path.join(".", "dataset", args.watermarker, args.dataset, "water_mark.csv")
    watermarked_data = pd.read_csv(watermarked_file)
    watermark_gt_str = watermarked_data.iloc[0]["Encoder"]
    if watermark_gt_str[0] == "[":  # Some historical none distructive bug :( will cause this reformatting
        watermark_gt_str = eval(watermark_gt_str)[0]
    watermark_gt = watermark_str_to_numpy(watermark_gt_str)

    # === Init watermarker ===
    # Loads backbone and normalization layer
    if params.verbose > 0:
        print('>>> Building backbone and normalization layer...')
    backbone = utils.build_backbone(path=params.model_path, name=params.model_name)
    normlayer = utils.load_normalization_layer(path=params.normlayer_path)
    model = utils.NormLayerWrapper(backbone, normlayer)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()

    # Load or generate carrier and angle
    if not os.path.exists(params.carrier_dir):
        os.makedirs(params.carrier_dir, exist_ok=True)
    D = model(torch.zeros((1,3,224,224)).to(device)).size(-1)
    K = params.num_bits
    carrier_path = os.path.join(params.carrier_dir,'carrier_%i_%i.pth'%(K, D))
    if os.path.exists(carrier_path):
        if params.verbose > 0:
            print('>>> Loading carrier from %s' % carrier_path)
        carrier = torch.load(carrier_path)
        assert D == carrier.shape[1]
    else:
        if params.verbose > 0:
            print('>>> Generating carrier into %s... (can take up to a minute)' % carrier_path)
        carrier = utils.generate_carriers(K, D, output_fpath=carrier_path)
    carrier = carrier.to(device, non_blocking=True) # direction vectors of the hyperspace

    # Init the dict to save watermarking summary
    save_csv_dir = os.path.join(output_root_path, "water_mark.csv")
    res_dict = {
        "ImageName": [],
        "Decoder": [],
    }

    for img_name in img_files:
        img_clean_path = os.path.join(dataset_input_path, img_name)
        print("***** ***** ***** *****")
        print("Processing Image: {} ...".format(img_clean_path))

        # === Decode ===
        img_pil = Image.open(img_clean_path ).convert("RGB")

        decoded_data = decode.decode_multibit([img_pil], carrier, model)[0]
        watermark_decode_str = watermark_np_to_str(np.where(decoded_data["msg"].cpu().numpy(), 1, 0))

        res_dict["ImageName"].append(img_name)
        res_dict["Decoder"].append([watermark_decode_str])

    df = pd.DataFrame(res_dict)
    df.to_csv(save_csv_dir, index=False)


if __name__ == "__main__":
    print("Use this script to download DiffusionDB dataset.")
    parser = get_parser()
    parser.add_argument(
        "--random_seed", dest="random_seed", type=int, help="Manually set random seed for reproduction.",
        default=13
    )
    parser.add_argument(
        '--clean_data_root', type=str, help="Root dir where the clean image dataset is located.",
        default=os.path.join(".", "dataset", "Clean")
    )
    parser.add_argument(
        "--dataset", dest="dataset", type=str, help="The dataset name: [COCO, DiffusionDB]",
        default="COCO"
    )
    parser.add_argument(
        "--watermarker", dest="watermarker", type=str, help="Specification of watermarking method. ['dwtDctSvd', 'rivaGan']",
        default="SSL"
    )
    args = parser.parse_args()
    main(args)
    print("Completed")