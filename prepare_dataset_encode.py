# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os

import numpy as np
import torch
from torchvision.transforms import ToPILImage

import data_augmentation
import encode
import evaluate
import utils
import utils_img
import decode
import pandas as pd
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def watermark_np_to_str(watermark_np):
    """
        Convert a watermark in np format into a str to display.
    """
    return "".join([str(i) for i in watermark_np.tolist()])


def get_parser():
    parser = argparse.ArgumentParser()

    def aa(*args, **kwargs):
        group.add_argument(*args, **kwargs)

    group = parser.add_argument_group('Experiments parameters')
    aa("--data_dir", type=str, default="input/", help="Folder directory (Default: /input)")
    aa("--carrier_dir", type=str, default="carriers/", help="Directions of the latent space in which the watermark is embedded (Default: /carriers)")
    aa("--output_dir", type=str, default="output/", help="Output directory for logs and images (Default: /output)")
    aa("--save_images", type=utils.bool_inst, default=True, help="Whether to save watermarked images (Default: False)")
    aa("--evaluate", type=utils.bool_inst, default=False, help="Whether to evaluate the detector (Default: True)")
    aa("--decode_only", type=utils.bool_inst, default=False, help="To decode only watermarked images (Default: False)")
    aa("--verbose", type=int, default=1)

    group = parser.add_argument_group('Messages parameters')
    aa("--msg_type", type=str, default='bit', choices=['text', 'bit'], help="Type of message (Default: bit)")
    aa("--msg_path", type=str, default=None, help="Path to the messages text file (Default: None)")
    aa("--num_bits", type=int, default=32, help="Number of bits of the message. (Default: None)")

    group = parser.add_argument_group('Marking parameters')
    aa("--target_psnr", type=float, default=42.0, help="Target PSNR value in dB. (Default: 42 dB)")
    aa("--target_fpr", type=float, default=1e-6, help="Target FPR of the dectector. (Default: 1e-6)")

    group = parser.add_argument_group('Neural-Network parameters')
    aa("--model_name", type=str, default='resnet50', help="Marking network architecture. See https://pytorch.org/vision/stable/models.html and https://rwightman.github.io/pytorch-image-models/models/ (Default: resnet50)")
    aa("--model_path", type=str, default="models/dino_r50_plus.pth", help="Path to the model (Default: /models/dino_r50_plus.pth)")
    aa("--normlayer_path", type=str, default="normlayers/out2048_yfcc_orig.pth", help="Path to the normalization layer (Default: /normlayers/out2048.pth)")

    group = parser.add_argument_group('Optimization parameters')
    aa("--epochs", type=int, default=100, help="Number of epochs for image optimization. (Default: 100)")
    aa("--data_augmentation", type=str, default="all", choices=["none", "all"], help="Type of data augmentation to use at marking time. (Default: All)")
    aa("--optimizer", type=str, default="Adam,lr=0.01", help="Optimizer to use. (Default: Adam,lr=0.01)")
    aa("--scheduler", type=str, default=None, help="Scheduler to use. (Default: None)")
    aa("--batch_size", type=int, default=1, help="Batch size for marking. (Default: 128)")
    aa("--lambda_w", type=float, default=5e4, help="Weight of the watermark loss. (Default: 1.0)")
    aa("--lambda_i", type=float, default=1.0, help="Weight of the image loss. (Default: 1.0)")

    return parser


def main(params):
    # Set seeds for reproductibility
    torch.manual_seed(0)
    np.random.seed(0)

    dataset_dir = os.path.join(
        "dataset", "Clean"
    )
    output_root = os.path.join(
        "dataset_processed", "SSL"
    )
    args = params
    input_dir = os.path.join(dataset_dir, args.dataset)
    output_dir = os.path.join(
        output_root, args.dataset
    )
    os.makedirs(output_dir, exist_ok=True)
    output_img_dir = os.path.join(output_dir, "encoder_img")
    os.makedirs(output_img_dir, exist_ok=True)

    # === Scan all images in the dataset (clean) ===
    img_files = [f for f in os.listdir(input_dir) if ".png" in f]
    print("Total number of images: [{}] --- (Should be 100)".format(len(img_files)))


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

    msgs = torch.rand([1, params.num_bits])>0.5
    print("Msg Generated: ", msgs)
    msg_encoded = watermark_np_to_str(np.where(msgs[0].cpu().numpy(), 1, 0))
    print("Encode msg: ", msg_encoded)
    
    # Construct data augmentation
    data_aug = data_augmentation.DifferentiableDataAugmentation()

    save_csv_dir = os.path.join(output_dir, "water_mark.csv")
    res_dict = {
        "ImageName": [],
        "Encoder": [],
        "Decoder": [],
        "Match": []
    }
    
    # Marking
    for file_name in img_files:
        img_path = os.path.join(input_dir, file_name)
        print('>>> Marking image {}'.format(img_path))

        img_pil = Image.open(img_path).convert("RGB")
        pt_imgs_out = encode.watermark_multibit_single_img(
            img_pil, msgs, carrier, model, data_aug, params
        )
        
        img_out = ToPILImage()(utils_img.unnormalize_img(pt_imgs_out))
        save_img_name = file_name
        img_out.save(os.path.join(output_img_dir, save_img_name))

        # Decode Image
        decoded_data = decode.decode_multibit([img_out], carrier, model)[0]
        msg_decoded = watermark_np_to_str(np.where(decoded_data["msg"].cpu().numpy(), 1, 0))
        
        print("Decode msg: ", msg_decoded)

        res_dict["ImageName"].append(file_name)
        res_dict["Encoder"].append([msg_encoded])
        res_dict["Decoder"].append([msg_decoded])
        res_dict["Match"].append(msg_encoded == msg_decoded)

    df = pd.DataFrame(res_dict)
    df.to_csv(save_csv_dir, index=False)


if __name__ == '__main__':

    # generate parser / parse parameters
    parser = get_parser()
    parser.add_argument(
        "--dataset", dest="dataset", type=str,
        default="COCO"
    )
    params = parser.parse_args()

    # run experiment
    main(params)
