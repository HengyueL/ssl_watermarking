import os
import numpy as np
import pandas as pd
import glob
import time
import shutil
import argparse



def watermark(input_dir, output_dir):
    
    model_path = 'models/dino_r50_plus.pth'
    normlayer_path = 'normlayers/out2048_yfcc_orig.pth'

    # for Encoder
    cmd = 'python main_multibit.py --data_dir {} --output_dir {} --model_path {} --normlayer_path {} --batch_size 32 --target_psnr 33 --num_bits 32'.format(input_dir, output_dir, model_path, normlayer_path)
    os.system(cmd)
   
    # # for Decoder
    # decoder_img_path = os.path.join(output_dir, 'imgs')
    # if not os.path.exists(decoder_img_path):
    #     os.makedirs(decoder_img_path)
    # src = input_dir
    # dst = os.path.join(decoder_img_path, input_dir.split('/')[-1])
    # shutil.copyfile(src, dst)

    
    # cmd = 'python main_multibit.py --decode_only True --data_dir {} --output_dir {} --model_path {} --normlayer_path {} --batch_size 1  --num_bits 32'.format(decoder_img_path, decoder_img_path, model_path, normlayer_path)
    # os.system(cmd)


def main(args):
    dataset_dir = os.path.join(
        "dataset_clean", "Clean"
    )
    output_root = os.path.join(
        ".", "dataset_processed", "SSL"
    )


    input_dir = os.path.join(dataset_dir, args.dataset)
    output_dir = os.path.join(
        output_root, args.dataset,
    )
    os.makedirs(output_dir, exist_ok=True)
    watermark(input_dir, output_dir)
  
   
if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description='Some arguments to play with.')
    parser.add_argument(
        "--dataset", dest="dataset", type=str,
        default="COCO"
    )
    args = parser.parse_args()
    main(args)

    print()
    print("Completed.")
    

    

 