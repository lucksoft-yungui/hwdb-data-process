import os
import argparse
import zipfile
import struct
import numpy as np
from PIL import Image


def read_from_gnt_file(gnt_file):
    header_size = 10
    while True:
        header = np.frombuffer(gnt_file.read(header_size), dtype='uint8')
        if not header.size: break
        sample_size = header[0] + (header[1] << 8) + (header[2] << 16) + (header[3] << 24)
        tag_code = header[5] + (header[4] << 8)
        width = header[6] + (header[7] << 8)
        height = header[8] + (header[9] << 8)
        if header_size + width * height != sample_size: break
        image = np.frombuffer(gnt_file.read(width * height), dtype='uint8').reshape((height, width))
        yield image, str(tag_code)  # use decimal tag_code as directory name


def handle_gnt_file(gnt_file, output_dir, img_size):
    for img, tagcode_str in read_from_gnt_file(gnt_file):
        char_folder = os.path.join(output_dir, tagcode_str)
        if not os.path.exists(char_folder):
            os.makedirs(char_folder)
        im = Image.fromarray(img)
        im = im.resize((img_size, img_size), Image.ANTIALIAS)
        image_name = f"{len(os.listdir(char_folder))}.png"
        image_path = os.path.join(char_folder, image_name)
        im.save(image_path)


def main(zip_file_path, output_dir, img_size):
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        for gnt_file in zip_ref.namelist():
            with zip_ref.open(gnt_file) as f:
                handle_gnt_file(f, output_dir, img_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("zipfile", help="Path to the zip file containing gnt files.")
    parser.add_argument("outdir", help="Path to the output directory.")
    parser.add_argument("imgsize", type=int, help="Size of the output image.")
    args = parser.parse_args()

    main(args.zipfile, args.outdir, args.imgsize)