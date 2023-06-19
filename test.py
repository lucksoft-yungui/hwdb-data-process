import os
import numpy as np
import struct
from PIL import Image

if __name__ == "__main__":
    f = open('/Users/peiyandong/Documents/code/ai/train-data/hwdb/single/Gnt1.0TrainPart1/001-f.gnt', 'rb') # 读入一份gnt文件
    header_size = 10
    while True:
        header = np.fromfile(f, dtype='uint8', count=header_size)
        if not header.size: break
        sample_size = header[0] + (header[1] << 8) + (header[2] << 16) + (header[3] << 24)
        tagcode = header[5] + (header[4] << 8)
        width = header[6] + (header[7] << 8)
        height = header[8] + (header[9] << 8)
        if header_size + width * height != sample_size:
            break
        image = np.fromfile(f, dtype='uint8', count=width * height).reshape((height, width))
        ff = struct.pack('>H', tagcode)
        tagcode_unicode = struct.pack('>H', tagcode).decode('gb2312') # 图片对应的汉字
        print(tagcode_unicode)
        im = Image.fromarray(image)
        file_dir = '/Users/peiyandong/Documents/code/ai/train-data/hwdb/single/1.0/' + tagcode_unicode;
        if not os.path.exists(file_dir):
            os.mkdir(file_dir)

        file_lists = os.listdir(file_dir)

        im.convert('RGB').save(file_dir + '\\' + str(len(file_lists)) + '.png')