import os
import random
import glob
import cv2
import numpy as np
from PIL import Image

current_index = 0

def gb2312_to_decimal(gb2312_char):
    return int(gb2312_char.encode('gb2312').hex(), 16)

def get_char_images(root_path):
    char_images = {}
    for path in glob.glob(os.path.join(root_path, "*")):
        char = os.path.basename(path)
        image_paths = glob.glob(os.path.join(path, "*.png"))
        char_images[char] = image_paths
    return char_images

def random_resize(image, max_dim):
    resize_factor = np.random.uniform(0.5, 1.0)
    new_dim = int(resize_factor * max_dim)

    # Note, we pass a tuple instead of an integer to cv2.resize
    resized_image = cv2.resize(image, (new_dim, new_dim))

    # Create a new blank image and place the scaled and moved number into it
    new_image = np.zeros((max_dim, max_dim, 3), np.uint8)

    # Randomly move the number within the acceptable vertical range
    vertical_position = np.random.randint(0, max_dim - new_dim)
    new_image[vertical_position:vertical_position+new_dim, :new_dim] = resized_image
    
    return new_image

def generate_dataset(words_file, char_images, output_dir, num_samples, label_file, max_length, min_spacing, max_spacing):
    global current_index

    os.makedirs(os.path.join(output_dir, "labels"), exist_ok=True)

    with open(words_file, 'r', encoding='utf-8') as f:
        words = [line.strip() for line in f]

    os.makedirs(os.path.join(output_dir, "imgs"), exist_ok=True)

    label_file_path = os.path.join(output_dir, "labels", label_file)
    with open(label_file_path, 'w', encoding='utf-8') as f:
        for i in range(num_samples):
            index = current_index + i
            word = random.choice(words)

            if len(word) > max_length:
                word = word[:max_length]

            images = []
            for char in word:
                char_code = str(gb2312_to_decimal(char))
                image_path = random.choice(char_images[char_code])
                image = cv2.imread(image_path)
                image = cv2.resize(image, (28, 28)) # Resize if needed
                image = cv2.bitwise_not(image) # Invert colors
                
                # 50% probability to scale the image
                if random.random() < 0.5:
                    image = random_resize(image, 28)

                images.append(image)

            blank_image = np.zeros((32, 168, 3), np.uint8)  # black blank image
            x = 0
            for img in images:
                spacing = random.randint(min_spacing, max_spacing)

                # center the character image vertically
                start_y = (32 - img.shape[0]) // 2
                blank_image[start_y:start_y+img.shape[0], x:x+img.shape[1]] = img
                x += img.shape[1] + spacing
            
            img_path = os.path.join(output_dir, "imgs", f"{index}.png")
            cv2.imwrite(img_path, blank_image)
            
            f.write(f"{index}.png {word}\n")
        current_index += num_samples

if __name__ == "__main__":
    train_images = get_char_images("/Users/peiyandong/Documents/code/ai/train-data/hwdb/single/1.0/raw/train")
    test_images = get_char_images("/Users/peiyandong/Documents/code/ai/train-data/hwdb/single/1.0/raw/test")

    generate_dataset("data/words.txt", train_images, "/Users/peiyandong/Documents/code/ai/train-data/hwdb/single/data", 10000, "hwdb_train.txt", 7, 0, 10)
    generate_dataset("data/words.txt", train_images, "/Users/peiyandong/Documents/code/ai/train-data/hwdb/single/data", 500, "hwdb_val.txt", 7, 0, 10)
    generate_dataset("data/words.txt", test_images, "/Users/peiyandong/Documents/code/ai/train-data/hwdb/single/data", 1000, "hwdb_test.txt", 7, 0, 10)