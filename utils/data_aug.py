from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import os, shutil, glob
import numpy as np
from skimage.io import imread, imshow, imsave
from other_utils import check_folder


TRAIN_PATH = r'C:/Users/16967/Desktop/BoneSegmentation_MA/dataset/patient1_img_precise/soft/'
# train_ids = next(os.walk(TRAIN_PATH))[2]
png_files = glob.glob(os.path.join(TRAIN_PATH, '*.png'))


path_gen_x = r'D:/dataset/patient1_img_precise/soft_aug/'
check_folder(path_gen_x)

path_gen_y = r'D:/dataset/patient1_label_precise/soft_aug/'
check_folder(path_gen_y)


# New images and masks generator with six transformations together

img_datagen = ImageDataGenerator(
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=0.05,
        zoom_range=0.05,
        horizontal_flip=True,
        fill_mode='nearest')

mask_datagen = ImageDataGenerator(
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=0.05,
        zoom_range=0.05,
        horizontal_flip=True,
        fill_mode='nearest')


f1 = glob.glob(r'C:/Users/16967/Desktop/BoneSegmentation_MA/dataset/patient1_img_precise/soft/*.png')
f2 = glob.glob(r'C:/Users/16967/Desktop/BoneSegmentation_MA/dataset/patient1_label_precise/soft/*.png')


img = []
mask = []

for i in range(len(png_files)):

    img1 = imread(f1[i])            # <class 'numpy.ndarray'>  (64, 144)
    x = img_to_array(img1)          # <class 'numpy.ndarray'> (64, 144, 1)
    x = np.expand_dims(x, axis=0)   # <class 'numpy.ndarray'> (1, 64, 144, 1)   This shape is a prerequisite for the next augmentation operation

    img2 = imread(f2[i]) 
    y = img_to_array(img2)  
    y = np.expand_dims(y, axis=0) 


    # Each image generates five new images, and masks with the same seed
    j = 0
    for batch in img_datagen.flow(x, batch_size=1,
                            save_to_dir=path_gen_x, 
                            save_prefix='img_gen_'+str(i), save_format='png', seed=i):  
        j = j + 1
        if j > 99:
            break       


    j = 0
    for batch in mask_datagen.flow(y, batch_size=1,
                            save_to_dir=path_gen_y, 
                            save_prefix='mask_gen_'+str(i), save_format='png', seed=i):  
        j = j + 1
        if j > 99:
            break      


print("Data Augmentation Done!")



