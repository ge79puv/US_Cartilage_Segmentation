import os
import shutil
import matplotlib.pyplot as plt
import numpy as np
import shutil
from glob import glob
from natsort import os_sorted
import cv2




def check_folder(path):
    
    folder = os.path.exists(path)

    if folder:
        shutil.rmtree(path)
        os.mkdir(path)
        print("delete all existing files in {} folder!".format(path))

    if not folder:                
        os.mkdir(path)
        print("create {} folder!".format(path))


def visual_heatmap(heatmap, img, save_path=None, name=None):            
    # visualization of CAM from cls network

    plt.matshow(heatmap)
    # plt.show()

    # img = cv2.imread(img_path)                                        # 用cv2加载原始图像
    img = img.data.cpu().numpy()
    img = np.uint8(img*255)                                             # (1, 3, 240, 320)
    img = img.squeeze()
    img = img.transpose((1, 2, 0))

    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))         # 将热力图的大小调整为与原始图像相同
    heatmap = np.uint8(255 * heatmap)                                   # 将热力图转换为RGB格式
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)              # 将热力图应用于原始图像
    superimposed_img = heatmap * 0.4 + img                              # 0.4是热力图强度因子

    if save_path:
        cv2.imwrite(save_path+'heatmap_'+name[0], superimposed_img)                
        # plt.savefig(save_path+'heatmap_ori_'+name[0])

    # cv2.imshow('heatmap.png', superimposed_img)  
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()




def k_nearest_neighbors(data, predict):
    # KNN algorithm for VAE post-processor

    # 计算predict点到各点的距离
    min_distances = np.inf
    nearest_neighbor = None
    for features in data:
            euclidean_distance = np.linalg.norm(np.array(features) - np.array(predict))
            if euclidean_distance < min_distances:
                min_distances, nearest_neighbor = euclidean_distance, features

    return nearest_neighbor



def move_img1(ori_folder, old_folder, new_folder):
    # copy the same name image from old_folder to new_folder, if they exist in ori_folder

    ori_file_list = os_sorted(next(os.walk(ori_folder))[2])
    old_file_list = glob(old_folder + '*.png')
    
    for oldfile in old_file_list:

        if not os.path.isfile(oldfile):
            print ("%s not exist!"%(oldfile))

        else:

            old_path, old_name = os.path.split(oldfile)   

            if old_name in ori_file_list:
                
                if not os.path.exists(new_folder):
                    os.makedirs(new_folder)        

                shutil.copy(oldfile, new_folder + old_name)    

                # print ("copy %s -> %s"%(oldfile, new_folder + old_name))
            

def move_img2(ori_folder, old_folder, new_folder):
    # Read the image name in txt, copy to corresponding folders 

    old_file_list = glob(old_folder + '*.png')
    img_ids = [i_id.strip() for i_id in open(ori_folder)]

    ori_file_list = []
    for name in img_ids:
        img_file = name[name.find('/')+1:name.find(' ')]
        ori_file_list.append(img_file)

    for oldfile in old_file_list:

        if not os.path.isfile(oldfile):
            print ("%s not exist!"%(oldfile))

        else:
            old_path, old_name = os.path.split(oldfile)   

            if old_name in ori_file_list:
                
                if not os.path.exists(new_folder):
                    os.makedirs(new_folder)        

                shutil.copy(oldfile, new_folder + old_name)    

                # print ("copy %s -> %s"%(oldfile, new_folder + old_name))    


if __name__ == "__main__":

    ori_folder = r'./dataset/patient1_img/cls/Testing_cls_new.txt'   # if update .txt, then update the images and masks
    old_folder = r'./dataset/patient1_label_new/chest/'
    new_folder = r'./dataset/patient1_label_new/seg/test/'

    # move_img1
    # r'C:/Users/16967/Desktop/Master thesis/code/dataset/patient1_label_new/seg/validation/Annotation/'
    # r'C:/Users/16967/Desktop/Master thesis/code/dataset/patient1_img/seg/chest/'
    # r'C:/Users/16967/Desktop/Master thesis/code/dataset/patient1_img/seg/validation/Annotation/'
    
    move_img2(ori_folder, old_folder, new_folder)


