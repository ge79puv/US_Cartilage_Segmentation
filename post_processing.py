import torch
import numpy as np
import torch.backends.cudnn as cudnn
import os
from tqdm import tqdm
from utils.other_metrics import Segmentation2DMetrics
import matplotlib.pyplot as plt
from torch.utils import data
from models.VAE import VAE
from utils.other_utils import k_nearest_neighbors
import h5py
from skimage import io
from dataset.my_dataset import MyDataSet_cls
import cv2
from skimage import morphology




torch.manual_seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

INPUT_SIZE = '320, 240'     
w, h = map(int, INPUT_SIZE.split(','))     
NAME = 'Valid/'



def comparision(seg_res_ori, seg_res, name, path):      

    seg_res = seg_res[:, 0, :, :].cpu().data.numpy()
    seg_res = (seg_res > 0.5).astype(np.uint8)

    # io.imsave(os.path.join(path, name[0]), (seg_res*255).transpose(1, 2, 0))     
    fig = plt.figure()
    ax = fig.add_subplot(121)
    ax.imshow(seg_res_ori[0].cpu().data.numpy().transpose((1, 2, 0)))
    ax.axis('off')
    ax = fig.add_subplot(122)
    ax.imshow(seg_res[0])                           
    ax.axis('off')
    fig.suptitle('original image, ground truth mask, predicted mask',fontsize=6)
    fig.savefig(path + name[0][11:-4] + '.png', dpi=200, bbox_inches='tight')
    ax.cla()
    fig.clf()
    plt.close()

    return None


def Jaccard(pred_arg, mask):
 
    pred_arg = pred_arg.cpu().data.numpy()                                     
    pred_arg = (pred_arg > 0.5).astype(np.uint8)                        
    
    mask = mask.cpu().data.numpy()

    y_true_f = mask[:, 0, :, :].reshape(mask[:, 0, :, :].shape[0] * mask[:, 0, :, :].shape[1] * mask[:, 0, :, :].shape[2])  
    y_pred_f = pred_arg.reshape(pred_arg.shape[0] * pred_arg.shape[1] * pred_arg.shape[2])  

    intersection = np.float64(np.sum(y_true_f * y_pred_f))
    jac_score = intersection / (np.sum(y_true_f) + np.sum(y_pred_f) - intersection)

    return jac_score


def main():

    cudnn.benchmark = True
    cudnn.enabled = True
    
    ############# Create coarse segmentation network
    
    model = VAE(input_shape=(3, 240, 320), output_shape=(1, 240, 320)).to(device)  

    model.cuda()
    model.train()
    model.float()


    ############# Load pretrained weights
    
    pretrained_dict = torch.load(r'./results/VAE_train/VAE_train.pth')
    model.load_state_dict(pretrained_dict)
                     

    ############# Load training and validation data

    data_total_root = './results/only_test/Patient_01_fine_precise_small/total/'            # 网络的最终输出作为VAE后处理的输入图片
    data_total_root_mask = './results/only_test/Patient_01_coarse_precise_small/total/'     # course_mask 有用  
    data_total_list = './dataset/patient1_img_precise/cls/Total_cls.txt'
    totalloader = data.DataLoader(MyDataSet_cls(data_total_root, data_total_root_mask, data_total_list, crop_size=(w, h)), batch_size=1, shuffle=False,
                                num_workers=2, pin_memory=True, drop_last=True)
    

    path = 'results/' + NAME
    if not os.path.isdir(path):
        os.mkdir(path)
    f_path = path + 'output_valid.txt'


    names_soft = [i_id.strip() for i_id in open('./results/Select/select_soft.txt')]


    ############# Start the training

    train_loss_total = []
    train_jac = []
    count = 0

    for i_iter, batch in tqdm(enumerate(totalloader)):      

        # Load all segmentation masks
        images, coarsemask, labels, name = batch 
        images = images.cuda()       
        coarsemask = coarsemask.cuda()    
        labels = labels.cuda()                       

        model.eval()
        with torch.no_grad():
            seg_res, z, mu, log_var = model(images)                    # torch.Size([b, 1, 256, 320])
            seg_res = images

        # Load the augmented latent vectors
        output_rs_path = './results/VAE_train/RS_samples/'
        with h5py.File(output_rs_path + "rs_samples_total.hdf5", "r") as dataset:
            rs_samples = dataset['total'][:]                                  


        # If it is soft images
        if name[0] in names_soft:      
            # Constraints Indicator
            seg_res_sq = torch.squeeze(seg_res[:, 0, :, :])                                  # torch.Size([256, 320])
            metrics_indicator = Segmentation2DMetrics(seg_res_sq.cpu().data.numpy(), 1)      # 0: background, 1: bone  
            valid_holes = metrics_indicator.count_holes(1)
            valid_connectivity = metrics_indicator.count_disconnectivity(1) 

            # Pre-processing: filter out too small outliers

            if valid_holes!=0 or valid_connectivity!=0:
              
              img = cv2.imread('./results/only_test/Patient_01_fine_precise_small/total/' + name[0])
              cv2.imwrite('./results/only_test/Patient_01_fine_precise_small/preprocessing/' + 'ori_'+name[0], img)
              
              img = np.array(img, dtype= bool)
              img = morphology.remove_small_objects(img, 230)       # 3 channel, so 3 times area
              img = morphology.remove_small_holes(img, 10000)

              img = img.astype(np.uint8)
              io.imsave('./results/only_test/Patient_01_fine_precise_small/preprocessing/' + name[0], (img*255.0).astype('uint8'))
              count += 1

              # cv2.imwrite('./results/only_test/Patient_01_fine_precise_small/preprocessing/' + name[0], img)

            # Latent vectors transformation (Post-processing)
            valid_connectivity_big = metrics_indicator.count_disconnectivity_big(1)

            if valid_holes>230 or valid_connectivity_big!=0:         # two bigger conditions

                img = cv2.imread('./results/only_test/Patient_01_fine_precise_small/total/' + name[0])
                cv2.imwrite('./results/only_test/Patient_01_fine_precise_small/postprocessing/' + 'ori_'+name[0], img)
                
                print(valid_holes, valid_connectivity_big, name)
                line = "valid_holes:{}, valid_connectivity_big:{}, name:{}  \n".format(valid_holes, valid_connectivity_big, name)
                f = open(f_path, "a+")
                f.write(line)
                f.close()

                seg_res_ori = seg_res
                
                # false_vector = model.encode(seg_res_ori.repeat(1, 3, 1, 1))
                nearest_neigh = k_nearest_neighbors(rs_samples, z.cpu().data.numpy())  # replace the casual error with its neighbours

                nearest_neigh = torch.tensor(nearest_neigh).cuda()
                seg_res = model.decode(nearest_neigh.to(torch.float32))
                count += 1

                comparision(seg_res_ori, seg_res, name, './results/only_test/Patient_01_fine_precise_small/postprocessing/')
    
        '''
        term = F.binary_cross_entropy(seg_res[:, 0, :, :], labels[:, 0, :, :], reduction='sum')  
        train_loss_total.append(term.cpu().data.numpy())
        train_jac.append(Jaccard(seg_res[:, 0, :, :], labels))
        '''

    # print("All images: lossTotal=%f, Jaccard=%f \n" % (np.nanmean(train_loss_total), np.nanmean(train_jac)))
    print("count:", count)
    


if __name__ == '__main__':
    main()







