import torch
import numpy as np
import torch.backends.cudnn as cudnn
import os
from tqdm import tqdm
from models.my_model import deeplabv3plus_enhanced, Xception, deeplabv3plus
from skimage import io
# from natsort import os_sorted
from skimage.transform import resize
from dataset.my_dataset import MyDataSet_seg
from torch.utils import data
from models.U_Net import UNet
import warnings



torch.manual_seed(0)
warnings.filterwarnings("ignore")


data_test_img_root = './results/only_test/Patient_01/'
data_test_coarse_root = './results/only_test/Patient_01_coarse_final/total_UNET/'
data_test_fine_root = './results/only_test/Patient_01_fine_final/total_old_rebalance/'


if not os.path.isdir(data_test_coarse_root):
    os.mkdir(data_test_coarse_root)

if not os.path.isdir(data_test_fine_root):
    os.mkdir(data_test_fine_root)


cudnn.enabled = True
cudnn.benchmark = True
 


train_ids = sorted(next(os.walk(data_test_img_root))[2])            
X_test = np.zeros((len(train_ids), 240, 320, 3))                   


for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):     

    path = data_test_img_root + id_
    img = io.imread(path)                                              

    img = resize(img, (240, 320, 3))  

    X_test[n] = img                                                   


X_test = torch.from_numpy(X_test).float()
print(X_test.shape)                             # torch.Size([952, 240, 320, 3])



# CoarseSN = deeplabv3plus(num_classes=2, input_channel=3)  
CoarseSN = UNet(num_classes=2)

CoarseSN.cuda()
CoarseSN.float()
pretrained_dict = torch.load(r'./results/CoarseSN/CoarseSN_UNET.pth')
CoarseSN.load_state_dict(pretrained_dict)  


for index, img in tqdm(enumerate(X_test)):

    img = img.cuda()
    img = img.permute((2, 0, 1)).unsqueeze(0) 

    CoarseSN.eval()
    with torch.no_grad():
        pred = CoarseSN(img)                                   
    
    pred = torch.softmax(pred, dim=1).cpu().data.numpy()        
    pred = np.argmax(pred[0], axis=0)                           
    
    io.imsave(os.path.join(data_test_coarse_root, train_ids[index]), (pred*255.0).astype('uint8'))




EnhanceSN = deeplabv3plus_enhanced(num_classes=2, input_channel=4)    
EnhanceSN.cuda()
EnhanceSN.float()
pretrained_dict = torch.load(r'./results/EnhanceSN/EnhanceSN_old_mixed.pth')
EnhanceSN.load_state_dict(pretrained_dict)  


MaskCN = Xception(num_classes=5, input_channel=4)
MaskCN.cuda() 
MaskCN.float()
pretrained_dict = torch.load(r'./results/EnhanceSN/MaskCN_old_mixed_updated.pth')
MaskCN.load_state_dict(pretrained_dict)



train_ids = sorted(next(os.walk(data_test_coarse_root))[2])  
Y_test = np.zeros((len(train_ids), 240, 320, 1))                    


for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):     

    path = data_test_coarse_root + id_    
    img = io.imread(path)                                              
    img = resize(img, (240, 320, 1))  

    Y_test[n] = img                                                 



Y_test = torch.from_numpy(Y_test).float()
print(Y_test.shape)



def val_mode_cam(X_test, MaskCN):

    val_cam = []

    for index, img in tqdm(enumerate(X_test)):

        img = img.cuda()
        coarsemask = Y_test[index].cuda()

        img = img.permute((2, 0, 1)).unsqueeze(0) 
        coarsemask = coarsemask.permute((2, 0, 1)).unsqueeze(0) 

        MaskCN.eval()
        with torch.no_grad():
            data_cla = torch.cat((img, coarsemask), dim=1)
            cla_cam = cam(MaskCN, data_cla)     

        val_cam.append(cla_cam[0])

    return val_cam



def cam(model, inputs):

    model.eval()
    with torch.no_grad():
        preds = model(inputs)                   # torch.Size([1, 4]) tensor([[ 18.5445, -12.0332,  -6.4492,  -3.7329]], device='cuda:0')
        class_idx = preds.argmax(dim=1)         # torch.Size([1]) tensor([0], device='cuda:0') 用于获取向量与正确分类之间的权重
        model_layers = model.get_layers()       # two tensors    model_layers[0].shape: torch.Size([1, 2048, 14, 14])    

    params = list(model.parameters())           # params[-2]: torch.Size([4, 2048])
    weights = np.squeeze(params[-2].data.cpu().numpy())     # (4, 2048) params[-2]: fc.weight, params[-1]: fc.bias
    bz, nc, h, w = model_layers[0].shape        

    output_cam = []
    for idx in range(bz):       
        cam = np.zeros((h, w), dtype=np.float32)
        for i, weight in enumerate(weights[class_idx[idx]]):            # idx = 0, class_idx[idx] = 0, i: 0-2048 channels
            cam += weight * model_layers[0][idx][i].data.cpu().numpy()

        cam_img = np.maximum(cam, 0)            # compare with 0 in each location
        cam_img = cam / np.max(cam_img)         # np.max: max value in cam_img

        # visual_heatmap(cam_img, img, "C:/Users/16967/Desktop/Master thesis/pictures/heatmap/", name)

        output_cam.append(cam_img)              

    return output_cam



test_cams = val_mode_cam(X_test, MaskCN)       


for index, img in tqdm(enumerate(X_test)):

    img = img.cuda()
    coarsemask = Y_test[index].cuda()

    img = img.permute((2, 0, 1)).unsqueeze(0) 
    coarsemask = coarsemask.permute((2, 0, 1)).unsqueeze(0) 

    MaskCN.eval()
    with torch.no_grad():                     
        input_cla = torch.cat((img, coarsemask), dim=1)
        preds = MaskCN(input_cla)                       
        model_layers = MaskCN.get_layers()
        cls_features = model_layers[0]          

    EnhanceSN.eval()
    with torch.no_grad():
        cla_cam = test_cams[index]                                           
        cla_cam = torch.from_numpy(cla_cam).unsqueeze(0).unsqueeze(0)        
        pred = EnhanceSN(img, cla_cam.cuda(), cls_features)

    pred = torch.softmax(pred, dim=1).cpu().data.numpy()
    pred = np.argmax(pred[0], axis=0)

    io.imsave(os.path.join(data_test_fine_root, train_ids[index]), (pred*255.0).astype('uint8'))





