import torch
import numpy as np
import torch.backends.cudnn as cudnn
import os
from tqdm import tqdm
from models.my_model import Xception
from dataset.my_dataset import MyDataSet_cls
from torch.utils import data
import cv2




torch.manual_seed(0)
INPUT_SIZE = '320, 240'     
w, h = map(int, INPUT_SIZE.split(','))
INPUT_CHANNEL = 4
NUM_CLASSES_CLS = 5
NAME = 'Select/Patient_01/'




def main():
    """Create the network and start the training."""
    
    cudnn.enabled = True

    ############# Create mask-guided classification network.
    model = Xception(num_classes=NUM_CLASSES_CLS, input_channel=INPUT_CHANNEL)
    model.cuda()
    model.float()
    cudnn.benchmark = True
    
    ############# Load training and validation data
    
    data_total_root = './results/only_test/Patient_01/'
    data_total_root_mask = './results/only_test/Patient_01_coarse_final/total_rebalance/'        
    data_total_list = './dataset/patient1_img_precise/cls/Total_cls.txt'
    totalloader = data.DataLoader(MyDataSet_cls(data_total_root, data_total_root_mask, data_total_list, crop_size=(w, h)), batch_size=1, shuffle=False,
                                num_workers=2, pin_memory=True, drop_last=True)

    path = 'results/' + NAME
    if not os.path.isdir(path):
        os.mkdir(path)
    

    pretrained_dict = torch.load(r'./results/MaskCN/Cls.pth')
    model.load_state_dict(pretrained_dict)
    

    ############# Start the training


    for i_iter, batch in tqdm(enumerate(totalloader)):     

        images, coarsemask, labels, name = batch        
        images = images.cuda()
        coarsemask = coarsemask.cuda()                  

        model.eval()
        with torch.no_grad():
            data_cla = torch.cat((images, coarsemask), dim=1)
            pred = model(data_cla)                                      # torch.Size([1, 5])

        pred_class = np.argmax(pred.cpu().data.numpy(), axis=1)         # [2] ['Patient-01-010727.png']

        if pred_class[0] == 0 or pred_class[0] == 2 or pred_class[0] == 4 or pred_class[0] == 1:
            '''
            f_path = path + 'select_others.txt'
            f1 = open(f_path, "a+")
            f1.write(name[0]+'\n')           # or directly transfer the predicted mask to black
            f1.close()
            '''
            image = cv2.imread('./results/only_test/Patient_01_coarse_final/total_rebalance/' + name[0])

            height, width, channel = image.shape
            
            for row in range(height):
                for col in range(width):
                    image[row][col][0] = 0
                    image[row][col][1] = 0
                    image[row][col][2] = 0

            cv2.imwrite('./results/only_test/Patient_01_coarse_final/select_soft_rebalance/' + name[0], image)


if __name__ == '__main__':
    main()



