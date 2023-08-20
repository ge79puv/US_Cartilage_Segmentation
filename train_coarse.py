import torch
import numpy as np
import torch.backends.cudnn as cudnn
import os
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from utils.metrics import Dice_Loss, HausdorffLoss, HausdorffDTLoss
import matplotlib.pyplot as plt
from dataset.my_dataset import MyDataSet_seg
from torch.utils import data
from models.my_model import deeplabv3plus
from skimage.io import imsave
from torchsummary import summary
from skimage import io
import torch.nn.functional as F
import cv2
from utils.BD_loss import SurfaceLoss, class2one_hot, one_hot2dist
from models.U_Net import UNet
import warnings
from torchviz import make_dot





warnings.filterwarnings("ignore")
torch.manual_seed(0)


INPUT_SIZE = '320, 240'     
w, h = map(int, INPUT_SIZE.split(','))
LEARNING_RATE = 0.00002          
NUM_CLASSES = 2            
EPOCH = 100
BATCH_SIZE = 16           
NAME = 'CoarseSN/'




def val_mode_seg(valloader, model, path, epoch):
    dice = []
    sen = []
    spe = []
    acc = []
    jac_score = []
    total = []
    only_hd = []

    for index, batch in enumerate(valloader):

        data, mask, name = batch            # torch.Size([1, 3, 240, 320])  torch.Size([1, 3, 240, 320])
        data = data.cuda()
        labels = mask.cuda()                # !!!
        mask = mask[0].data.numpy()          
        val_mask = np.int64(mask > 0)       # 1 0       

        model.eval()
        with torch.no_grad():
            pred = model(data)              # torch.Size([1, 2, 240, 320]) 

        Loss_func1 = Dice_Loss()
        dice_loss = Loss_func1(pred, labels[:, 0:2, :, :])          # !!!

        pred = torch.softmax(pred, dim=1).cpu().data.numpy()        # (1, 2, 240, 320)
        pred_arg = np.argmax(pred[0], axis=0)                       # (240, 320)

        # io.imsave(os.path.join(path, name[0]), (pred_arg*255.0).astype('uint8'))
        
        labels_copy = labels                            # !!!

        Loss_func2 = SurfaceLoss()
        labels = class2one_hot(labels[:, 0, :, :], 2)   # !!!
        labels = torch.from_numpy(one_hot2dist(labels.cpu().numpy())).float()
        bd_loss = Loss_func2(torch.tensor(pred).cuda(), labels.cuda())

        HD_func = HausdorffLoss()
        pred = np.argmax(pred, axis=1)                  # !!! 
        hd_loss = HD_func(torch.tensor(pred).cuda(), labels_copy[:, 0, :, :])
        
        total_loss = dice_loss      # + bd_loss

        total.append(total_loss.cpu().data.numpy())
        only_hd.append(hd_loss.cpu().data.numpy())
        
        y_true_f = val_mask[0].reshape(val_mask[0].shape[0]*val_mask[0].shape[1])    
        y_pred_f = pred_arg.reshape(pred_arg.shape[0]*pred_arg.shape[1])             

        intersection = np.float64(np.sum(y_true_f * y_pred_f))
        dice.append((2. * intersection) / (np.sum(y_true_f) + np.sum(y_pred_f)))
        sen.append(intersection / np.sum(y_true_f))
        intersection0 = np.float64(np.sum((1 - y_true_f) * (1 - y_pred_f)))
        spe.append(intersection0 / np.sum(1 - y_true_f))
        acc.append(accuracy_score(y_true_f, y_pred_f))
        jac_score.append(intersection / (np.sum(y_true_f) + np.sum(y_pred_f) - intersection))
        
        
        if index in [108]:          
            fig = plt.figure()
            ax = fig.add_subplot(131)
            ax.imshow(data[0].cpu().data.numpy().transpose(1, 2, 0))
            ax.axis('off')
            ax = fig.add_subplot(132)
            mask = mask.transpose((1, 2, 0))
            ax.imshow(mask)
            ax.axis('off')
            ax = fig.add_subplot(133)
            ax.imshow(pred_arg)
            ax.axis('off')
            fig.suptitle('original image, ground truth mask, predicted mask',fontsize=6)
            fig.savefig(path + name[0][11:-4] + '_e' + str(epoch) + '.png', dpi=200, bbox_inches='tight')
            ax.cla()
            fig.clf()
            plt.close()
        
    return np.array(acc), np.array(dice), np.array(sen), np.array(spe), np.array(jac_score), np.array(only_hd), np.array(total)


def Jaccard(pred_arg, mask):

    pred_arg = torch.softmax(pred_arg, dim=1).cpu().data.numpy()
    pred_arg = np.argmax(pred_arg, axis=1)
    mask = mask.cpu().data.numpy()

    y_true_f = mask[:, 0, :, :].reshape(mask[:, 0, :, :].shape[0] * mask[:, 0, :, :].shape[1] * mask[:, 0, :, :].shape[2])  # , order='F'
    y_pred_f = pred_arg.reshape(pred_arg.shape[0] * pred_arg.shape[1] * pred_arg.shape[2])  

    intersection = np.float64(np.sum(y_true_f * y_pred_f))
    jac_score = intersection / (np.sum(y_true_f) + np.sum(y_pred_f) - intersection)

    return jac_score


def main():
    
    
    ############# Create coarse segmentation network ############### 

    model = deeplabv3plus(num_classes=NUM_CLASSES, input_channel=3)      
    # model = UNet(num_classes=NUM_CLASSES)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler_lr = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', 
                                    patience=10, verbose=1, factor=0.5, min_lr=1e-6)

    model.cuda()
    model.train()
    model.float()                 

    cudnn.enabled = True
    cudnn.benchmark = True


    Loss_func1 = Dice_Loss()   
    Loss_func2 = SurfaceLoss()  
    HD_func = HausdorffDTLoss()


    ############# Load training and validation data ############### 

    data_train_img_root = './dataset/patient1_img_precise/seg/train/'
    data_train_label_root = './dataset/patient1_label_precise/seg/train/'
    data_train_list = './dataset/patient1_img_precise/cls/Training_cls_new.txt' 
    trainloader = data.DataLoader(MyDataSet_seg(data_train_img_root, data_train_label_root, data_train_list, crop_size=(w, h)),
                                    batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
                                     
    data_val_img_root = './dataset/patient1_img_precise/seg/validation/'
    data_val_label_root = './dataset/patient1_label_precise/seg/validation/'
    data_val_list = './dataset/patient1_img_precise/cls/Validation_cls_new.txt' 
    valloader = data.DataLoader(MyDataSet_seg(data_val_img_root, data_val_label_root, data_val_list, crop_size=(w, h)), 
                                    batch_size=1, shuffle=False, num_workers=2, pin_memory=True, drop_last=True)
                                     
    data_test_img_root = './dataset/patient1_img_precise/seg/test/'
    data_test_label_root = './dataset/patient1_label_precise/seg/test/'
    data_test_list = './dataset/patient1_img_precise/cls/Testing_cls_new.txt' 
    testloader = data.DataLoader(MyDataSet_seg(data_test_img_root, data_test_label_root, data_test_list, crop_size=(w, h)), 
                                    batch_size=1, shuffle=False, num_workers=2, pin_memory=True, drop_last=True)

    path = 'results/' + NAME
    if not os.path.isdir(path):
        os.mkdir(path)
    f_path = path + 'output_coarse.txt'

    val_loss = []
    val_hd_loss = []
    val_jac = []
    val_dice = []
    best_score = 0.


    
    ############# Start the training ############### 

    for epoch in range(EPOCH):

        train_loss_total = []
        train_jac = []
        train_loss_term1 = []
        train_loss_term2 = []
        

        for i_iter, batch in tqdm(enumerate(trainloader)):

            images, labels, name = batch          
            images = images.cuda()               
            labels = labels.cuda()                 

            optimizer.zero_grad()

            preds = model(images)       
            
            term1 = Loss_func1(preds, labels[:, 0:2, :, :])    
            term = term1      
            

            # train with: dice + BD loss: 
            # preds = F.softmax(preds, dim=1)                   
            # labels = class2one_hot(labels[:, 0, :, :], 2)    
            # labels = torch.from_numpy(one_hot2dist(labels.cpu().numpy())).float()     # (b,num_class,h,w) 0 1
            # term2 = Loss_func2(preds, labels.cuda())
            # term = term1 * (1-0.01*epoch) + term2 * 0.01*epoch    # BD loss rebalance training
            # term = term1 + term2

            # computation graph, backpropagation
            # g = make_dot(term)
            # g.render(filename='graph', view=True)     

            if i_iter % 10 == 0:
                print(term)

            term.backward()
            optimizer.step()

            train_loss_total.append(term.cpu().data.numpy())
            train_jac.append(Jaccard(preds, labels))            
            train_loss_term1.append(term1.cpu().data.numpy()) 
            # train_loss_term2.append(term2.cpu().data.numpy()) 
            

        print("train_epoch%d: lossTotal=%f, Jaccard=%f, loss_bce=%f, loss_haus=%f \n" % (epoch, np.nanmean(train_loss_total), np.nanmean(train_jac)
                                                        , np.nanmean(train_loss_term1), np.nanmean(train_loss_term2)))


        ############# Start the validation #############

        [vacc, vdice, vsen, vspe, vjac_score, vhd_loss, vtotal_loss] = val_mode_seg(valloader, model, path, epoch)
        line_val = "val%d: vacc=%f, vdice=%f, vsensitivity=%f, vspecifity=%f, vjac=%f, vtotal=%f, vhdloss=%f \n" % \
                (epoch, np.nanmean(vacc), np.nanmean(vdice), np.nanmean(vsen), np.nanmean(vspe),
                    np.nanmean(vjac_score), np.nanmean(vhd_loss), np.nanmean(vtotal_loss))

        scheduler_lr.step(np.nanmean(vjac_score))

        print(line_val)
        f = open(f_path, "a+")
        f.write(line_val)

        ############# Plot val curve #############

        val_loss.append(np.nanmean(vtotal_loss))
        val_hd_loss.append(np.nanmean(vhd_loss))
        val_jac.append(np.nanmean(vjac_score))
        val_dice.append(np.nanmean(vdice))

        
        plt.figure()
        plt.plot(val_jac, label='val jaccard', color='blue', linestyle='--')
        plt.legend(loc='best')

        plt.savefig(os.path.join(path, 'jaccard_old.png'))
        plt.clf()
        plt.close()
        plt.show()

        plt.close('all')
        
        ############# Save network #############

        total_score = np.nanmean(vjac_score)     

        if total_score > best_score:
            best_score = total_score
            best_model = model.state_dict()
            print('Best model score : %.4f'%(best_score))
            torch.save(best_model, path + 'CoarseSN' + '.pth')
    

    Loss = np.array(val_loss)
    np.save('./results/CoarseSN/Loss_old', Loss)

    HD_Loss = np.array(val_hd_loss)
    np.save('./results/CoarseSN/HD_Loss_old', HD_Loss)

    IOU = np.array(val_jac)
    np.save('./results/CoarseSN/IOU_old', IOU)  

    Dice = np.array(val_dice)
    np.save('./results/CoarseSN/Dice_old', Dice)      


    plt.figure(1)
    plt.plot(Loss, label='Loss')
    plt.plot(HD_Loss, label='HD_Loss')
    plt.title('Loss')
    plt.legend()
    # plt.axis([0, None, 0, 1])
    plt.savefig(path + 'Loss_old')
    plt.show()

    plt.figure(2)
    plt.plot(IOU, label='IOU')
    plt.plot(Dice, label='Dice')
    plt.title('IOU')
    plt.legend()
    plt.savefig(path + 'IOU_old')
    plt.show()

    
    ############# Start the test #############

    pretrained_dict = torch.load(r'./results/CoarseSN/CoarseSN.pth')
    model.load_state_dict(pretrained_dict)    

    [tacc, tdice, tsen, tspe, tjac_score, thd_loss, ttotal_loss] = val_mode_seg(testloader, model, path, epoch)   
    line_test = "test%d: tacc=%f, tdice=%f, tsensitivity=%f, tspecifity=%f, tjac=%f, thdloss=%f, ttotal=%f \n" % \
            (epoch, np.nanmean(tacc), np.nanmean(tdice), np.nanmean(tsen), np.nanmean(tspe),
                np.nanmean(tjac_score), np.nanmean(thd_loss), np.nanmean(ttotal_loss))

    print(line_test)
    f = open(f_path, "a+")
    f.write(line_test)

    
if __name__ == '__main__':
    main()



   
    '''
    hausdorff_sd = cv2.createHausdorffDistanceExtractor()
    pretrained_dict = torch.load(r'./results/CoarseSN/CoarseSN.pth')
    model.load_state_dict(pretrained_dict) 
    
    HD = []

    for i_iter, batch in tqdm(enumerate(trainloader)):

        images, labels, name = batch
        images = images.cuda()              
        labels = labels.cuda()                

        model.eval()
        with torch.no_grad():
            preds = model(images)    

        preds = torch.softmax(preds, dim=1).cpu().data.numpy()
        preds = np.argmax(preds, axis=1)

        hd_loss = HD_func(torch.tensor(preds).cuda(), labels[:, 0, :, :])
        HD.append(hd_loss.cpu().data.numpy())

    print(np.nanmean(HD))
    # Dice: 3.6051726, alpha1: 4.4096727, rebalance: 3.429865, newHD: 5.1110916 
    '''



