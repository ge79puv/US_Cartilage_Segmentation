import torch
import numpy as np
import torch.backends.cudnn as cudnn
import os
from tqdm import tqdm
from models.my_model import Xception, deeplabv3plus_enhanced
from sklearn.metrics import accuracy_score
from utils.metrics import Dice_Loss, HausdorffLoss
import matplotlib.pyplot as plt
from dataset.my_dataset import MyDataSet_seg
from torch.utils import data
from utils.other_utils import visual_heatmap
from skimage import io
import torch.nn.functional as F
from utils.BD_loss import SurfaceLoss, class2one_hot, one_hot2dist
import warnings






torch.manual_seed(0)
warnings.filterwarnings("ignore")

INPUT_SIZE = '320, 240'                   
w, h = map(int, INPUT_SIZE.split(','))
LEARNING_RATE = 2e-5
INPUT_CHANNEL = 4
NUM_CLASSES_SEG = 2
NUM_CLASSES_CLS = 5
BATCH_SIZE = 16
EPOCH = 100
NAME = 'EnhanceSN/'





def val_mode_seg(valloader, val_cams, EnhanceSN, MaskCN, path, epoch):
    dice = []
    sen = []
    spe = []
    acc = []
    jac_score = []
    total = []
    only_hd = []


    for index, batch in tqdm(enumerate(valloader)):

        data, coarsemask, mask, name = batch
        data = data.cuda()
        coarsemask = coarsemask.cuda()

        labels = mask.cuda()                # !!!
        mask = mask[0].data.numpy()
        val_mask = np.int64(mask > 0)

        with torch.no_grad():           
            input_cla = torch.cat((data, coarsemask), dim=1)
            preds = MaskCN(input_cla)                       
            model_layers = MaskCN.get_layers()
            cls_features = model_layers[0]          

        EnhanceSN.eval()
        with torch.no_grad():
            cla_cam = val_cams[index]                                           
            cla_cam = torch.from_numpy(cla_cam).unsqueeze(0).unsqueeze(0)       
            pred = EnhanceSN(data, cla_cam.cuda(), cls_features)

        Loss_func1 = Dice_Loss()
        dice_loss = Loss_func1(pred, labels[:, 0:2, :, :])  # !!!

        pred = torch.softmax(pred, dim=1).cpu().data.numpy()
        pred_arg = np.argmax(pred[0], axis=0)

        # io.imsave(os.path.join(path, name[0]), (pred_arg*255.0).astype('uint8'))

        labels_copy = labels      # !!!

        Loss_func2 = SurfaceLoss()
        labels = class2one_hot(labels[:, 0, :, :], 2)   # !!!
        labels = torch.from_numpy(one_hot2dist(labels.cpu().numpy())).float()
        bd_loss = Loss_func2(torch.tensor(pred).cuda(), labels.cuda())


        HD_func = HausdorffLoss()
        pred = np.argmax(pred, axis=1)          # !!!  
        hd_loss = HD_func(torch.tensor(pred).cuda(), labels_copy[:, 0, :, :])
        
        total_loss = dice_loss + bd_loss

        total.append(total_loss.cpu().data.numpy())
        only_hd.append(hd_loss.cpu().data.numpy())

        y_true_f = val_mask[0].reshape(val_mask[0].shape[0] * val_mask[0].shape[1])
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
            fig.suptitle('RGB image,ground truth mask, predicted mask', fontsize=6)
            fig.savefig(path + name[0] + '_e' + str(epoch) + '.png', dpi=200, bbox_inches='tight')  
            ax.cla()
            fig.clf()
            plt.close()
        
    return np.array(acc), np.array(dice), np.array(sen), np.array(spe), np.array(jac_score), np.array(only_hd), np.array(total)




def val_mode_cam(valloader, MaskCN):

    val_cam = []
    for index, batch in tqdm(enumerate(valloader)):

        data, coarsemask, mask, name = batch    
        coarsemask = coarsemask.cuda()      

        with torch.no_grad():
            data_cla = torch.cat((data, coarsemask), dim=1)
            cla_cam = cam(MaskCN, data_cla, data, name)   

        val_cam.append(cla_cam[0])

    return val_cam


def Jaccard(pred_arg, mask):

    pred_arg = torch.softmax(pred_arg, dim=1).cpu().data.numpy()
    pred_arg = np.argmax(pred_arg, axis=1)

    mask = mask.cpu().data.numpy()

    y_true_f = mask[:, 0, :, :].reshape(mask[:, 0, :, :].shape[0] * mask[:, 0, :, :].shape[1] * mask[:, 0, :, :].shape[2])
    y_pred_f = pred_arg.reshape(pred_arg.shape[0] * pred_arg.shape[1] * pred_arg.shape[2])

    intersection = np.float64(np.sum(y_true_f * y_pred_f))
    jac_score = intersection / (np.sum(y_true_f) + np.sum(y_pred_f) - intersection)

    return jac_score


def cam(model, inputs, img, name):
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

        output_cam.append(cam_img)              # Normalization to (0, 1), image and mask also in (0, 1)

    return output_cam



def main():
    """Create the network and start the training."""
    model_urls = {'CoarseSN': './results/CoarseSN/CoarseSN.pth', 'MaskCN': './results/MaskCN/Cls.pth'}

    cudnn.benchmark = True
    cudnn.enabled = True

    ############# Create mask-guided classification network.
    MaskCN = Xception(num_classes=NUM_CLASSES_CLS, input_channel=INPUT_CHANNEL)
    MaskCN.cuda()              

    ############# Load pretrained weights
    pretrained_dict = torch.load(model_urls['MaskCN'])
    MaskCN.load_state_dict(pretrained_dict)
    MaskCN.eval()



    ############# Create enhanced segmentation network for other segmentation.
    EnhanceSN_others = deeplabv3plus_enhanced(num_classes=NUM_CLASSES_SEG, input_channel = 4)
    optimizer = torch.optim.Adam(EnhanceSN_others.parameters(), lr=LEARNING_RATE)
    EnhanceSN_others.cuda()

    scheduler_lr = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', 
                                patience=10, verbose=1, factor=0.5, min_lr=1e-6)

    ############# Load pretrained weights
    pretrained_dict = torch.load(model_urls['CoarseSN'])
    net_dict = EnhanceSN_others.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in net_dict) and (v.shape == net_dict[k].shape)}
    
    net_dict.update(pretrained_dict)
    EnhanceSN_others.load_state_dict(net_dict)

    EnhanceSN_others.train()
    EnhanceSN_others.float()



    dice_loss = Dice_Loss()
    boundary_loss = SurfaceLoss() 


    ############# Load training and validation data of others

    data_train_img_root = './dataset/patient1_img_precise/cls/separate_txt/only_others/train/'
    data_train_label_root = './dataset/patient1_label_precise/cls/separate_txt/only_others/train/'
    data_train_root_coursemask = './results/CoarseSN/total_old/'          
    data_train_list = './dataset/patient1_img_precise/cls/separate_txt/only_others/Training_cls_new.txt'
    trainloader_others = data.DataLoader(MyDataSet_seg(data_train_img_root, data_train_label_root, data_train_list, data_train_root_coursemask, crop_size=(w, h)),
                                  batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    
    data_val_img_root = './dataset/patient1_img_precise/cls/separate_txt/only_others/validation/'
    data_val_label_root = './dataset/patient1_label_precise/cls/separate_txt/only_others/validation/'
    data_val_root_coursemask = './results/CoarseSN/total_old/'          
    data_val_list = './dataset/patient1_img_precise/cls/separate_txt/only_others/Validation_cls_new.txt'
    valloader_others = data.DataLoader(MyDataSet_seg(data_val_img_root, data_val_label_root, data_val_list, data_val_root_coursemask, crop_size=(w, h)), batch_size=1, shuffle=False,
                                num_workers=2, pin_memory=True, drop_last=True)

    data_test_img_root = './dataset/patient1_img_precise/cls/separate_txt/only_others/test/'
    data_test_label_root = './dataset/patient1_label_precise/cls/separate_txt/only_others/test/'
    data_test_root_coursemask = './results/CoarseSN/total_old/'          
    data_test_list = './dataset/patient1_img_precise/cls/separate_txt/only_others/Testing_cls_new.txt'
    testloader_others = data.DataLoader(MyDataSet_seg(data_test_img_root, data_test_label_root, data_test_list, data_test_root_coursemask, crop_size=(w, h)), batch_size=1, shuffle=False,
                                num_workers=2, pin_memory=True, drop_last=True)




    path = 'results/' + NAME
    if not os.path.isdir(path):
        os.mkdir(path)
    f_path = path + 'output_other_alpha.txt'

    val_loss = []
    val_hd_loss = []
    val_jac = []
    val_dice = []
    best_score = 0.

    
    ############# Start the training
    for epoch in range(EPOCH):

        train_loss_total = []
        train_jac = []
        train_loss_term1 = []
        train_loss_term2 = []
                

        for i_iter, batch in tqdm(enumerate(trainloader_others)):

            images, coarsemask, labels, name = batch    
            images = images.cuda()
            coarsemask = coarsemask.cuda()              
            labels = labels.cuda()                      

            with torch.no_grad():
                input_cla = torch.cat((images, coarsemask), dim=1)
                cla_cam = cam(MaskCN, input_cla, images, name)        

            cla_cam = torch.from_numpy(np.stack(cla_cam)).unsqueeze(1).cuda()    # torch.Size([2, 1, 15, 20]) '320, 240'
            optimizer.zero_grad()


            with torch.no_grad():      
                preds = MaskCN(input_cla)                       
                model_layers = MaskCN.get_layers()
                cls_features = model_layers[0]          # torch.Size([1, 2048, 15, 20])


            EnhanceSN_others.train()
            preds = EnhanceSN_others(images, cla_cam, cls_features)

            term1 = dice_loss(preds, labels[:, 0:2, :, :])
            
            preds = F.softmax(preds, dim=1)

            labels = class2one_hot(labels[:, 0, :, :], 2)
            labels = torch.from_numpy(one_hot2dist(labels.cpu().numpy())).float()
            
            term2 = boundary_loss(preds, labels.cuda())            
            
            term = term1 + term2
            # term = term1 * (1-0.01*epoch) + term2 * 0.01*epoch

            if i_iter % 10 == 0:
                print(term1, term2)

            term.backward()
            optimizer.step()

            train_loss_total.append(term.cpu().data.numpy())
            train_jac.append(Jaccard(preds, labels))    
            train_loss_term1.append(term1.cpu().data.numpy()) 
            train_loss_term2.append(term2.cpu().data.numpy())  


        print("train_epoch%d: lossTotal=%f, Jaccard=%f, loss_bce=%f, loss_haus=%f \n" % (epoch, np.nanmean(train_loss_total), np.nanmean(train_jac)
                                                        , np.nanmean(train_loss_term1), np.nanmean(train_loss_term2)))

        ############# Start the validation

        val_cams = val_mode_cam(valloader_others, MaskCN)

        [vacc, vdice, vsen, vspe, vjac_score, vhd_loss, vtotal_loss] = val_mode_seg(valloader_others, val_cams, EnhanceSN_others, MaskCN, path, epoch)
        line_val = "val%d: vacc=%f, vdice=%f, vsensitivity=%f, vspecifity=%f, vjac=%f, vtotal=%f, vhdloss=%f \n" % \
                   (epoch, np.nanmean(vacc), np.nanmean(vdice), np.nanmean(vsen), np.nanmean(vspe),
                    np.nanmean(vjac_score), np.nanmean(vhd_loss), np.nanmean(vtotal_loss))

        scheduler_lr.step(np.nanmean(vjac_score))

        print(line_val)
        f = open(f_path, "a+")
        f.write(line_val)

        val_loss.append(np.nanmean(vtotal_loss))
        val_hd_loss.append(np.nanmean(vhd_loss))
        val_jac.append(np.nanmean(vjac_score))
        val_dice.append(np.nanmean(vdice))

        ############# Plot val curve
        plt.figure()
        plt.plot(val_jac, label='val jaccard', color='blue', linestyle='--')
        plt.legend(loc='best')

        plt.savefig(os.path.join(path, 'jaccard_other_alpha.png'))
        plt.clf()
        plt.close()
        plt.show()

        plt.close('all')

        ############# Save network

        total_score = np.nanmean(vjac_score)       

        if total_score > best_score:
            best_score = total_score
            best_epoch = epoch
            best_model = EnhanceSN_others.state_dict()
            print('Best model score : %.4f'%(best_score))
            torch.save(best_model, path + 'EnhanceSN_other_alpha' + '.pth')
            torch.save(MaskCN.state_dict(), path + 'MaskCN_other_alpha_updated' + '.pth')



    Loss = np.array(val_loss)
    np.save('./results/EnhanceSN/Loss_other_alpha', Loss)

    HD_Loss = np.array(val_hd_loss)
    np.save('./results/EnhanceSN/HD_Loss_other_alpha', HD_Loss)

    IOU = np.array(val_jac)
    np.save('./results/EnhanceSN/IOU_other_alpha', IOU)  

    Dice = np.array(val_dice)
    np.save('./results/EnhanceSN/Dice_other_alpha', Dice)      


    plt.figure(1)
    plt.plot(Loss, label='Loss')
    plt.plot(HD_Loss, label='HD_Loss')
    plt.title('Loss')
    plt.legend()
    # plt.axis([0, None, 0, 1])
    plt.savefig(path + 'Loss_other_alpha')
    plt.show()

    plt.figure(2)
    plt.plot(IOU, label='IOU')
    plt.plot(Dice, label='Dice')
    plt.title('IOU')
    plt.legend()
    plt.savefig(path + 'IOU_other_alpha')
    plt.show()


    
    ############# Start the test
    pretrained_dict = torch.load(r'./results/EnhanceSN/EnhanceSN_other_alpha.pth')    
    EnhanceSN_others.load_state_dict(pretrained_dict) 

    pretrained_dict = torch.load(r'./results/EnhanceSN/MaskCN_other_alpha_updated.pth')     
    MaskCN.load_state_dict(pretrained_dict) 

    test_cams = val_mode_cam(testloader_others, MaskCN)      
    epoch = 666
    [tacc, tdice, tsen, tspe, tjac_score, thd_loss, ttotal_loss] = val_mode_seg(testloader_others, test_cams, EnhanceSN_others, MaskCN, path, epoch)
    line_test = "test%d: tacc=%f, tdice=%f, tsensitivity=%f, tspecifity=%f, tjac=%f, ttotal=%f, thdloss=%f \n" % \
                (epoch, np.nanmean(tacc), np.nanmean(tdice), np.nanmean(tsen), np.nanmean(tspe),
                np.nanmean(tjac_score), np.nanmean(thd_loss), np.nanmean(ttotal_loss))

    print(line_test)
    f = open(f_path, "a+")
    f.write(line_test) 


if __name__ == '__main__':
    main()



