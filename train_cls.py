import torch
import torch.nn as nn
import numpy as np
import torch.backends.cudnn as cudnn
import os
from tqdm import tqdm
from sklearn import metrics
from models.my_model import Xception
import matplotlib.pyplot as plt
from dataset.my_dataset import MyDataSet_cls
from torch.utils import data
from torchsummary import summary




torch.manual_seed(0)
model_urls = {'Xception': 'weights/xception-43020ad28.pth'}        # pretrained model

INPUT_SIZE = '320, 240'     
w, h = map(int, INPUT_SIZE.split(','))
LEARNING_RATE = 0.0001      
INPUT_CHANNEL = 4
NUM_CLASSES_CLS = 5
EPOCH = 50
BATCH_SIZE = 16
NAME = 'MaskCN/'


def cla_evaluate(label, binary_score, pro_score):               # y_true, y_pred, y_prob

    acc = metrics.accuracy_score(label, binary_score)           # (TP + TN) / (TP + FP + TN + FN)
    AP = metrics.average_precision_score(label, pro_score)      # Area under Precision-Recall Curve
    auc = metrics.roc_auc_score(label, pro_score)               # Area under ROC (TP-FP)
    CM = metrics.confusion_matrix(label, binary_score)          # Confusion Matrix (2, 2) four classes 

    # in binary classification, the count of TN is [0, 0], FN is [1, 0], TP is [1, 1] and FP is [0, 1].
    # print(CM) 
    sens = float(CM[1, 1]) / float(CM[1, 1] + CM[1, 0])         # TP / (TP + FN)        
    spec = float(CM[0, 0]) / float(CM[0, 0] + CM[0, 1])         # TN / (TN + FP)

    return acc, auc, AP, sens, spec


def val_mode_Scls(valloader, model):
    
    pro_score = []
    label_val = []
    
    for index, batch in enumerate(valloader):

        data, coarsemask, label, name = batch       # label: tensor([0], dtype=torch.int32) torch.Size([1])
        data = data.cuda()
        coarsemask = coarsemask.cuda()         

        model.eval()
        with torch.no_grad():
            data_cla = torch.cat((data, coarsemask), dim=1)
            pred = model(data_cla)

        pro_score.append(torch.softmax(pred[0], dim=0).cpu().data.numpy())    # probability calculation
        label_val.append(label[0].data.numpy())        # [array(2), array(2), array(0), array(1)]

    pro_score = np.array(pro_score)     # (X, 5)  sum of one row possibilities is 1, X: amount of val set
    label_val = np.array(label_val)     # (X, )    0 1 2 3 

    binary_score = np.eye(5)[np.argmax(pro_score, axis=-1)]   # one-hot code  (X, 5)
    label_val = np.eye(5)[np.int64(label_val)]                # (X, 5)

    # background
    label_val_a = label_val[:, 0]
    pro_score_a = pro_score[:, 0]
    binary_score_a = binary_score[:, 0]
    val_acc_b, val_auc_b, val_AP_b, sens_b, spec_b = cla_evaluate(label_val_a, binary_score_a, pro_score_a)
    # chest
    label_val_a = label_val[:, 1]
    pro_score_a = pro_score[:, 1]
    binary_score_a = binary_score[:, 1]
    val_acc_c, val_auc_c, val_AP_c, sens_c, spec_c = cla_evaluate(label_val_a, binary_score_a, pro_score_a)
    # hard
    label_val_a = label_val[:, 2]
    pro_score_a = pro_score[:, 2]
    binary_score_a = binary_score[:, 2]
    val_acc_h, val_auc_h, val_AP_h, sens_h, spec_h = cla_evaluate(label_val_a, binary_score_a, pro_score_a)
    # soft
    label_val_a = label_val[:, 3]
    pro_score_a = pro_score[:, 3]
    binary_score_a = binary_score[:, 3]
    val_acc_s, val_auc_s, val_AP_s, sens_s, spec_s = cla_evaluate(label_val_a, binary_score_a, pro_score_a)
    # transition
    label_val_a = label_val[:, 4]
    pro_score_a = pro_score[:, 4]
    binary_score_a = binary_score[:, 4]
    val_acc_t, val_auc_t, val_AP_t, sens_t, spec_t = cla_evaluate(label_val_a, binary_score_a, pro_score_a)

    return val_acc_b, val_auc_b, val_AP_b, sens_b, spec_b, \
            val_acc_c, val_auc_c, val_AP_c, sens_c, spec_c, \
            val_acc_h, val_auc_h, val_AP_h, sens_h, spec_h, \
            val_acc_s, val_auc_s, val_AP_s, sens_s, spec_s, \
            val_acc_t, val_auc_t, val_AP_t, sens_t, spec_t



def main():
    
    ############# Create classification network ############### 

    model = Xception(num_classes=NUM_CLASSES_CLS, input_channel=INPUT_CHANNEL)
    # summary(model, input_size=[(3, 240, 320)], batch_size=2, device="cuda")  
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    pretrained_dict = torch.load(model_urls['Xception'])

    net_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in net_dict) and (v.shape == net_dict[k].shape)}
    net_dict.update(pretrained_dict)
    model.load_state_dict(net_dict)


    model.cuda()
    model.train()
    model.float()

    cudnn.enabled = True
    cudnn.benchmark = True

    ce_loss = nn.CrossEntropyLoss()


    ############# Load training, validation, test data ############### 

    data_train_root = './dataset/patient1_img_precise/cls/train/'
    data_train_root_mask = './results/CoarseSN/train_old/'        
    data_train_list = './dataset/patient1_img_precise/cls/Training_cls_new.txt'
    trainloader = data.DataLoader(MyDataSet_cls(data_train_root, data_train_root_mask, data_train_list, crop_size=(w, h)),  # 0.5
                                  batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)  # , max_iters=STEPS * BATCH_SIZE
    
    data_val_root = './dataset/patient1_img_precise/cls/validation/'
    data_val_root_mask = './results/CoarseSN/validation_old/'
    data_val_list = './dataset/patient1_img_precise/cls/Validation_cls_new.txt'
    valloader = data.DataLoader(MyDataSet_cls(data_val_root, data_val_root_mask, data_val_list, crop_size=(w, h)), batch_size=1, shuffle=False,
                                num_workers=2, pin_memory=True, drop_last=True)

    data_test_root = './dataset/patient1_img_precise/cls/test/'
    data_test_root_mask = './results/CoarseSN/test_old/'
    data_test_list = './dataset/patient1_img_precise/cls/Testing_cls_new.txt'
    testloader = data.DataLoader(MyDataSet_cls(data_test_root, data_test_root_mask, data_test_list, crop_size=(w, h)), batch_size=1, shuffle=False,
                                num_workers=2, pin_memory=True, drop_last=True)
    
    path = 'results/' + NAME
    if not os.path.isdir(path):
        os.mkdir(path)
    f_path = path + 'output_cls.txt'

    val_b = []
    val_c = []
    val_h = []    
    val_s = []
    val_t = []
    val_mean = []
    best_score = 0.

    
    ############# Start the training ############### 
    
    for epoch in range(EPOCH):

        train_loss = []

        for i_iter, batch in tqdm(enumerate(trainloader)):     

            images, coarsemask, labels, name = batch        
            # torch.Size([2, 3, 626, 844]) torch.Size([2, 1, 626, 844]) torch.Size([2]) 

            images = images.cuda()
            coarsemask = coarsemask.cuda()          
            labels = labels.cuda()                          
            
            input_cla = torch.cat((images, coarsemask), dim=1)           # four channels    
            optimizer.zero_grad()

            preds = model(input_cla)                        # torch.Size([2, 4])    
            
            term = ce_loss(preds, labels.long())            
            term.backward()
            optimizer.step()

            train_loss.append(term.cpu().data.numpy())

        print("train_epoch%d: loss=%f\n" % (epoch, np.nanmean(train_loss)))    


        ############# Start the validation #############

        [val_acc_b, val_auc_b, val_AP_b, val_sens_b, val_spec_b, val_acc_c, val_auc_c, val_AP_c, val_sens_c, val_spec_c, \
         val_acc_h, val_auc_h, val_AP_h, val_sens_h, val_spec_h, val_acc_s, val_auc_s, val_AP_s, val_sens_s, val_spec_s, \
         val_acc_t, val_auc_t, val_AP_t, val_sens_t, val_spec_t] = val_mode_Scls(valloader, model)


        line_val_b = "val%d:vacc_b=%f,vauc_b=%f,vAP_b=%f,vsens_b=%f,vspec_b=%f \n" % (
        epoch, val_acc_b, val_auc_b, val_AP_b, val_sens_b, val_spec_b)
        line_val_c = "val%d:vacc_c=%f,vauc_c=%f,vAP_c=%f,vsens_c=%f,vspec_c=%f \n" % (
        epoch, val_acc_c, val_auc_c, val_AP_c, val_sens_c, val_spec_c)
        line_val_h = "val%d:vacc_h=%f,vauc_h=%f,vAP_h=%f,vsens_h=%f,vspec_h=%f \n" % (
        epoch, val_acc_h, val_auc_h, val_AP_h, val_sens_h, val_spec_h)
        line_val_s = "val%d:vacc_s=%f,vauc_s=%f,vAP_s=%f,vsens_s=%f,vspec_s=%f \n" % (
        epoch, val_acc_s, val_auc_s, val_AP_s, val_sens_s, val_spec_s)
        line_val_t = "val%d:vacc_t=%f,vauc_t=%f,vAP_t=%f,vsens_t=%f,vspec_t=%f \n" % (
        epoch, val_acc_t, val_auc_t, val_AP_t, val_sens_t, val_spec_t)

        print(line_val_b)
        print(line_val_c)
        print(line_val_h)
        print(line_val_s)
        print(line_val_t)

        f = open(f_path, "a+")
        f.write(line_val_b)        
        f.write(line_val_c)
        f.write(line_val_h)
        f.write(line_val_s)
        f.write(line_val_t)

        val_b.append(np.nanmean(val_auc_b))
        val_c.append(np.nanmean(val_auc_c))
        val_h.append(np.nanmean(val_auc_h))
        val_s.append(np.nanmean(val_auc_s))
        val_t.append(np.nanmean(val_auc_t))
        val_mean.append((np.nanmean(val_auc_b) + np.nanmean(val_auc_c) + np.nanmean(val_auc_h) + np.nanmean(val_auc_s) + np.nanmean(val_auc_t)) / 5.)
        

        ############# Plot val curves #############

        plt.figure()
        plt.plot(val_b, label='val_b', color='purple')
        plt.plot(val_c, label='val_c', color='red')
        plt.plot(val_h, label='val_h', color='green')
        plt.plot(val_s, label='val_s', color='yellow')
        plt.plot(val_t, label='val_t', color='black')        
        plt.plot(val_mean, label='val_mean', color='blue')
        plt.legend(loc='best')

        plt.savefig(os.path.join(path, 'loss.png'))
        plt.clf()
        plt.close()
        plt.show()

        plt.close('all')

        ############# Save network #############
        
        total_score = np.nanmean(val_mean)       

        if total_score > best_score:
            best_score = total_score
            best_model = model.state_dict()
            print('Best model score : %.4f'%(best_score))
            torch.save(best_model, path + 'Cls' + '.pth')

        torch.save(model.state_dict(), path + 'Cls_e' + str(epoch) + '.pth')
    

    ############# Start the test #############

    pretrained_dict = torch.load(r'./results/MaskCN/Cls.pth')
    model.load_state_dict(pretrained_dict)    

    [test_acc_b, test_auc_b, test_AP_b, test_sens_b, test_spec_b, test_acc_c, test_auc_c, test_AP_c, test_sens_c, test_spec_c, \
     test_acc_h, test_auc_h, test_AP_h, test_sens_h, test_spec_h, test_acc_s, test_auc_s, test_AP_s, test_sens_s, test_spec_s, \
     test_acc_t, test_auc_t, test_AP_t, test_sens_t, test_spec_t] = val_mode_Scls(testloader, model)


    line_test_b = "test%d:tacc_b=%f,tauc_b=%f,tAP_b=%f,tsens_b=%f,tspec_b=%f \n" % (
    epoch, test_acc_b, test_auc_b, test_AP_b, test_sens_b, test_spec_b)
    line_test_c = "test%d:tacc_c=%f,tauc_c=%f,tAP_c=%f,tsens_c=%f,tspec_c=%f \n" % (
    epoch, test_acc_c, test_auc_c, test_AP_c, test_sens_c, test_spec_c)
    line_test_h = "test%d:tacc_h=%f,tauc_h=%f,tAP_h=%f,tsens_h=%f,tspec_h=%f \n" % (
    epoch, test_acc_h, test_auc_h, test_AP_h, test_sens_h, test_spec_h)
    line_test_s = "test%d:tacc_s=%f,tauc_s=%f,tAP_s=%f,tsens_s=%f,tspec_s=%f \n" % (
    epoch, test_acc_s, test_auc_s, test_AP_s, test_sens_s, test_spec_s)
    line_test_t = "test%d:tacc_t=%f,tauc_t=%f,tAP_t=%f,tsens_t=%f,tspec_t=%f \n" % (
    epoch, test_acc_t, test_auc_t, test_AP_t, test_sens_t, test_spec_t)

    print(line_test_b)
    print(line_test_c)
    print(line_test_h)
    print(line_test_s)
    print(line_test_t)

    f = open(f_path, "a+")
    f.write(line_test_b)    
    f.write(line_test_c)
    f.write(line_test_h)
    f.write(line_test_s)
    f.write(line_test_t)


if __name__ == '__main__':
    main()

