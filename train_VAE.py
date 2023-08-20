import torch
import numpy as np
import torch.backends.cudnn as cudnn
import os
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from utils.metrics import Dice_Loss, kl_div_zmuv, HausdorffLoss
import matplotlib.pyplot as plt
from dataset.my_dataset import MyDataSet_seg
from torch.utils import data
from models.VAE import VAE
from utils.rejection_sampling import RejectionSampler
import torch.nn.functional as F
from torch.autograd import Variable
from functools import reduce
from operator import mul
import itertools
import h5py




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(0)

INPUT_SIZE = '320, 240'     
w, h = map(int, INPUT_SIZE.split(','))
LEARNING_RATE = 0.0001
NUM_CLASSES = 2           
EPOCH = 50
BATCH_SIZE = 16
NAME = 'VAE_train/'
NUM_RS = 10000





def val_mode_seg(valloader, model, path, epoch):
    dice = []
    sen = []
    spe = []
    acc = []
    jac_score = []

    for index, batch in enumerate(valloader):

        data, mask, name = batch            # torch.Size([1, 3, 240, 320])  torch.Size([1, 3, 240, 320])
        data = data.cuda()
        mask = mask.cuda()

        model.eval()
        with torch.no_grad():
            pred, z, mu, log_var = model(mask)              # torch.Size([1, 2, 240, 320])  

        pred_arg = pred[:, 0, :, :].cpu().data.numpy()
        pred_arg = (pred_arg > 0.5).astype(np.uint8)
        mask = mask.cpu().data.numpy()

        y_true_f = mask[:, 0, :, :].reshape(mask[:, 0, :, :].shape[0] * mask[:, 0, :, :].shape[1] * mask[:, 0, :, :].shape[2])  
        y_pred_f = pred_arg.reshape(pred_arg.shape[0] * pred_arg.shape[1] * pred_arg.shape[2])  

        intersection = np.float64(np.sum(y_true_f * y_pred_f))
        dice.append((2. * intersection) / (np.sum(y_true_f) + np.sum(y_pred_f)))
        sen.append(intersection / np.sum(y_true_f))
        intersection0 = np.float64(np.sum((1 - y_true_f) * (1 - y_pred_f)))
        spe.append(intersection0 / np.sum(1 - y_true_f))
        acc.append(accuracy_score(y_true_f, y_pred_f))
        jac_score.append(intersection / (np.sum(y_true_f) + np.sum(y_pred_f) - intersection))

        if index in [102]:         
            fig = plt.figure()
            ax = fig.add_subplot(131)
            ax.imshow(data[0].cpu().data.numpy().transpose(1, 2, 0))
            ax.axis('off')
            ax = fig.add_subplot(132)
            mask = mask[0].transpose((1, 2, 0))      
            ax.imshow(mask)
            ax.axis('off')
            ax = fig.add_subplot(133)
            ax.imshow(pred_arg[0])                   
            ax.axis('off')
            fig.suptitle('original image, ground truth mask, predicted mask',fontsize=6)
            fig.savefig(path + name[0][11:-4] + '_e' + str(epoch) + '.png', dpi=200, bbox_inches='tight')
            ax.cla()
            fig.clf()
            plt.close()

    return np.array(acc), np.array(dice), np.array(sen), np.array(spe), np.array(jac_score)


def Jaccard(pred_arg, mask): 

    pred_arg[:, 0, :, :].cpu().data.numpy() 
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
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler_lr = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', 
                                        patience=3, verbose=1, factor=0.3, min_lr=1e-6)


    model.train()
    model.float()                 
    


    ############# Load training and validation data
    data_train_img_root = './dataset/patient1_img_precise/seg/train_VAE/'
    data_train_label_root = './dataset/patient1_label_precise/seg/train_VAE/'
    data_train_list = './dataset/patient1_img_precise/cls/Training_VAE.txt' 
    trainloader = data.DataLoader(MyDataSet_seg(data_train_img_root, data_train_label_root, data_train_list, crop_size=(w, h)),
                                    batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
                                     
    data_val_img_root = './dataset/patient1_img_precise/seg/validation_VAE/'
    data_val_label_root = './dataset/patient1_label_precise/seg/validation_VAE/'
    data_val_list = './dataset/patient1_img_precise/cls/Validation_VAE.txt' 
    valloader = data.DataLoader(MyDataSet_seg(data_val_img_root, data_val_label_root, data_val_list, crop_size=(w, h)), 
                                    batch_size=1, shuffle=False, num_workers=2, pin_memory=True, drop_last=True)
                                     
    data_test_img_root = './dataset/patient1_img_precise/seg/test_VAE/'
    data_test_label_root = './dataset/patient1_label_precise/seg/test_VAE/'
    data_test_list = './dataset/patient1_img_precise/cls/Testing_VAE.txt' 
    testloader = data.DataLoader(MyDataSet_seg(data_test_img_root, data_test_label_root, data_test_list, crop_size=(w, h)), 
                                    batch_size=1, shuffle=False, num_workers=2, pin_memory=True, drop_last=True)

    path = 'results/' + NAME
    if not os.path.isdir(path):
        os.mkdir(path)
    f_path = path + 'output_vae.txt'

    val_jac = []
    best_score = 0.


    
    ############# Start the training
    for epoch in range(EPOCH):

        train_loss_total = []
        train_jac = []

        for i_iter, batch in tqdm(enumerate(trainloader)):

            images, labels, name = batch
            labels = labels.cuda()                  # torch.Size([8, 3, 240, 320])    

            optimizer.zero_grad()

            model.train()
            labels_reconst, z, mu, log_var = model(labels)      
            # torch.Size([2, 1, 240, 320]) torch.Size([2, 5]) torch.Size([2, 5]) torch.Size([2, 5])

            term1 = F.binary_cross_entropy(labels_reconst[:, 0, :, :], labels[:, 0, :, :], reduction='sum')
            term2 = kl_div_zmuv(mu, log_var)      

            term = term1 + term2 

            if i_iter % 10 == 0:
                print(term1, term2)

            term.backward()
            optimizer.step()

            train_loss_total.append(term.cpu().data.numpy())
            train_jac.append(Jaccard(labels_reconst[:, 0, :, :], labels))   
    

        print("train_epoch%d: lossTotal=%f, Jaccard=%f \n" % (epoch, np.nanmean(train_loss_total), np.nanmean(train_jac)))


        ############# Start the validation
        [vacc, vdice, vsen, vspe, vjac_score] = val_mode_seg(valloader, model, path, epoch)
        line_val = "val%d: vacc=%f, vdice=%f, vsensitivity=%f, vspecifity=%f, vjac=%f \n" % \
                (epoch, np.nanmean(vacc), np.nanmean(vdice), np.nanmean(vsen), np.nanmean(vspe),
                    np.nanmean(vjac_score))

        scheduler_lr.step(np.nanmean(vjac_score))

        print(line_val)
        f = open(f_path, "a+")
        f.write(line_val)

        ############# Plot val curve
        val_jac.append(np.nanmean(vjac_score))
        plt.figure()
        plt.plot(val_jac, label='val jaccard', color='blue', linestyle='--')
        plt.legend(loc='best')

        plt.savefig(os.path.join(path, 'jaccard.png'))
        plt.clf()
        plt.close()
        plt.show()

        plt.close('all')

        ############# Save network

        total_score = np.nanmean(vjac_score)       

        if total_score > best_score:
            best_score = total_score
            best_model = model.state_dict()
            print('Best model score : %.4f'%(best_score))
            torch.save(best_model, path + 'VAE_train' + '.pth')

    

    ############# Start the test
    pretrained_dict = torch.load(r'./results/VAE_train/VAE_train.pth')
    model.load_state_dict(pretrained_dict)    

    [tacc, tdice, tsen, tspe, tjac_score] = val_mode_seg(testloader, model, path, epoch)    
    line_test = "test%d: tacc=%f, tdice=%f, tsensitivity=%f, tspecifity=%f, tjac=%f \n" % \
            (epoch, np.nanmean(tacc), np.nanmean(tdice), np.nanmean(tsen), np.nanmean(tspe),
                np.nanmean(tjac_score))

    print(line_test)
    f = open(f_path, "a+")
    f.write(line_test)
    f.close()
    
    # Rejection sampling and save them
    pretrained_dict = torch.load(r'./results/VAE_train/VAE_train.pth')
    model.load_state_dict(pretrained_dict) 

    data_rs = itertools.chain(trainloader, valloader)
    data_rs = itertools.chain(data_rs, testloader)
    num_batches = len(trainloader) + len(valloader)+ len(testloader)         
    
    progress_bar = True
    if progress_bar:
        data_rs = tqdm(data_rs, desc="Encoding groundtruths", total=num_batches, unit="batch")

    model.cpu()
    with torch.no_grad():
        dataset_samples = [model(batch[1])[1].cpu() for batch in data_rs]       # print(batch[1], batch[1].shape, batch[1].dtype) torch.Size([1, 3, 608, 832])

    dataset_samples = torch.cat(dataset_samples).numpy()    

    rejection_sampler = RejectionSampler(dataset_samples, model)  
    rs_samples = rejection_sampler.sample(num_samples=NUM_RS)           
    # M x D numpy array, M: `num_samples` D: the dimensionality of the sampled data


    output_rs_path = './results/VAE_train/RS_samples/'
    if not os.path.isdir(output_rs_path):
        os.mkdir(output_rs_path)

    # Creates an HDF5 dataset containing the samples, split in multiple groups
    rs_samples = {'total': rs_samples}

    with h5py.File(output_rs_path + "rs_samples.hdf5", "w") as dataset:
        for samples_group_name, samples_group in rs_samples.items():
            dataset.create_dataset(samples_group_name, data=samples_group)

    '''

    output_rs_path = './results/VAE_train/RS_samples/'
    f1 = h5py.File(output_rs_path + "rs_samples_1_2000.hdf5", "r")
    f2 = h5py.File(output_rs_path + "rs_samples_2_2000.hdf5", "r")
    f3 = h5py.File(output_rs_path + "rs_samples_3_20000.hdf5", "r")
    f4 = h5py.File(output_rs_path + "rs_samples_4_2000.hdf5", "r")
    f5 = h5py.File(output_rs_path + "rs_samples_5_2000.hdf5", "r")
    f6 = h5py.File(output_rs_path + "rs_samples_6_2000.hdf5", "r")
    f7 = h5py.File(output_rs_path + "rs_samples_7_4000.hdf5", "r")
    f8 = h5py.File(output_rs_path + "rs_samples_8_10000.hdf5", "r")
    f9 = h5py.File(output_rs_path + "rs_samples_9_6000.hdf5", "r")
    f10 = h5py.File(output_rs_path + "rs_samples_10_20000.hdf5", "r")
    f11 = h5py.File(output_rs_path + "rs_samples_11_20000.hdf5", "r")
    f12 = h5py.File(output_rs_path + "rs_samples_12_20000.hdf5", "r")

    f = np.concatenate((f1['total'][:], f2['total'][:], f3['total'][:], f4['total'][:], 
                        f5['total'][:], f6['total'][:], f7['total'][:], f8['total'][:],
                        f9['total'][:], f10['total'][:], f11['total'][:], f12['total'][:]), axis=0)
    print(f.shape)

    rs_samples = {'total': f}

    with h5py.File(output_rs_path + "rs_samples_total.hdf5", "w") as dataset:
        for samples_group_name, samples_group in rs_samples.items():
            dataset.create_dataset(samples_group_name, data=samples_group)    

    f = h5py.File(output_rs_path + "rs_samples.hdf5", "r")
    print(f[('total')])        # <HDF5 dataset "total": shape (1272, 32), type "<f4">
    print(f['total'][:])     
    f.close()
    '''

if __name__ == '__main__':
    main()



