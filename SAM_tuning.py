import torch
import numpy as np
import torch.backends.cudnn as cudnn
import os
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from utils.metrics import dice_loss, HausdorffLoss
import matplotlib.pyplot as plt
from dataset.my_dataset import MyDataSet_seg
from torch.utils import data
from skimage import io
import torch.nn.functional as F
import warnings
from segment_anything import sam_model_registry
from torch.nn.functional import threshold, normalize




torch.manual_seed(0)
warnings.filterwarnings("ignore")


model_urls = {'SAM_B': 'weights/sam_vit_b.pth'}

INPUT_SIZE = '1024, 1024'     
w, h = map(int, INPUT_SIZE.split(','))
LEARNING_RATE = 0.000005            
NUM_CLASSES = 2             
EPOCH = 50
BATCH_SIZE = 1            # 16
NAME = 'SAM_tuning/'



def val_mode_seg(valloader, model, path, epoch):
    dice = []
    sen = []
    spe = []
    acc = []
    jac_score = []
    total = []
    only_hd = []

    # mask_generator = SamAutomaticMaskGenerator(model)

    for index, batch in enumerate(valloader):

        data, mask, name = batch            
        data = data.cuda()
        data_sam = data[0].cpu().numpy()
        # tensor(0.8980, device='cuda:0') tensor(0., device='cuda:0') tensor(1.) tensor(0.)
        labels = mask.cuda()               
        mask = mask[0].data.numpy()         
        val_mask = np.int64(mask > 0)      
        
        data_sam = data_sam.transpose((1, 2, 0)) 
        data_sam = (data_sam * 255).astype(np.uint8)

        
        with torch.no_grad():
          image_embedding = model.image_encoder(data)
          sparse_embeddings, dense_embeddings = model.prompt_encoder(
              points=None,
              boxes=None,
              masks=None,
          )

          low_res_masks, iou_predictions = model.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
          )
          
          # upscaled_masks = model.postprocess_masks(low_res_masks, (1024, 1024), (1024, 1024)).cuda()
          # binary_mask = normalize(threshold(upscaled_masks, 0.0, 0))  # torch.Size([1, 1, 1024, 1024])
          
          low_res_pred = torch.sigmoid(low_res_masks)       # (1, 1, 256, 256)

          low_res_pred = F.interpolate(
              low_res_pred,
              size=(1024, 1024),
              mode="bilinear",
              align_corners=False,
          )                        

          low_res_pred = low_res_pred.squeeze(1).cpu().numpy()  
          binary_mask = (low_res_pred > 0.5).astype(np.uint8)
        
        labels_copy = labels      

        HD_func = HausdorffLoss()
        pred = torch.tensor(binary_mask).cuda() 
        
        hd_loss = HD_func(pred, labels_copy[:, 0, :, :])
        
        only_hd.append(hd_loss.cpu().data.numpy())
        pred_arg = pred.squeeze(0).cpu().data.numpy()

        print(val_mask[0].sum(), val_mask[0].shape, pred_arg.sum(), pred_arg.shape)

        y_true_f = val_mask[0].reshape(val_mask[0].shape[0]*val_mask[0].shape[1])    
        y_pred_f = pred_arg.reshape(pred_arg.shape[0]*pred_arg.shape[1])             

        intersection = np.float64(np.sum(y_true_f * y_pred_f))
        dice.append((2. * intersection) / (np.sum(y_true_f) + np.sum(y_pred_f)))
        sen.append(intersection / np.sum(y_true_f))
        intersection0 = np.float64(np.sum((1 - y_true_f) * (1 - y_pred_f)))
        spe.append(intersection0 / np.sum(1 - y_true_f))
        acc.append(accuracy_score(y_true_f, y_pred_f))
        jac_score.append(intersection / (np.sum(y_true_f) + np.sum(y_pred_f) - intersection))
        
        
        # if index in [108]:          
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
        
    return np.array(acc), np.array(dice), np.array(sen), np.array(spe), np.array(jac_score), np.array(only_hd)



def Jaccard(pred_arg, mask):

    pred_arg = pred_arg.cpu().data.numpy()                         
    
    mask = mask.cpu().data.numpy()

    y_true_f = mask[:, 0, :, :].reshape(mask[:, 0, :, :].shape[0] * mask[:, 0, :, :].shape[1] * mask[:, 0, :, :].shape[2]) 
    y_pred_f = pred_arg.reshape(pred_arg.shape[0] * pred_arg.shape[1] * pred_arg.shape[2]) 

    intersection = np.float64(np.sum(y_true_f * y_pred_f))
    jac_score = intersection / (np.sum(y_true_f) + np.sum(y_pred_f) - intersection)

    return jac_score



def main():
    """Create the network and start the training."""
    
    cudnn.benchmark = True
    cudnn.enabled = True
    
    ############# Create coarse segmentation network

    model = sam_model_registry["vit_b"](checkpoint=None)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)      # .mask_decoder

    scheduler_lr = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', 
                                    patience=10, verbose=1, factor=0.5, min_lr=1e-8)

    ############# Load pretrained weights
    
    pretrained_dict = torch.load(model_urls['SAM_B'])
    net_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in net_dict) and (v.shape == net_dict[k].shape)}
    net_dict.update(pretrained_dict)
    model.load_state_dict(net_dict)
    

    model.cuda()
    model.train()
    model.float()                 
    
    loss_fn = torch.nn.MSELoss()  


    ############# Load training and validation data

    '''
    data_train_img_root = './dataset/patient1_img_precise/seg/train/'
    data_train_label_root = './dataset/patient1_label_precise/seg/train/'
    data_train_list = './dataset/patient1_img_precise/cls/Training_cls_new.txt' # './dataset/patient1_label_new/seg/training/Training_seg.txt'
    trainloader = data.DataLoader(MyDataSet_seg(data_train_img_root, data_train_label_root, data_train_list, crop_size=(w, h)),
                                    batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
                                     
    data_val_img_root = './dataset/patient1_img_precise/seg/validation/'
    data_val_label_root = './dataset/patient1_label_precise/seg/validation/'
    data_val_list = './dataset/patient1_img_precise/cls/Validation_cls_new.txt' # './dataset/patient1_label_new/seg/validation/Validation_seg.txt'
    valloader = data.DataLoader(MyDataSet_seg(data_val_img_root, data_val_label_root, data_val_list, crop_size=(w, h)), 
                                    batch_size=1, shuffle=False, num_workers=2, pin_memory=True, drop_last=True)
                                     
    data_test_img_root = './dataset/patient1_img_precise/seg/test/'
    data_test_label_root = './dataset/patient1_label_precise/seg/test/'
    data_test_list = './dataset/patient1_img_precise/cls/Testing_cls_new.txt' # './dataset/patient1_label_new/seg/validation/Validation_seg.txt'
    testloader = data.DataLoader(MyDataSet_seg(data_test_img_root, data_test_label_root, data_test_list, crop_size=(w, h)), 
                                    batch_size=1, shuffle=False, num_workers=2, pin_memory=True, drop_last=True)
    '''

    data_train_img_root = './dataset/patient1_img_precise/cls/separate_txt/only_soft/train/'
    data_train_label_root = './dataset/patient1_label_precise/cls/separate_txt/only_soft/train/'         
    data_train_list = './dataset/patient1_img_precise/cls/separate_txt/only_soft/Training_cls.txt'
    trainloader = data.DataLoader(MyDataSet_seg(data_train_img_root, data_train_label_root, data_train_list, crop_size=(w, h)),
                                  batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    
    data_val_img_root = './dataset/patient1_img_precise/cls/separate_txt/only_soft/validation/'
    data_val_label_root = './dataset/patient1_label_precise/cls/separate_txt/only_soft/validation/'         
    data_val_list = './dataset/patient1_img_precise/cls/separate_txt/only_soft/Validation_cls.txt'
    valloader = data.DataLoader(MyDataSet_seg(data_val_img_root, data_val_label_root, data_val_list, crop_size=(w, h)), batch_size=1, shuffle=False,
                                num_workers=2, pin_memory=True, drop_last=True)

    data_test_img_root = './dataset/patient1_img_precise/cls/separate_txt/only_soft/test/'
    data_test_label_root = './dataset/patient1_label_precise/cls/separate_txt/only_soft/test/'        
    data_test_list = './dataset/patient1_img_precise/cls/separate_txt/only_soft/Testing_cls.txt'
    testloader = data.DataLoader(MyDataSet_seg(data_test_img_root, data_test_label_root, data_test_list, crop_size=(w, h)), batch_size=1, shuffle=False,
                                num_workers=2, pin_memory=True, drop_last=True)
    

    path = 'results/' + NAME
    if not os.path.isdir(path):
        os.mkdir(path)
    f_path = path + 'output_sam.txt'


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
        
        model.train()
        for i_iter, batch in tqdm(enumerate(trainloader)):

            images, labels, name = batch            
            images = images.cuda()                  
            labels = labels.cuda()                   

            optimizer.zero_grad()


            with torch.no_grad():
                image_embedding = model.image_encoder(images)
                sparse_embeddings, dense_embeddings = model.prompt_encoder(
                    points=None,
                    boxes=None,
                    masks=None,
                )

            low_res_masks, iou_predictions = model.mask_decoder(
              image_embeddings=image_embedding,
              image_pe=model.prompt_encoder.get_dense_pe(),
              sparse_prompt_embeddings=sparse_embeddings,
              dense_prompt_embeddings=dense_embeddings,
              multimask_output=False,
            )

            upscaled_masks = model.postprocess_masks(low_res_masks, (1024, 1024), (1024, 1024)).cuda()
            binary_mask = normalize(threshold(upscaled_masks, 0.0, 0))      # torch.Size([2, 1, 1024, 1024])

            term = loss_fn(binary_mask.squeeze(1), labels[:, 0, :, :])                                          
            
            if i_iter % 50 == 0:
                print(term)

            term.backward()
            optimizer.step()

            train_loss_total.append(term.cpu().data.numpy())
            train_jac.append(Jaccard(binary_mask.squeeze(1), labels))        
            train_loss_term1.append(term.cpu().data.numpy()) 
            

        print("train_epoch%d: lossTotal=%f, Jaccard=%f, loss_bce=%f, loss_haus=%f \n" % (epoch, np.nanmean(train_loss_total), np.nanmean(train_jac)
                                                        , np.nanmean(train_loss_term1), np.nanmean(train_loss_term2)))

        ############# Start the validation
        [vacc, vdice, vsen, vspe, vjac_score, vhd_loss] = val_mode_seg(valloader, model, path, epoch)
        line_val = "val%d: vacc=%f, vdice=%f, vsensitivity=%f, vspecifity=%f, vjac=%f, vhdloss=%f \n" % \
                (epoch, np.nanmean(vacc), np.nanmean(vdice), np.nanmean(vsen), np.nanmean(vspe),
                    np.nanmean(vjac_score), np.nanmean(vhd_loss))

        scheduler_lr.step(np.nanmean(vjac_score))

        print(line_val)
        f = open(f_path, "a+")
        f.write(line_val)

        ############# Plot val curve

        val_hd_loss.append(np.nanmean(vhd_loss))
        val_jac.append(np.nanmean(vjac_score))
        val_dice.append(np.nanmean(vdice))

        
        plt.figure()
        plt.plot(val_jac, label='val jaccard', color='blue', linestyle='--')
        plt.legend(loc='best')

        plt.savefig(os.path.join(path, 'jaccard_sam.png'))
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
            torch.save(best_model, path + 'SAM' + '.pth')

    


    HD_Loss = np.array(val_hd_loss)
    np.save('./results/SAM_tuning/HD_Loss_sam', HD_Loss)

    IOU = np.array(val_jac)
    np.save('./results/SAM_tuning/IOU_sam', IOU)  

    Dice = np.array(val_dice)
    np.save('./results/SAM_tuning/Dice_sam', Dice)      


    plt.figure(1)
    # plt.plot(Loss, label='Loss')
    plt.plot(HD_Loss, label='HD_Loss')
    plt.title('Loss')
    plt.legend()
    # plt.axis([0, None, 0, 1])
    plt.savefig(path + 'Loss_sam')
    plt.show()

    plt.figure(2)
    plt.plot(IOU, label='IOU')
    plt.plot(Dice, label='Dice')
    plt.title('IOU')
    plt.legend()
    plt.savefig(path + 'IOU_sam')
    plt.show()

    
    ############# Start the test
    pretrained_dict = torch.load(r'./results/SAM_tuning/SAM.pth')
    model.load_state_dict(pretrained_dict)    

    [tacc, tdice, tsen, tspe, tjac_score, thd_loss, ttotal_loss] = val_mode_seg(valloader, model, path, epoch)   
    line_test = "test%d: tacc=%f, tdice=%f, tsensitivity=%f, tspecifity=%f, tjac=%f, thdloss=%f \n" % \
            (epoch, np.nanmean(tacc), np.nanmean(tdice), np.nanmean(tsen), np.nanmean(tspe),
                np.nanmean(tjac_score), np.nanmean(thd_loss))

    print(line_test)
    f = open(f_path, "a+")
    f.write(line_test)

    
if __name__ == '__main__':
    main()


