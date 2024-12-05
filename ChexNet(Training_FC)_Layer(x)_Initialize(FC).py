import os
import csv
import time
import copy
import torch

import numpy as np
import pandas as pd
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision import models
from sklearn.metrics import roc_auc_score, f1_score
from torch import optim
from torch.optim import Adam
from PIL import Image
from torchinfo import summary
# ---------------------------------------------- main() 실행하는 코드 부분 ----------------------------------------------- #

def main():
    
# 0.이미지, 라벨 경로 지정. 
    df = pd.read_csv('C:/Users/61156_D1-3/Desktop/dataset/LABEL_rearrange.csv') 
    image_dir = 'C:/Users/61156_D1-3/Desktop/dataset/images/'

# 1.Train / Valid / Test 분할 
# Ratio 7:1:2 = 78468:11219:22433 
    train_set = df[df['fold']=='train']
    val_set = df[df['fold']=='val']
    test_set = df[df['fold']=='test']
    print(f'----------------------------------------------------------------')
    print(f'Training set size: {len(train_set)}')
    print(f'Validation set size: {len(val_set)}')
    print(f'Test set size: {len(test_set)}')
    print(f'----------------------------------------------------------------')

    num_epochs = 100
    num_classes = 14

# 2. Data_Transform           
# normalized pixel = (input pixel - mean) / std   -> [R,G,B] 3channel에 대한 평균과 표준편차
# 1024x1024 image -(Resize)-> 224x224.
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean,std)
        ]),
        'evaluate': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
    }
    
    # Image_Dataset
    train_dataset = CXRDataset( dataframe=train_set, root_dir=image_dir, transform=data_transforms['train'] )
    val_dataset   = CXRDataset( dataframe=val_set  , root_dir=image_dir, transform=data_transforms['evaluate'])
    test_dataset  = CXRDataset( dataframe=test_set , root_dir=image_dir, transform=data_transforms['evaluate'])
    
    # Pytorch에서 DataLoader에 대한 설명 (https://tutorials.pytorch.kr/beginner/basics/data_tutorial.html)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True , num_workers=8)
    val_loader   = DataLoader(val_dataset  , batch_size=16, shuffle=False, num_workers=8)
    test_loader  = DataLoader(test_dataset , batch_size=16, shuffle=False, num_workers=8)

    dataloaders   = {'train': train_loader      , 'val': val_loader}
    dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}

    
# 3. Define BASE_MODEL
    checkpoint = torch.load('./model.pth.tar')
    base_model = DenseNet121(out_size=14)

    state_dict = checkpoint['state_dict']
    new_state_dict = {}
    
    #for k, v in state_dict.items():
    #   new_key = k.replace('norm.1', 'norm1').replace('norm.2', 'norm2').replace('conv.2','conv2').replace('conv.1','conv1').replace('classifier.0' , 'classifier.1')
    #   new_key = new_key.replace('module.', '')
    
    for k, v in state_dict.items():
        new_key = k.replace('norm.1', 'norm1').replace('norm.2', 'norm2').replace('conv.2','conv2').replace('conv.1','conv1')
        new_key = new_key.replace('module.', '')
        
#        if 'classifier.0' in new_key:
#            new_key = new_key.replace('classifier.0', 'classifier.1')
#        elif 'classifier.1' in new_key:
#            new_key = new_key.replace('classifier.1', 'classifier.0')
        new_state_dict[new_key] = v
        
#Cmodule.densenet121.classifier.0.bias, densenet121.classifier.0.bias
# 4. GPU 사용 설정
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# GPU가 사용 가능한 경우 첫 번째 GPU(cuda:0)를 사용하고, 그렇지 않으면 CPU를 사용하도록 설정.
# model.to(device): 모델을 지정된 장치(device)로 이동. 이를 통해 모델의 모든 parameter와 버퍼가 GPU나 CPU로 전송되도록 함. 
    gpu_count = torch.cuda.device_count()
    print("Available GPU count:" + str(gpu_count))
  #  base_model.load_state_dict(new_state_dict, strict=False)
    base_model = base_model.to(device)
    summary(base_model)
    
    
# 5. Hyper_parameter 설정
# 손실함수(Loss Function): 교차 엔트로피 손실 함수(BCELoss; Binary-Cross Entropy Loss)
# BCELoss함수 Pytorch 공식 문서 (https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html)
# - model.parameters()         = 모델의 모든 parameter를 반환함. (weight와 bias 포함)
# - lambda p: p.requires_grad  = lambda 함수로 'p'가 학습 가능한 파라미터(requires_grad=True)인지 확인함.
# - filter ( ... )             = 즉, 학습 가능한 파라미터들만 필터링하여 optimizer에게 전달.
# - weight_decay               = overfitting 방지를 위한 가중치 감쇠항(정규화; L2-Regularization)
    criterion = nn.BCELoss(reduction='sum')
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, base_model.parameters()),
        lr=0.001,
        betas=(0.9,0.999),
        weight_decay=1e-6
        )


# 6. Scheduler 설정
# PyTorch의 학습률 스케줄러 중 하나인 'ReduceLROnPlateau' 설정. 모델의 성능이 향상되지 않을 때 학습률을 동적으로 조정하는(줄여주는) 역할.
# 매개변수1) optimizer = 위 (5.)에서의 Adam 옵티마이저 전달
# 매개변수2) mode      = 관찰 metric이 최소화(min)되는지 최대화(max)되는지 설정. min -> 손실이 감소하는 경우를 모니터링
# 매개변수3) factor    = 학습률을 줄이는 비율 (0.1 = 10%로 감소)
# 매개변수4) patience  = 성능 향상이 없더라도 학습률을 줄이기 전에 기다리는 epoch 수
# 매개변수5) cooldown  = 학습률이 줄어든 후, 다시 성능을 모니터링 하기 전에 기다리는 epoch 수
# 매개변수6) verbose   = 학습률이 줄어들 때마다 이를 알리는 메세지를 출력할지의 여부. 
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=1, cooldown=0, verbose=True)


# 7. 사용자 정의 함수 "model_training"과 "model_evaluating" 사용.
    trained_model, best_epoch = model_training   (base_model, criterion, optimizer, num_epochs, dataloaders, dataset_sizes, scheduler=scheduler)
    test_loss                 = model_evaluating (trained_model, criterion, test_loader, len(test_dataset))
    
    
# 8. Test 결과 출력 (AUROC, F1_score)
    auroc_values, best_threshold, best_f1 = compute_metric(test_loader, trained_model, device, num_classes)
    print(f"Best Average F1 score: {best_f1}")
    print(f"Best Threshold: {best_threshold}")

    label_list = ['Atelectasis','Cardiomegaly','Effusion','Infiltration','Mass','Nodule','Pneumonia','Pneumothorax', 'Consolidation','Edema','Emphysema','Fibrosis', 'Pleural_Thickening','Hernia']
    order_list = [1, 10, 2, -1, 3, 4, 5, 0, 7, -2, 6, -3, -5, 8]
    reordered_label_list = [label_list[i] for i in order_list]
    reordered_auroc_value = [auroc_values[i] for i in order_list]
        
    for i,auroc in enumerate(reordered_auroc_value):
        print(f'AUROC for class {reordered_label_list[i]}: {auroc:.4f}')

    print('*' * 60)




# ----------------------------------------------- Custom Class 01. CXRDataset ---------------------------------------------- #
# __init__    : class의 생성자. 객체가 생성될 때 자동으로 호출됨. 객체의 초기화 작업 수행.
# __len__     : len() 함수가 객체에 대해 호출될 때, 자동으로 호출
# __getitem__ : 객체의 특정 인덱스에 접근할 때 호출되는 메서드로, 객체의 indexing연산 obj[idx]이 수행될 때 자동으로 호출
#               ex) image, labels = dataset[0] ->  __getitem__ 메서드가 호출되어 첫 번째 샘플을 반환함. 


class CXRDataset(Dataset):

# Class를 초기화할 때, dataframe, root_dir, transform(default=none) 함수 또는 객체를 초기화함
    def __init__(self, dataframe, root_dir, transform=None):
        self.dataframe = dataframe
        self.root_dir = root_dir
        self.transform = transform

# 데이터셋의 전체 sample 수 반환
    def __len__(self):
        return len(self.dataframe)

# dataframe에서 특정 sample에 접근하기 위한 'index'(idx) 
    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.dataframe.iloc[idx, 0])    # root_dir에서 idx에 해당하는 이미지 로드
        image = Image.open(img_name).convert('RGB')                            # idx에 해당하는 이미지를 RGB 포멧으로 변환
        labels = self.dataframe.iloc[idx, 6:20].values.astype('float')         # dataframe에서 label을 추출하여 (.csv파일의 6~20 column) float형으로 변환
        if self.transform:                                                     # 변환함수 'transform'이 주어지면 이미지를 변환
            image = self.transform(image)
        return image, labels                                                   # 튜플(Tuple)형태로 image와 label 반환

# ----------------------------------------------- Custom Class 02. DenseNet121 ---------------------------------------------- #
class DenseNet121(nn.Module):

    def __init__(self, out_size):
        super(DenseNet121, self).__init__()
        self.densenet121 = models.densenet121(weights='DEFAULT')
        for param in self.densenet121.parameters():
            param.requires_grad=False
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Identity()
        self.fc1 = nn.Sequential(
            nn.Dropout(p=0.35),
            nn.Linear(num_ftrs,out_size),
            nn.Sigmoid()
        )
        
    def forward(self, image):
        x1 = self.densenet121(image)
        x1 = self.fc1(x1)
        return x1

#         for param in self.densenet121.parameters():
#             param.requires_grad=False
#         num_ftrs = self.densenet121.classifier.in_features
# --------------------------------------------- Custom Func 02. model_training --------------------------------------------- #

def model_training(
    model, 
    criterion, 
    optimizer, 
    num_epochs, 
    dataloaders, 
    dataset_sizes, 
    scheduler=None, 
    checkpoint_path='checkpoint', 
    checkpoint_path2='checkpoint_best'):


    # CSV 파일 초기화 (헤더 작성)
    file_exists = os.path.isfile('training_log.csv')
    with open('training_log.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['Epoch', 'Train Loss', 'Val Loss'])  # 헤더 작성

    time_start = time.time()
    # Checkpoint 불러오기
    model, optimizer, start_epoch, best_loss = load_checkpoint(model, optimizer, checkpoint_path)
    best_epoch = start_epoch - 1

    for epoch in range(start_epoch, num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 30)
        
        epoch_loss_values = {}


# dataloaders(자료형: dictionary)의 phase(key) 구성=  'train' : train_loader, 'val' : val_loader
        for phase in ['train', 'val']:
        
        # dataloaders = train_loader
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            total_done = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device).float()
                labels = labels.to(device).float()
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item()
                total_done += inputs.size(0)

                if total_done % (300 * inputs.size(0)) == 0:
                    print(f"Completed {total_done} so far in epoch")

            epoch_loss = running_loss / dataset_sizes[phase]
            print(f'{phase} Loss: {epoch_loss:.4f}')

            epoch_loss_values[phase] = epoch_loss

        # dataloaders = val_loader
            if phase == 'val':
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_epoch = epoch
                    save_checkpoint({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'best_loss': best_loss,
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, checkpoint_path2)
                    save_checkpoint({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'best_loss': best_loss,
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, checkpoint_path)
                else: 
                    save_checkpoint({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'best_loss': best_loss,
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, checkpoint_path)
                if scheduler:
                    scheduler.step(epoch_loss)
                    print(f"Scheduler step complete. New learning rate: {optimizer.param_groups[0]['lr']}")
        
# 5회 이상 best_epoch가 업데이트 되지 않으면 train 종료
        if (((epoch - best_epoch)>=15) & (epoch>=20)): 
            print("no improvement in 15 epoch, break")
            break
            
        # CSV에 에포크 손실 기록 추가
        with open('training_log.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch, epoch_loss_values['train'], epoch_loss_values['val']])
    
    time_finish = time.time() - time_start
    print('*' * 60)
    print(f'Training complete in {time_finish // 60:.0f}m {time_finish % 60:.0f}s')
    print(f'Best val loss: {best_loss:.4f}')

    model.load_state_dict(torch.load(checkpoint_path2)['model_state_dict'])

    return model, best_epoch

# -------------------------------------------- Custom Func 03. model_evaluating -------------------------------------------- #

def model_evaluating(model, criterion, dataloader, dataset_size):
    model.eval()
    running_loss = 0.0

    for inputs, labels in dataloader:
        inputs = inputs.to(device).float()
        labels = labels.to(device).float()

        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        running_loss += loss.item()

    epoch_loss = running_loss / dataset_size
    print('Test Loss: {:.4f}'.format(epoch_loss))

    return epoch_loss

# --------------------------------------------- Custom Func 04. compute_auroc --------------------------------------------- #

def compute_metric(dataloader, model, device, num_classes):
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device).float()
            outputs = model(inputs)
            all_labels.append(labels.cpu().numpy())
            all_preds.append(outputs.cpu().numpy())
    all_labels = np.concatenate(all_labels, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    
    auroc_values = []
    best_threshold =0.0
    best_f1 = 0.0

    for i in range(num_classes):
        auroc = roc_auc_score(all_labels[:, i], all_preds[:, i])
        auroc_values.append(auroc)
    
    for threshold in np.arange(0.0, 1.0, 0.001):
        f1_values= []
        for i in range(num_classes):
            binarized_preds = (all_preds[:, i] > threshold).astype(int)
            f1 = f1_score(all_labels[:, i], binarized_preds)
            f1_values.append(f1)

        avg_f1_score = np.mean(f1_values)
        if best_f1<avg_f1_score:
            best_f1=avg_f1_score
            best_threshold=threshold
        
    return auroc_values, best_threshold, best_f1


# ------------------------------------------ Custom Func 06. save&load_checkpoint ------------------------------------------ #

def save_checkpoint(state, filename='results/checkpoint.pth.tar'):
    torch.save(state, filename)

def load_checkpoint(model, optimizer, filename='results/checkpoint.pth.tar'):
    if os.path.isfile(filename):
        print(f"=> loading checkpoint '{filename}'")
        checkpoint = torch.load(filename,map_location=device)
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['best_loss']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"=> loaded checkpoint '{filename}' (epoch {checkpoint['epoch']})")
    else:
        print(f"=> no checkpoint found at '{filename}'")
        start_epoch = 1
        best_loss = float('inf')
        last_train_loss = -1
    return model, optimizer, start_epoch, best_loss

# ----------------------------------------------- 실제로 .py에서 실행되는 부분 --------------------------------------------- #

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    main()