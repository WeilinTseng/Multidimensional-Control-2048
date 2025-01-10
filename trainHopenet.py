import os
import torch
import torch.nn as nn
from torchvision import transforms
import torchvision
import datasets, hopenet
import torch.utils.model_zoo as model_zoo

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import matplotlib
matplotlib.use('TKAgg')

import csv
import matplotlib.pyplot as plt
import numpy as np

# 初始化或創建CSV檔案來儲存評估結果
result_file = 'evaluation_results.csv'
if not os.path.exists(result_file):
    with open(result_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'MAE_Yaw', 'MAE_Pitch', 'MAE_Roll'])  # 加入標題

# 定義忽略的參數
def get_ignored_params(model):
    b = [model.conv1, model.bn1, model.fc_finetune]
    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            if 'bn' in module_name:
                module.eval()  # 設為推論模式
            for name, param in module.named_parameters():
                yield param

# 定義非忽略的參數
def get_non_ignored_params(model):
    b = [model.layer1, model.layer2, model.layer3, model.layer4]
    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            if 'bn' in module_name:
                module.eval()  # 設為推論模式
            for name, param in module.named_parameters():
                yield param

# 定義全連接層的參數
def get_fc_params(model):
    b = [model.fc_yaw, model.fc_pitch, model.fc_roll]
    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            for name, param in module.named_parameters():
                yield param

# 載入已過濾的模型權重
def load_filtered_state_dict(model, snapshot):
    model_dict = model.state_dict()
    snapshot = {k: v for k, v in snapshot.items() if k in model_dict}
    model_dict.update(snapshot)
    model.load_state_dict(model_dict)

# 主程式
if __name__ == '__main__':
    # 配置變數
    gpu_id = 0 if torch.cuda.is_available() else -1  # 判斷是否使用GPU
    num_epochs = 50  # 訓練的epoch數量
    batch_size = 32  # 批次大小
    learning_rate = 0.00001  # 學習率
    data_dir = "300W_LP"  # 資料目錄
    filename_list = "300WLP_FileInFo.txt"  # 檔案列表
    output_string = "300w_lp"  # 輸出字串
    snapshot_path = ""  # 預訓練模型的路徑
    alpha = 1  # 較高的正則化權重

    # 定義和初始化訓練過程中的評估指標
    train_metrics = {
        'epoch': [],
        'MAE_Yaw': [], 'MAE_Pitch': [], 'MAE_Roll': [],  # 預測角度的平均絕對誤差
        'MSE_Yaw': [], 'MSE_Pitch': [], 'MSE_Roll': [],  # 預測角度的均方誤差
        'MAPE_Yaw': [], 'MAPE_Pitch': [], 'MAPE_Roll': [],  # 預測角度的平均絕對百分比誤差
        'MSLE_Yaw': [], 'MSLE_Pitch': [], 'MSLE_Roll': []  # 預測角度的均方對數誤差
    }

    # 定義每個指標的CSV檔案
    metrics_files = {
        'MAE': 'mae_results.csv',
        'MSE': 'mse_results.csv',
        'MAPE': 'mape_results.csv',
        'MSLE': 'msle_results.csv',
    }

    # 如果這些檔案不存在，則創建並初始化它們
    for metric, file_name in metrics_files.items():
        if not os.path.exists(file_name):
            with open(file_name, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Epoch', 'Yaw', 'Pitch', 'Roll', 'Overall'])  # 每個指標的標題

    # 創建必要的目錄
    os.makedirs('output/snapshots', exist_ok=True)
    os.makedirs('output/plots', exist_ok=True)
    torch.backends.cudnn.enabled = True  # 啟用CuDNN加速

    # 初始化ResNet50結構的Hopenet模型
    model = hopenet.Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)

    # 如果有快照路徑，載入預訓練權重
    if snapshot_path:
        saved_state_dict = torch.load(snapshot_path)
        model.load_state_dict(saved_state_dict)
    else:
        # 否則從網路上載入ResNet50的預訓練權重
        load_filtered_state_dict(model, model_zoo.load_url('https://download.pytorch.org/models/resnet50-19c8e357.pth'))

    # 設置設備：使用GPU或CPU
    device = torch.device(f'cuda:{gpu_id}' if gpu_id >= 0 else 'cpu')
    model.to(device)  # 將模型移動到指定設備
    
    print("Loading data.")

    # 資料轉換
    transformations = transforms.Compose([
        transforms.Resize(240),  # 調整圖片大小
        transforms.RandomCrop(224),  # 隨機裁剪
        transforms.ToTensor(),  # 轉換為Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 正規化
    ])

    # 載入資料集
    pose_dataset = datasets.Pose_300W_LP(data_dir, filename_list, transformations)
    train_loader = torch.utils.data.DataLoader(dataset=pose_dataset,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            num_workers=2)

    # 定義模型的Softmax層和損失函數
    softmax = nn.Softmax(dim=1).to(device)
    criterion = nn.CrossEntropyLoss().to(device)  # 用於分類的交叉熵損失
    reg_criterion = nn.MSELoss().to(device)  # 用於回歸的均方誤差損失
    idx_tensor = torch.arange(66, dtype=torch.float32).to(device)  # 用於計算角度的索引

    # 設置優化器，根據不同參數設置學習率
    optimizer = torch.optim.Adam([
        {'params': get_ignored_params(model), 'lr': 0},  # 對於忽略的參數，學習率為0
        {'params': get_non_ignored_params(model), 'lr': learning_rate},  # 其他層設學習率
        {'params': get_fc_params(model), 'lr': learning_rate * 5}  # 全連接層的學習率為學習率的5倍
    ], lr=learning_rate)

    print('Ready to train network.')
    # 訓練過程
    for epoch in range(num_epochs):
        # 初始化計數器，用於累積每個指標的損失和MAE
        total_mae_yaw, total_mae_pitch, total_mae_roll = 0, 0, 0
        total_mse_yaw, total_mse_pitch, total_mse_roll = 0, 0, 0
        total_mape_yaw, total_mape_pitch, total_mape_roll = 0, 0, 0
        total_msle_yaw, total_msle_pitch, total_msle_roll = 0, 0, 0
        correct_predictions = 0
        total_samples = 0

        # 訓練迴圈，逐批處理數據
        for i, (images, labels, cont_labels, name) in enumerate(train_loader):
            images = images.to(device)  # 將圖片轉移到GPU或CPU
            label_yaw = labels[:, 0].to(device)  # 目標的俯仰角度
            label_pitch = labels[:, 1].to(device)  # 目標的偏航角度
            label_roll = labels[:, 2].to(device)  # 目標的滾轉角度
            
            label_yaw_cont = cont_labels[:, 0].to(device)  # 連續的俯仰角度標籤
            label_pitch_cont = cont_labels[:, 1].to(device)  # 連續的偏航角度標籤
            label_roll_cont = cont_labels[:, 2].to(device)  # 連續的滾轉角度標籤
            
            # 限制角度範圍
            label_yaw = torch.clamp(label_yaw, min=0, max=65)
            label_pitch = torch.clamp(label_pitch, min=0, max=65)
            label_roll = torch.clamp(label_roll, min=0, max=65)
            
            # 前向傳播
            yaw, pitch, roll = model(images)

            # 計算分類損失
            loss_yaw = criterion(yaw, label_yaw)
            loss_pitch = criterion(pitch, label_pitch)
            loss_roll = criterion(roll, label_roll)

            # 計算回歸損失
            yaw_predicted = torch.sum(softmax(yaw) * idx_tensor, 1) * 3 - 99  # 預測俯仰角度
            pitch_predicted = torch.sum(softmax(pitch) * idx_tensor, 1) * 3 - 99  # 預測偏航角度
            roll_predicted = torch.sum(softmax(roll) * idx_tensor, 1) * 3 - 99  # 預測滾轉角度

            loss_reg_yaw = reg_criterion(yaw_predicted, label_yaw_cont)
            loss_reg_pitch = reg_criterion(pitch_predicted, label_pitch_cont)
            loss_reg_roll = reg_criterion(roll_predicted, label_roll_cont)

            # 計算總損失
            total_loss = loss_yaw + alpha * loss_reg_yaw + loss_pitch + alpha * loss_reg_pitch + loss_roll + alpha * loss_reg_roll

            # 累積各指標的損失
            total_mae_yaw += torch.abs(yaw_predicted - label_yaw_cont).sum().item()
            total_mae_pitch += torch.abs(pitch_predicted - label_pitch_cont).sum().item()
            total_mae_roll += torch.abs(roll_predicted - label_roll_cont).sum().item()

            total_mse_yaw += torch.square(yaw_predicted - label_yaw_cont).sum().item()
            total_mse_pitch += torch.square(pitch_predicted - label_pitch_cont).sum().item()
            total_mse_roll += torch.square(roll_predicted - label_roll_cont).sum().item()

            total_mape_yaw += (torch.abs((yaw_predicted - label_yaw_cont) / (label_yaw_cont + 1e-7)) * 100).sum().item()
            total_mape_pitch += (torch.abs((pitch_predicted - label_pitch_cont) / (label_pitch_cont + 1e-7)) * 100).sum().item()
            total_mape_roll += (torch.abs((roll_predicted - label_roll_cont) / (label_roll_cont + 1e-7)) * 100).sum().item()

            total_msle_yaw += torch.square(torch.log1p(torch.abs(yaw_predicted)) - torch.log1p(torch.abs(label_yaw_cont))).sum().item()
            total_msle_pitch += torch.square(torch.log1p(torch.abs(pitch_predicted)) - torch.log1p(torch.abs(label_pitch_cont))).sum().item()
            total_msle_roll += torch.square(torch.log1p(torch.abs(roll_predicted)) - torch.log1p(torch.abs(label_roll_cont))).sum().item()

            correct_predictions += (torch.abs(yaw_predicted - label_yaw_cont) < 5).sum().item()  # 判斷預測是否準確（例如：預測誤差小於5度）
            total_samples += label_yaw.size(0)

            # 執行反向傳播
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # 每1000個batch打印一次損失
            if (i + 1) % 1000 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Iter [{i+1}/{len(train_loader)}] '
                    f'Losses: Yaw {loss_yaw.item():.4f}, Pitch {loss_pitch.item():.4f}, Roll {loss_roll.item():.4f}')

        
        # 計算整體指標
        mae = (total_mae_yaw + total_mae_pitch + total_mae_roll) / (3 * total_samples)
        mse = (total_mse_yaw + total_mse_pitch + total_mse_roll) / (3 * total_samples)
        mape = (total_mape_yaw + total_mape_pitch + total_mape_roll) / (3 * total_samples)
        msle = (total_msle_yaw + total_msle_pitch + total_msle_roll) / (3 * total_samples)

        # 計算每個角度的指標 (yaw, pitch, roll)
        avg_mae_yaw = total_mae_yaw / total_samples
        avg_mae_pitch = total_mae_pitch / total_samples
        avg_mae_roll = total_mae_roll / total_samples

        avg_mse_yaw = total_mse_yaw / total_samples
        avg_mse_pitch = total_mse_pitch / total_samples
        avg_mse_roll = total_mse_roll / total_samples

        avg_mape_yaw = total_mape_yaw / total_samples
        avg_mape_pitch = total_mape_pitch / total_samples
        avg_mape_roll = total_mape_roll / total_samples

        avg_msle_yaw = total_msle_yaw / total_samples
        avg_msle_pitch = total_msle_pitch / total_samples
        avg_msle_roll = total_msle_roll / total_samples

        # 儲存每個指標的結果，供之後可視化使用
        train_metrics['MAE_Yaw'].append(avg_mae_yaw)
        train_metrics['MAE_Pitch'].append(avg_mae_pitch)
        train_metrics['MAE_Roll'].append(avg_mae_roll)

        train_metrics['MSE_Yaw'].append(avg_mse_yaw)
        train_metrics['MSE_Pitch'].append(avg_mse_pitch)
        train_metrics['MSE_Roll'].append(avg_mse_roll)

        train_metrics['MAPE_Yaw'].append(avg_mape_yaw)
        train_metrics['MAPE_Pitch'].append(avg_mape_pitch)
        train_metrics['MAPE_Roll'].append(avg_mape_roll)

        train_metrics['MSLE_Yaw'].append(avg_msle_yaw)
        train_metrics['MSLE_Pitch'].append(avg_msle_pitch)
        train_metrics['MSLE_Roll'].append(avg_msle_roll)


        # 將每個 epoch 的指標結果寫入 CSV 檔案
        with open(metrics_files['MAE'], mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch + 1, avg_mae_yaw, avg_mae_pitch, avg_mae_roll, mae])

        with open(metrics_files['MSE'], mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch + 1, avg_mse_yaw, avg_mse_pitch, avg_mse_roll, mse])

        with open(metrics_files['MAPE'], mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch + 1, avg_mape_yaw, avg_mape_pitch, avg_mape_roll, mape])

        with open(metrics_files['MSLE'], mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch + 1, avg_msle_yaw, avg_msle_pitch, avg_msle_roll, msle])

        # 顯示每個 epoch 的指標結果
        print(f'Epoch [{epoch+1}/{num_epochs}] - MAE: Yaw {avg_mae_yaw:.4f}, Pitch {avg_mae_pitch:.4f}, Roll {avg_mae_roll:.4f}')

        # 儲存每個 epoch 的模型快照
        if epoch % 1 == 0 and epoch < num_epochs:
            print('儲存模型快照...')
            torch.save(model.state_dict(), f'output/snapshots/{output_string}_epoch_{epoch+1}.pkl')


    # 定義指標的名稱
    # 將不同類型的指標分組，便於繪製
    metrics_grouped = {
        'MAE': ['MAE_Yaw', 'MAE_Pitch', 'MAE_Roll'],
        'MSE': ['MSE_Yaw', 'MSE_Pitch', 'MSE_Roll'],
        'MAPE': ['MAPE_Yaw', 'MAPE_Pitch', 'MAPE_Roll'],
        'MSLE': ['MSLE_Yaw', 'MSLE_Pitch', 'MSLE_Roll'],
    }

    # 每種指標的標題
    titles = {
        'MAE': '平均絕對誤差 (MAE)',
        'MSE': '均方誤差 (MSE)',
        'MAPE': '平均絕對百分比誤差 (MAPE)',
        'MSLE': '均方對數誤差 (MSLE)',
    }

    # 定義每個指標顏色
    colors = {
        'MAE_Yaw': 'green',
        'MAE_Pitch': 'blue',
        'MAE_Roll': 'orange',
        'MSE_Yaw': 'green',
        'MSE_Pitch': 'blue',
        'MSE_Roll': 'orange',
        'MAPE_Yaw': 'green',
        'MAPE_Pitch': 'blue',
        'MAPE_Roll': 'orange',
        'MSLE_Yaw': 'green',
        'MSLE_Pitch': 'blue',
        'MSLE_Roll': 'orange',
    }

    # 繪製每種指標的圖形
    for metric_type, metric_names in metrics_grouped.items():
        plt.figure(figsize=(10, 6))
        for metric in metric_names:
            if metric in train_metrics:  # 確保指標存在於 train_metrics 中
                plt.plot(range(1, len(train_metrics[metric]) + 1), train_metrics[metric],
                        label=metric.replace('_', ' '), color=colors[metric])
        
        plt.xlabel('Epochs')
        plt.ylabel(titles[metric_type])
        plt.title(f'{titles[metric_type]} 隨 Epochs 變化')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # 儲存指標圖形
        plt.savefig(f'output/plots/{metric_type}_metrics.png')
        plt.close()

    print("指標已成功儲存。")