import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import os
import time
from tqdm import tqdm
from datetime import datetime

class FusionModel(nn.Module):
    def __init__(self, audio_dim, video_dim, num_classes, dropout=0.5):
        super(FusionModel, self).__init__()

        # 音频和视频特征的线性转换
        self.audio_fc = nn.Linear(audio_dim, 512)

        self.video_conv = nn.Sequential(
            nn.Conv3d(in_channels=video_dim, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2))
        )
        self.video_fc = nn.Linear(512 * 4 * 4 * 4, 512)  # 修改视频特征展平处理后的线性层

        self.audio_to_time_steps = nn.Linear(512, 512 * 10)

        # 加权平均融合
        self.audio_weight = nn.Parameter(torch.tensor(0.5))  # 可学习的权重
        self.video_weight = nn.Parameter(torch.tensor(0.5))  # 可学习的权重

        # 分类器（MLP）
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, audio_features, video_features):
        # 处理音频特征
        audio_features = audio_features.view(-1, 768)  # 保持音频特征展平后的形状
        audio_features = self.audio_fc(audio_features)
        audio_features = self.audio_to_time_steps(audio_features)
        audio_features = audio_features.view(-1, 10, 512).permute(1, 0, 2).contiguous()  # 转换为时间步长维度

        # 处理视频特征
        batch_size, num_frames, channels, depth, height, width = video_features.shape
        video_features = video_features.view(batch_size * num_frames, channels, depth, height, width)
        video_features = self.video_conv(video_features)
        video_features = video_features.view(batch_size, num_frames, -1)
        video_features = self.video_fc(video_features)
        video_features = video_features.permute(1, 0, 2).contiguous()  # 确保视频特征维度一致

        # 加权平均融合音频和视频特征
        fused_features = self.audio_weight * audio_features.mean(dim=0) + self.video_weight * video_features.mean(dim=0)

        # 分类
        logits = self.fc(fused_features)
        return logits

# 定义 EarlyStopping 类
class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0.001):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model, fold):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, fold)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, fold)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, fold):
        if not os.path.exists('attention_model_V18/saved_checkpoints_V18'):
            os.makedirs('attention_model_V18/saved_checkpoints_V18')

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = f'attention_model_V18/saved_checkpoints_V18/checkpoint_fold_{fold}_{timestamp}.pth'

        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model to {model_path}')
        torch.save(model.state_dict(), model_path)
        self.val_loss_min = val_loss

# 计算性能指标
def calculate_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='binary')
    recall = recall_score(y_true, y_pred, average='binary')
    f1 = f1_score(y_true, y_pred, average='binary')
    return accuracy, precision, recall, f1

# 加载数据
def load_fold_data(fold_path):
    audio_train = np.load(os.path.join(fold_path, 'train_audio.npy'))
    video_train = np.load(os.path.join(fold_path, 'train_video.npy'))
    labels_train = np.load(os.path.join(fold_path, 'train_labels.npy'))

    audio_val = np.load(os.path.join(fold_path, 'test_audio.npy'))
    video_val = np.load(os.path.join(fold_path, 'test_video.npy'))
    labels_val = np.load(os.path.join(fold_path, 'test_labels.npy'))

    train_dataset = TensorDataset(torch.Tensor(audio_train), torch.Tensor(video_train), torch.LongTensor(labels_train))
    val_dataset = TensorDataset(torch.Tensor(audio_val), torch.Tensor(video_val), torch.LongTensor(labels_val))

    return train_dataset, val_dataset

# 日志记录
log_file = f'V18_training_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'

def log_message(message):
    print(message)
    with open(log_file, 'a') as f:
        f.write(message + '\n')

# 训练和验证函数
def train_model(train_loader, val_loader, model, criterion, optimizer, num_epochs=50, fold=1):
    best_val_loss = float('inf')
    early_stopping = EarlyStopping(patience=10, verbose=True)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for audio_batch, video_batch, labels in tqdm(train_loader, desc=f"Fold {fold}, Epoch {epoch+1}/{num_epochs}"):
            audio_batch = audio_batch.to(torch.device('cpu'))
            video_batch = video_batch.to(torch.device('cpu'))
            labels = labels.to(torch.device('cpu'))

            optimizer.zero_grad()
            outputs = model(audio_batch, video_batch)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        log_message(f"Fold {fold}, Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}")

        # 验证过程
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for audio_batch, video_batch, labels in val_loader:
                audio_batch = audio_batch.to(torch.device('cpu'))
                video_batch = video_batch.to(torch.device('cpu'))
                labels = labels.to(torch.device('cpu'))

                outputs = model(audio_batch, video_batch)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())

        val_loss /= len(val_loader)
        accuracy, precision, recall, f1 = calculate_metrics(all_labels, all_preds)
        log_message(f"Fold {fold}, Validation Loss: {val_loss}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

        # Early Stopping
        early_stopping(val_loss, model, fold)
        if early_stopping.early_stop:
            log_message("Early stopping")
            break

# 主训练循环
def main():
    num_folds = 5
    num_epochs = 50
    learning_rate = 1e-4
    weight_decay = 1e-4

    for fold in range(1, num_folds + 1):
        fold_path = f'/223040263/gwt/workspace/AdBRC/cy/5fold_data_V9/fold_{fold}'
        log_message(f'Processing fold {fold}')
        train_dataset, val_dataset = load_fold_data(fold_path)
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)

        model = FusionModel(audio_dim=768, video_dim=2048, num_classes=2).to(torch.device('cpu'))

        # 使用 Weighted Cross-Entropy Loss
        class_weights = torch.FloatTensor([0.4, 0.6]).to(torch.device('cpu'))
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # 训练模型
        train_model(train_loader, val_loader, model, criterion, optimizer, num_epochs=num_epochs, fold=fold)

if __name__ == "__main__":
    main()
