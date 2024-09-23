import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix
import numpy as np
import os
import time
from tqdm import tqdm
from datetime import datetime

"""改动如下
base V11
只要audio cross
取消test 数据集，遵循标准5fold
2 attention layer respectively
epoch 20 早停
先self后cross
"""

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha  # 权重，处理类别不平衡
        self.reduction = reduction

    def forward(self, inputs, targets):
        # 交叉熵损失
        ce_loss = nn.CrossEntropyLoss(weight=self.alpha, reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)  # 计算 pt
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        # 计算 loss 的返回值
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

import torch
import torch.nn as nn

class AttentionFusionModel(nn.Module):
    def __init__(self, audio_dim, video_dim, num_heads, num_classes, dropout=0.65):
        super(AttentionFusionModel, self).__init__()

        # 音频特征处理
        self.audio_fc = nn.Linear(audio_dim, 512)
        self.audio_to_time_steps = nn.Linear(512, 512 * 10)

        # 视频特征处理
        self.video_conv = nn.Sequential(
            nn.Conv3d(in_channels=video_dim, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2))
        )
        self.video_fc = nn.Linear(512 * 4 * 4 * 4, 512)

        # 位置编码
        self.positional_encoding = nn.Parameter(torch.randn(10, 512))  # 假设有10个时间步

        # 自注意力机制 (音频和视频分别都有自注意力)
        self.audio_self_attn = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=512, num_heads=num_heads, dropout=dropout) for _ in range(2)
        ])
        self.video_self_attn = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=512, num_heads=num_heads, dropout=dropout) for _ in range(2)
        ])

        # 交叉注意力机制 (音频到视频)
        self.cross_attn_audio = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=512, num_heads=num_heads, dropout=dropout) for _ in range(2)
        ])

        # 可学习门控参数
        self.gate_audio = nn.Parameter(torch.tensor(0.5))
        self.gate_video = nn.Parameter(torch.tensor(0.5))

        # 层归一化
        self.audio_norm = nn.LayerNorm(512)
        self.video_norm = nn.LayerNorm(512)

        # 分类层
        self.fc = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, audio_features, video_features):
        device = next(self.parameters()).device
        audio_features = audio_features.to(device)
        video_features = video_features.to(device)

        # 处理音频特征
        audio_features = self.audio_fc(audio_features.view(-1, 768))
        audio_features = self.audio_to_time_steps(audio_features)
        audio_features = audio_features.view(-1, 10, 512).permute(1, 0, 2).contiguous()

        # 处理视频特征
        batch_size, num_frames, channels, depth, height, width = video_features.shape
        video_features = self.video_conv(video_features.view(batch_size * num_frames, channels, depth, height, width))
        video_features = self.video_fc(video_features.view(batch_size, num_frames, -1))
        video_features = video_features.permute(1, 0, 2).contiguous()

        # 位置编码
        positional_encoding = self.positional_encoding.unsqueeze(1).expand(-1, audio_features.size(1), -1)
        audio_features = audio_features + positional_encoding
        video_features = video_features + positional_encoding

        # 自注意力机制 (分别处理音频和视频)
        for i in range(2):
            audio_self, _ = self.audio_self_attn[i](audio_features, audio_features, audio_features)
            video_self, _ = self.video_self_attn[i](video_features, video_features, video_features)

            # 残差连接和层归一化
            audio_features = self.audio_norm(audio_self + audio_features)
            video_features = self.video_norm(video_self + video_features)

        # 交叉注意力机制 (只应用于音频，音频从视频中提取信息)
        for i in range(2):
            audio_cross, _ = self.cross_attn_audio[i](audio_features, video_features, video_features)

            # 残差连接和层归一化
            audio_features = self.audio_norm(audio_features + audio_cross)

        # 加权融合音频和视频特征
        audio_final = self.gate_audio * audio_features.mean(dim=0)
        video_final = self.gate_video * video_features.mean(dim=0)

        # 层归一化
        audio_final = self.audio_norm(audio_final)
        video_final = self.video_norm(video_final)

        # 融合并分类
        fused_features = torch.cat([audio_final, video_final], dim=1)
        logits = self.fc(fused_features)
        return logits

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
        if not os.path.exists('attention_model_V17/saved_checkpoints_V17_layer2'):
            os.makedirs('attention_model_V17/saved_checkpoints_V17_layer2')

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = f'attention_model_V17/saved_checkpoints_V17_layer2/checkpoint_fold_{fold}_{timestamp}.pth'

        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model to {model_path}')

        torch.save(model.state_dict(), model_path)
        self.val_loss_min = val_loss

def calculate_metrics(confusion_mat):
    print("Confusion matrix shape:", confusion_mat.shape)
    print("Confusion matrix:")
    print(confusion_mat)

    if confusion_mat.size == 1:
        print("Warning: Only one class present in the confusion matrix")
        return 0, 0, 0, 0, 0

    try:
        tn, fp, fn, tp = confusion_mat.ravel()
    except ValueError as e:
        print(f"Error unpacking confusion matrix: {e}")
        print("Returning zeros for all metrics")
        return 0, 0, 0, 0, 0

    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return accuracy, precision, recall, specificity, f1

def load_fold_data(fold_path):
    print(f"Loading data from {fold_path}")
    start_time = time.time()

    audio_train = np.load(os.path.join(fold_path, 'train_audio.npy'))
    video_train = np.load(os.path.join(fold_path, 'train_video.npy'))
    labels_train = np.load(os.path.join(fold_path, 'train_labels.npy'))

    # 假设 audio_val.npy、video_val.npy 和 labels_val.npy 是验证集
    audio_val = np.load(os.path.join(fold_path, 'test_audio.npy'))
    video_val = np.load(os.path.join(fold_path, 'test_video.npy'))
    labels_val = np.load(os.path.join(fold_path, 'test_labels.npy'))

    # 构建 TensorDataset
    train_dataset = TensorDataset(torch.Tensor(audio_train), torch.Tensor(video_train), torch.LongTensor(labels_train))
    val_dataset = TensorDataset(torch.Tensor(audio_val), torch.Tensor(video_val), torch.LongTensor(labels_val))

    print(f"Data loading completed in {time.time() - start_time:.2f} seconds")
    print(f"Train set size: {len(train_dataset)}, Validation set size: {len(val_dataset)}")

    return train_dataset, val_dataset

audio_dim = 768
video_dim = 2048
num_heads = 8  # 将 num_heads 改为8
num_classes = 2

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cuda')  
device = torch.device('cpu')  

model = AttentionFusionModel(audio_dim, video_dim, num_heads, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-5)


log_file = f'V17_patience20_layer_2_training_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'

def log_message(message):
    print(message)
    with open(log_file, 'a') as f:
        f.write(message + '\n')

# 5fold
num_folds = 5
best_fold = None
best_val_loss = float('inf')
metrics_per_fold = []
confusion_matrices = []
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cuda')  
device = torch.device('cpu')


total_start_time = time.time()
total_pbar = tqdm(total=num_folds, desc="Total Progress")

for fold in range(1, num_folds + 1):
    fold_path = f'/223040263/gwt/workspace/AdBRC/cy/5fold_data_V9/fold_{fold}'
    log_message(f'Processing fold {fold}')
    train_dataset, val_dataset = load_fold_data(fold_path)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=16, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=16, pin_memory=True)

    # 定义带权重的损失函数
    train_labels = [label.item() for _, _, label in train_dataset]
    class_counts = np.bincount(train_labels)
    class_weights = torch.FloatTensor([0.4, 0.6]).to(device)

    criterion = FocalLoss(gamma=2.0, alpha=class_weights).to(device)
    model = AttentionFusionModel(audio_dim, video_dim, num_heads, num_classes).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-5)
    
    early_stopping = EarlyStopping(patience=20, verbose=True)

    for epoch in range(50):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Fold {fold}, Epoch {epoch+1}")

        for audio_batch, video_batch, labels in pbar:
            audio_batch = audio_batch.to(device)
            video_batch = video_batch.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(audio_batch, video_batch)
            loss = criterion(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        # 验证
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for audio_batch, video_batch, labels in val_loader:
                audio_batch = audio_batch.to(device)
                video_batch = video_batch.to(device)
                labels = labels.to(device)
                outputs = model(audio_batch, video_batch)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())

        conf_matrix = confusion_matrix(all_labels, all_preds)
        accuracy, precision, recall, specificity, f1 = calculate_metrics(conf_matrix)

        log_message(f"Fold {fold}, Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}, "
                    f"Val Loss: {val_loss/len(val_loader):.4f}, Accuracy: {accuracy:.4f}, "
                    f"Precision: {precision:.4f}, Recall: {recall:.4f}, "
                    f"Specificity: {specificity:.4f}, F1: {f1:.4f}")

        early_stopping(val_loss / len(val_loader), model, fold)
        if early_stopping.early_stop:
            log_message("Early stopping")
            break

    torch.cuda.empty_cache()

    # 在测试集上评估
    model.eval()
    test_loss = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for audio_batch, video_batch, labels in test_loader:
            outputs = model(audio_batch, video_batch)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    conf_matrix = confusion_matrix(all_labels, all_preds)
    accuracy, precision, recall, specificity, f1 = calculate_metrics(conf_matrix)

    log_message(f"Fold {fold} Test Results:")
    log_message(f"Test Loss: {test_loss/len(test_loader):.4f}, Accuracy: {accuracy:.4f}, "
                f"Precision: {precision:.4f}, Recall: {recall:.4f}, "
                f"Specificity: {specificity:.4f}, F1: {f1:.4f}")

    # 更新最优fold和验证损失
    if early_stopping.val_loss_min < best_val_loss:
        best_val_loss = early_stopping.val_loss_min
        best_fold = fold

    metrics_per_fold.append({
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'f1': f1,
        'val_loss': val_loss / len(val_loader),
        'test_loss': test_loss / len(test_loader)
    })
    confusion_matrices.append(conf_matrix)
    
    total_pbar.update(1)

total_pbar.close()
total_time = time.time() - total_start_time
log_message(f"Total training time: {total_time:.2f} seconds")

# 计算平均指标和标准差
total_conf_matrix = np.sum(confusion_matrices, axis=0)
avg_accuracy, avg_precision, avg_recall, avg_specificity, avg_f1 = calculate_metrics(total_conf_matrix)

std_metrics = {
    'accuracy': np.std([m['accuracy'] for m in metrics_per_fold]),
    'precision': np.std([m['precision'] for m in metrics_per_fold]),
    'recall': np.std([m['recall'] for m in metrics_per_fold]),
    'specificity': np.std([m['specificity'] for m in metrics_per_fold]),
    'f1': np.std([m['f1'] for m in metrics_per_fold]),
    'val_loss': np.std([m['val_loss'] for m in metrics_per_fold]),
    'test_loss': np.std([m['test_loss'] for m in metrics_per_fold])
}

log_message(f"Average metrics across all folds:")
log_message(f"Accuracy: {avg_accuracy:.4f} ± {std_metrics['accuracy']:.4f}")
log_message(f"Precision: {avg_precision:.4f} ± {std_metrics['precision']:.4f}")
log_message(f"Recall: {avg_recall:.4f} ± {std_metrics['recall']:.4f}")
log_message(f"Specificity: {avg_specificity:.4f} ± {std_metrics['specificity']:.4f}")
log_message(f"F1: {avg_f1:.4f} ± {std_metrics['f1']:.4f}")
log_message(f"Validation Loss: {np.mean([m['val_loss'] for m in metrics_per_fold]):.4f} ± {std_metrics['val_loss']:.4f}")
log_message(f"Test Loss: {np.mean([m['test_loss'] for m in metrics_per_fold]):.4f} ± {std_metrics['test_loss']:.4f}")

log_message(f"Best model saved from fold {best_fold} with val_loss = {best_val_loss:.4f}")

log_message("Total Confusion Matrix:")
log_message(str(total_conf_matrix))

log_message("Individual Fold Confusion Matrices:")
for i, cm in enumerate(confusion_matrices):
    log_message(f"Fold {i+1}:")
    log_message(str(cm))