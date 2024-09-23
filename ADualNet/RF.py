"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import os
import time
from tqdm import tqdm
from datetime import datetime
from sklearn.preprocessing import StandardScaler

# 特征提取器
class FeatureExtractor(nn.Module):
    def __init__(self, audio_dim, video_dim):
        super(FeatureExtractor, self).__init__()

        self.audio_fc = nn.Linear(audio_dim, 512)
        self.video_conv = nn.Sequential(
            nn.Conv3d(in_channels=video_dim, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2))
        )
        self.video_fc = nn.Linear(512 * 4 * 4 * 4, 512)
        self.audio_to_time_steps = nn.Linear(512, 512 * 10)

    def forward(self, audio_features, video_features):
        # 处理音频特征
        audio_features = audio_features.view(-1, 768)
        audio_features = self.audio_fc(audio_features)
        audio_features = self.audio_to_time_steps(audio_features)
        audio_features = audio_features.view(-1, 10, 512).permute(1, 0, 2).contiguous()

        # 处理视频特征
        batch_size, num_frames, channels, depth, height, width = video_features.shape
        video_features = video_features.view(batch_size * num_frames, channels, depth, height, width)
        video_features = self.video_conv(video_features)
        video_features = video_features.view(batch_size, num_frames, -1)
        video_features = self.video_fc(video_features)
        video_features = video_features.permute(1, 0, 2).contiguous()

        # 融合音频和视频特征
        audio_features = audio_features.mean(dim=0)
        video_features = video_features.mean(dim=0)
        fused_features = torch.cat([audio_features, video_features], dim=1)  # (batch_size, 1024)
        return fused_features

# EarlyStopping 类
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
        if not os.path.exists('RF/saved_checkpoints'):
            os.makedirs('RF/saved_checkpoints')

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = f'RF/saved_checkpoints/checkpoint_fold_{fold}_{timestamp}.pth'

        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model to {model_path}')
        torch.save(model.state_dict(), model_path)
        self.val_loss_min = val_loss

# 计算性能指标
def calculate_metrics(conf_matrix):
    tn, fp, fn, tp = conf_matrix.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return accuracy, precision, recall, specificity, f1

# 加载fold数据
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
log_file = f'RF_training_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'

def log_message(message):
    print(message)
    with open(log_file, 'a') as f:
        f.write(message + '\n')

# 训练和验证函数
def train_model(train_loader, val_loader, feature_extractor, rf, scaler, num_epochs=50, fold=1):
    best_val_loss = float('inf')
    early_stopping = EarlyStopping(patience=10, verbose=True)
    confusion_matrices = []
    fold_metrics = {'train': [], 'val': []}

    for epoch in range(num_epochs):
        feature_extractor.train()
        running_loss = 0.0

        # 提取特征并训练随机森林
        all_train_features = []
        all_train_labels = []

        for audio_batch, video_batch, labels in tqdm(train_loader, desc=f"Fold {fold}, Epoch {epoch+1}/{num_epochs}"):
            audio_batch = audio_batch.to(torch.device('cpu'))
            video_batch = video_batch.to(torch.device('cpu'))
            labels = labels.to(torch.device('cpu'))

            # 提取特征
            with torch.no_grad():
                features = feature_extractor(audio_batch, video_batch).cpu().numpy()
                all_train_features.extend(features)
                all_train_labels.extend(labels.cpu().numpy())

        # 标准化特征
        all_train_features = scaler.fit_transform(all_train_features)

        # 训练随机森林
        rf.fit(all_train_features, all_train_labels)

        # 验证过程
        feature_extractor.eval()
        all_val_features = []
        all_val_labels = []

        with torch.no_grad():
            for audio_batch, video_batch, labels in val_loader:
                audio_batch = audio_batch.to(torch.device('cpu'))
                video_batch = video_batch.to(torch.device('cpu'))
                labels = labels.to(torch.device('cpu'))

                # 提取验证集特征
                features = feature_extractor(audio_batch, video_batch).cpu().numpy()
                all_val_features.extend(features)
                all_val_labels.extend(labels.cpu().numpy())

        # 标准化验证集特征
        all_val_features = scaler.transform(all_val_features)

        # 进行随机森林预测
        val_preds = rf.predict(all_val_features)
        conf_matrix = confusion_matrix(all_val_labels, val_preds)
        confusion_matrices.append(conf_matrix)

        # 记录每个epoch的验证集和训练集指标
        val_accuracy, val_precision, val_recall, val_specificity, val_f1 = calculate_metrics(conf_matrix)
        fold_metrics['val'].append([val_accuracy, val_precision, val_recall, val_specificity, val_f1])

        log_message(f"Fold {fold}, Epoch {epoch+1}, Val Accuracy: {val_accuracy:.4f}, Val Precision: {val_precision:.4f}, "
                    f"Val Recall: {val_recall:.4f}, Val Specificity: {val_specificity:.4f}, Val F1: {val_f1:.4f}")

        val_loss = 1 - val_accuracy  # 简单地将验证损失定义为1-准确率
        early_stopping(val_loss, feature_extractor, fold)
        if early_stopping.early_stop:
            log_message("Early stopping")
            break

    return confusion_matrices, fold_metrics

# 主训练循环
def main():
    num_folds = 5
    num_epochs = 50
    all_conf_matrices = []
    all_fold_metrics = {'train': [], 'val': []}

    for fold in range(1, num_folds + 1):
        fold_path = f'/223040263/gwt/workspace/AdBRC/cy/5fold_data_V9/fold_{fold}'
        log_message(f'Processing fold {fold}')
        train_dataset, val_dataset = load_fold_data(fold_path)
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)

        feature_extractor = FeatureExtractor(audio_dim=768, video_dim=2048).to(torch.device('cpu'))

        # 使用随机森林分类
        rf = RandomForestClassifier(n_estimators=100, class_weight={0: 0.4, 1: 0.6})
        scaler = StandardScaler()

        fold_conf_matrices, fold_metrics = train_model(train_loader, val_loader, feature_extractor, rf, scaler, num_epochs=num_epochs, fold=fold)
        all_conf_matrices.extend(fold_conf_matrices)
        all_fold_metrics['train'].extend(fold_metrics['train'])
        all_fold_metrics['val'].extend(fold_metrics['val'])

    # 打印和记录总的混淆矩阵及统计指标
    total_conf_matrix = np.sum(all_conf_matrices, axis=0)
    avg_accuracy, avg_precision, avg_recall, avg_specificity, avg_f1 = calculate_metrics(total_conf_matrix)

    # 计算标准差
    std_metrics = {
        'accuracy': np.std([m[0] for m in all_fold_metrics['val']]),
        'precision': np.std([m[1] for m in all_fold_metrics['val']]),
        'recall': np.std([m[2] for m in all_fold_metrics['val']]),
        'specificity': np.std([m[3] for m in all_fold_metrics['val']]),
        'f1': np.std([m[4] for m in all_fold_metrics['val']])
    }

    log_message(f"Final Results after {num_folds}-fold Cross-Validation:")
    log_message(f"Average Accuracy: {avg_accuracy:.4f} ± {std_metrics['accuracy']:.4f}")
    log_message(f"Average Precision: {avg_precision:.4f} ± {std_metrics['precision']:.4f}")
    log_message(f"Average Recall: {avg_recall:.4f} ± {std_metrics['recall']:.4f}")
    log_message(f"Average Specificity: {avg_specificity:.4f} ± {std_metrics['specificity']:.4f}")
    log_message(f"Average F1 Score: {avg_f1:.4f} ± {std_metrics['f1']:.4f}")
    log_message(f"Total Confusion Matrix: {total_conf_matrix}")

if __name__ == "__main__":
    main()
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import os
import time
from tqdm import tqdm
from datetime import datetime

# 特征提取器
class FeatureExtractor(nn.Module):
    def __init__(self, audio_dim, video_dim):
        super(FeatureExtractor, self).__init__()

        self.audio_fc = nn.Linear(audio_dim, 512)
        self.video_conv = nn.Sequential(
            nn.Conv3d(in_channels=video_dim, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2))
        )
        self.video_fc = nn.Linear(512 * 4 * 4 * 4, 512)
        self.audio_to_time_steps = nn.Linear(512, 512 * 10)

    def forward(self, audio_features, video_features):
        # 处理音频特征
        audio_features = audio_features.view(-1, 768)
        audio_features = self.audio_fc(audio_features)
        audio_features = self.audio_to_time_steps(audio_features)
        audio_features = audio_features.view(-1, 10, 512).permute(1, 0, 2).contiguous()

        # 处理视频特征
        batch_size, num_frames, channels, depth, height, width = video_features.shape
        video_features = video_features.view(batch_size * num_frames, channels, depth, height, width)
        video_features = self.video_conv(video_features)
        video_features = video_features.view(batch_size, num_frames, -1)
        video_features = self.video_fc(video_features)
        video_features = video_features.permute(1, 0, 2).contiguous()

        # 融合音频和视频特征
        audio_features = audio_features.mean(dim=0)
        video_features = video_features.mean(dim=0)
        fused_features = torch.cat([audio_features, video_features], dim=1)  # (batch_size, 1024)
        return fused_features

# EarlyStopping 类
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
        if not os.path.exists('RF/saved_checkpoints'):
            os.makedirs('RF/saved_checkpoints')

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = f'RF/saved_checkpoints/checkpoint_fold_{fold}_{timestamp}.pth'

        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model to {model_path}')
        #torch.save(model.state_dict(), model_path)
        self.val_loss_min = val_loss

# 计算性能指标
def calculate_metrics(conf_matrix):
    tn, fp, fn, tp = conf_matrix.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return accuracy, precision, recall, specificity, f1

# 修改的加载fold数据函数
def load_fold_data(fold_path):
    print(f"Loading data from {fold_path}")
    start_time = time.time()

    audio_train = np.load(os.path.join(fold_path, 'train_audio.npy'))
    video_train = np.load(os.path.join(fold_path, 'train_video.npy'))
    labels_train = np.load(os.path.join(fold_path, 'train_labels.npy'))

    # 使用 StratifiedShuffleSplit 进行分层划分，保持训练集和验证集中类别比例一致
    stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=0.1)
    for train_idx, val_idx in stratified_split.split(audio_train, labels_train):
        audio_train_split, audio_val_split = audio_train[train_idx], audio_train[val_idx]
        video_train_split, video_val_split = video_train[train_idx], video_train[val_idx]
        labels_train_split, labels_val_split = labels_train[train_idx], labels_train[val_idx]

    audio_test = np.load(os.path.join(fold_path, 'test_audio.npy'))
    video_test = np.load(os.path.join(fold_path, 'test_video.npy'))
    labels_test = np.load(os.path.join(fold_path, 'test_labels.npy'))

    # 构建 TensorDataset
    train_dataset = TensorDataset(torch.Tensor(audio_train_split), torch.Tensor(video_train_split), torch.LongTensor(labels_train_split))
    val_dataset = TensorDataset(torch.Tensor(audio_val_split), torch.Tensor(video_val_split), torch.LongTensor(labels_val_split))
    test_dataset = TensorDataset(torch.Tensor(audio_test), torch.Tensor(video_test), torch.LongTensor(labels_test))

    print(f"Data loading completed in {time.time() - start_time:.2f} seconds")
    print(f"Train set size: {len(train_dataset)}, Validation set size: {len(val_dataset)}, Test set size: {len(test_dataset)}")

    return train_dataset, val_dataset, test_dataset

# 日志记录
log_file = f'RF_test_training_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'

def log_message(message):
    print(message)
    with open(log_file, 'a') as f:
        f.write(message + '\n')

# 训练和验证函数
def train_model(train_loader, val_loader, feature_extractor, rf, scaler, num_epochs=50, fold=1):
    best_val_loss = float('inf')
    early_stopping = EarlyStopping(patience=10, verbose=True)
    confusion_matrices = []
    fold_metrics = {'train': [], 'val': []}

    for epoch in range(num_epochs):
        feature_extractor.train()

        # 提取特征并训练随机森林
        all_train_features = []
        all_train_labels = []

        for audio_batch, video_batch, labels in tqdm(train_loader, desc=f"Fold {fold}, Epoch {epoch+1}/{num_epochs}"):
            audio_batch = audio_batch.to(torch.device('cpu'))
            video_batch = video_batch.to(torch.device('cpu'))
            labels = labels.to(torch.device('cpu'))

            # 提取特征
            with torch.no_grad():
                features = feature_extractor(audio_batch, video_batch).cpu().numpy()
                all_train_features.extend(features)
                all_train_labels.extend(labels.cpu().numpy())

        # 标准化特征
        all_train_features = scaler.fit_transform(all_train_features)

        # 训练随机森林
        rf.fit(all_train_features, all_train_labels)

        # 验证过程
        feature_extractor.eval()
        all_val_features = []
        all_val_labels = []

        with torch.no_grad():
            for audio_batch, video_batch, labels in val_loader:
                audio_batch = audio_batch.to(torch.device('cpu'))
                video_batch = video_batch.to(torch.device('cpu'))
                labels = labels.to(torch.device('cpu'))

                # 提取验证集特征
                features = feature_extractor(audio_batch, video_batch).cpu().numpy()
                all_val_features.extend(features)
                all_val_labels.extend(labels.cpu().numpy())

        # 标准化验证集特征
        all_val_features = scaler.transform(all_val_features)

        # 进行随机森林预测
        val_preds = rf.predict(all_val_features)
        conf_matrix = confusion_matrix(all_val_labels, val_preds)
        confusion_matrices.append(conf_matrix)

        # 记录每个epoch的验证集和训练集指标
        val_accuracy, val_precision, val_recall, val_specificity, val_f1 = calculate_metrics(conf_matrix)
        log_message(f"Fold {fold}, Epoch {epoch+1}, Validation Accuracy: {val_accuracy:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, Specificity: {val_specificity:.4f}, F1: {val_f1:.4f}")

        # Early Stopping
        val_loss = 1 - val_accuracy  # 使用精度衡量loss
        early_stopping(val_loss, rf, fold)
        if early_stopping.early_stop:
            log_message("Early stopping")
            break

    return confusion_matrices, fold_metrics

# 主训练循环
def main():
    num_folds = 5
    num_epochs = 50
    learning_rate = 1e-4
    weight_decay = 1e-4
    all_conf_matrices = []
    all_fold_metrics = {'train': [], 'val': []}

    for fold in range(1, num_folds + 1):
        fold_path = f'/223040263/gwt/workspace/AdBRC/cy/5fold_data_V9/fold_{fold}'
        log_message(f'Processing fold {fold}')
        train_dataset, val_dataset, test_dataset = load_fold_data(fold_path)
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)

        feature_extractor = FeatureExtractor(audio_dim=768, video_dim=2048).to(torch.device('cpu'))

        # 使用随机森林分类
        rf = RandomForestClassifier(n_estimators=100, class_weight={0: 0.4, 1: 0.6})
        scaler = StandardScaler()

        fold_conf_matrices, fold_metrics = train_model(train_loader, val_loader, feature_extractor, rf, scaler, num_epochs=num_epochs, fold=fold)
        all_conf_matrices.extend(fold_conf_matrices)
        all_fold_metrics['train'].extend(fold_metrics['train'])
        all_fold_metrics['val'].extend(fold_metrics['val'])

    # 打印和记录总的混淆矩阵及统计指标
    total_conf_matrix = np.sum(all_conf_matrices, axis=0)
    avg_accuracy, avg_precision, avg_recall, avg_specificity, avg_f1 = calculate_metrics(total_conf_matrix)

    # 计算标准差
    std_metrics = {
        'accuracy': np.std([m[0] for m in all_fold_metrics['val']]),
        'precision': np.std([m[1] for m in all_fold_metrics['val']]),
        'recall': np.std([m[2] for m in all_fold_metrics['val']]),
        'specificity': np.std([m[3] for m in all_fold_metrics['val']]),
        'f1': np.std([m[4] for m in all_fold_metrics['val']])
    }

    log_message(f"Final Results after {num_folds}-fold Cross-Validation:")
    log_message(f"Average Accuracy: {avg_accuracy:.4f} ± {std_metrics['accuracy']:.4f}")
    log_message(f"Average Precision: {avg_precision:.4f} ± {std_metrics['precision']:.4f}")
    log_message(f"Average Recall: {avg_recall:.4f} ± {std_metrics['recall']:.4f}")
    log_message(f"Average Specificity: {avg_specificity:.4f} ± {std_metrics['specificity']:.4f}")
    log_message(f"Average F1 Score: {avg_f1:.4f} ± {std_metrics['f1']:.4f}")
    log_message(f"Total Confusion Matrix: {total_conf_matrix}")

if __name__ == "__main__":
    main()
