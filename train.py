import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataset_utils import DataManager, IEMOCAPDataset
from models import LSTMModel, TransformerModel
from features.wav2vec_feature import Wav2VecFeatureExtractor
import numpy as np
from tqdm import tqdm
import yaml
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class Trainer:
    def __init__(self, config_path="configs/default.yaml"):
        # 加载配置
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        # 初始化
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_extractor = Wav2VecFeatureExtractor()
        
        # 准备数据
        self.data_manager = DataManager()
        self.metadata = self.data_manager.prepare_data()
        
        # 创建数据集
        self.train_set, self.valid_set, self.test_set = self.data_manager.create_datasets(
            self.metadata,
            feature_type="wav2vec",
            max_length=self.config["max_length"],
            sr=self.config["sr"]
        )
        
        # 创建模型
        if self.config["model_type"] == "lstm":
            model = LSTMModel(
                input_dim=self.config["input_dim"],
                hidden_dim=self.config["hidden_dim"],
                num_layers=self.config["num_layers"],
                num_classes=6,
                dropout=self.config["dropout"]
            )
        else:
            model = TransformerModel(
                input_dim=self.config["input_dim"],
                num_heads=self.config["num_heads"],
                num_layers=self.config["num_layers"],
                num_classes=6,
                dropout=self.config["dropout"]
            )
        
        # 使用DataParallel包装模型
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs!")
            model = nn.DataParallel(model)
        
        self.model = model.to(self.device)
        
        # 优化器和损失函数
        # 当使用DataParallel时，需要通过.module访问原始模型的方法
        self.optimizer = self.model.module.get_optimizer(
            lr=float(self.config["lr"]),
            weight_decay=float(self.config["weight_decay"])
        )
        self.criterion = nn.CrossEntropyLoss()
        
        # 日志
        self.writer = SummaryWriter(log_dir=os.path.join("logs", self.config["experiment_name"]))
        self.best_valid_acc = 0.0
        
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        train_loader = DataLoader(
            self.train_set,
            batch_size=self.config["batch_size"],
            shuffle=True,
            num_workers=4
        )
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch_idx, (features, labels, _) in enumerate(pbar):
            features = features.to(self.device)
            labels = labels.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            logits = self.model(features)
            loss = self.criterion(logits, labels)
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            # 记录
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({"loss": loss.item()})
        
        # 计算指标
        train_acc = accuracy_score(all_labels, all_preds)
        avg_loss = total_loss / len(train_loader)
        
        # 写入TensorBoard
        self.writer.add_scalar("Loss/train", avg_loss, epoch)
        self.writer.add_scalar("Accuracy/train", train_acc, epoch)
        
        return avg_loss, train_acc
    
    def validate(self, epoch):
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        valid_loader = DataLoader(
            self.valid_set,
            batch_size=self.config["batch_size"],
            shuffle=False,
            num_workers=4
        )
        
        with torch.no_grad():
            for features, labels, _ in valid_loader:
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                logits = self.model(features)
                loss = self.criterion(logits, labels)
                
                total_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # 计算指标
        valid_acc = accuracy_score(all_labels, all_preds)
        avg_loss = total_loss / len(valid_loader)
        
        # 写入TensorBoard
        self.writer.add_scalar("Loss/valid", avg_loss, epoch)
        self.writer.add_scalar("Accuracy/valid", valid_acc, epoch)
        
        # 保存最佳模型
        if valid_acc > self.best_valid_acc:
            self.best_valid_acc = valid_acc
            torch.save(self.model.state_dict(), f"./weights/best_model_{self.config['model_type']}_lr{self.config['lr']}_maxl{self.config['max_length']}.pth")
            print(f"New best model saved with valid acc: {valid_acc:.4f}")
        
        return avg_loss, valid_acc
    
    def train(self):
        for epoch in range(1, self.config["epochs"] + 1):
            train_loss, train_acc = self.train_epoch(epoch)
            valid_loss, valid_acc = self.validate(epoch)
            
            print(f"Epoch {epoch}: "
                  f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
                  f"Valid Loss: {valid_loss:.4f}, Acc: {valid_acc:.4f}")
    
    def evaluate(self, model_path=None):
        if model_path:
            self.model.load_state_dict(torch.load(model_path, weights_only=True))
        
        self.model.eval()
        all_preds = []
        all_labels = []
        
        test_loader = DataLoader(
            self.test_set,
            batch_size=self.config["batch_size"],
            shuffle=False,
            num_workers=4
        )
        
        with torch.no_grad():
            for features, labels, _ in test_loader:
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                logits = self.model(features)
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # 计算准确率
        test_acc = accuracy_score(all_labels, all_preds)
        print(f"Test Accuracy: {test_acc:.4f}")
        
        # 绘制混淆矩阵
        self.plot_confusion_matrix(all_labels, all_preds)
        
        return test_acc
    
    def plot_confusion_matrix(self, y_true, y_pred):
        emotions = ['neutral', 'happy', 'sad', 'angry', 'excited', 'frustrated']
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                    xticklabels=emotions, yticklabels=emotions)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.savefig(f"./assets/confusion_matrix_{self.config['model_type']}.png")
        plt.close()

if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()
    trainer.evaluate(f"./weights/best_model_{trainer.config['model_type']}_lr{trainer.config['lr']}_maxl{trainer.config['max_length']}.pth")