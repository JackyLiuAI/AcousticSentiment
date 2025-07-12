import torch
from models import LSTMModel, TransformerModel
from features.wav2vec_feature import Wav2VecFeatureExtractor
from dataset_utils import DataManager, IEMOCAPDataset
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import yaml

class Evaluator:
    def __init__(self, config_path="configs/default.yaml", model_path=None):
        # 加载配置
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        # 初始化
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 创建模型
        if self.config["model_type"] == "lstm":
            self.model = LSTMModel(
                input_dim=self.config["input_dim"],
                hidden_dim=self.config["hidden_dim"],
                num_layers=self.config["num_layers"],
                num_classes=6,
                dropout=self.config["dropout"]
            ).to(self.device)
        else:
            self.model = TransformerModel(
                input_dim=self.config["input_dim"],
                num_heads=self.config["num_heads"],
                num_layers=self.config["num_layers"],
                num_classes=6,
                dropout=self.config["dropout"]
            ).to(self.device)
        
        # 加载模型权重
        if model_path:
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict, strict=False)
        
        # 准备数据
        self.data_manager = DataManager()
        self.metadata = self.data_manager.prepare_data()
        
        # 创建测试集
        _, _, self.test_set = self.data_manager.create_datasets(
            self.metadata,
            feature_type="wav2vec",
            max_length=self.config["max_length"],
            sr=self.config["sr"]
        )
    
    def evaluate(self):
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
        plt.savefig(f"./assets/{self.config['model_type']}_lr{self.config['lr']}_epo{self.config['epochs']}_ml{self.config['max_length']}_h{self.config['num_heads']}_l{self.config['num_layers']}_bs{self.config['batch_size']}_wd{self.config['weight_decay']}_do{self.config['dropout']}.png")
        plt.close()

if __name__ == "__main__":
    evaluator = Evaluator(model_path=f"weights/transformer_lr1e-5_epo1_ml400_h4_l2_bs32_wd1e-5_do0.3.pth")
    evaluator.evaluate()