from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from .DatasetProcessor import DatasetProcessor
from typing import Dict, List, Tuple
from .IEMOCAPDataset import IEMOCAPDataset
from torch.utils.data import Dataset

class DataManager:
    """管理数据集加载和预处理"""
    
    def __init__(self, base_path: str = "../Datasets/IEMOCAP"):
        self.base_path = base_path
        self.processor = DatasetProcessor(base_path)
        
    def prepare_data(self, overwrite: bool = False) -> Dict:
        """
        准备数据集
        
        Args:
            overwrite: 是否覆盖已处理的数据
            
        Returns:
            元数据字典
        """
        # 检查是否已有处理好的元数据
        metadata = self.processor.load_metadata()
        
        if not metadata or overwrite:
            print("Processing dataset...")
            metadata = self.processor.process_dataset(overwrite)
            print(f"Processed {len(metadata)} samples")
        else:
            print(f"Loaded {len(metadata)} existing samples")
            
        return metadata
    
    def create_datasets(
        self,
        metadata: Dict,
        feature_type: str = "mfcc",
        max_length: int = 400,
        sr: int = 16000,
        n_mfcc: int = 40,
        augment_train: bool = True,
        valid_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[Dataset, Dataset, Dataset]:
        """
        创建训练、验证和测试集
        
        Args:
            metadata: 元数据字典
            feature_type: 特征类型
            max_length: 最大帧数
            sr: 采样率
            n_mfcc: MFCC系数数量
            augment_train: 是否增强训练数据
            valid_size: 验证集比例
            random_state: 随机种子
            
        Returns:
            (train_set, valid_set, test_set)
        """
        # 划分训练和验证集
        train_meta = {
            k: v for k, v in metadata.items() 
            if "train" in v["mode"] or "valid" in v["mode"]
        }
        
        test_meta = {
            k: v for k, v in metadata.items()
            if v["mode"] == "test"
        }
        
        # 进一步划分训练和验证集
        train_ids, valid_ids = train_test_split(
            list(train_meta.keys()),
            test_size=valid_size,
            random_state=random_state,
            stratify=[v["label"] for v in train_meta.values()]
        )
        
        train_meta_final = {k: train_meta[k] for k in train_ids}
        valid_meta_final = {k: train_meta[k] for k in valid_ids}
        
        # 创建数据集
        train_set = IEMOCAPDataset(
            train_meta_final,
            mode="train",
            feature_type=feature_type,
            max_length=max_length,
            sr=sr,
            n_mfcc=n_mfcc,
            augment=augment_train
        )
        
        valid_set = IEMOCAPDataset(
            valid_meta_final,
            mode="valid",
            feature_type=feature_type,
            max_length=max_length,
            sr=sr,
            n_mfcc=n_mfcc,
            augment=False
        )
        
        test_set = IEMOCAPDataset(
            test_meta,
            mode="test",
            feature_type=feature_type,
            max_length=max_length,
            sr=sr,
            n_mfcc=n_mfcc,
            augment=False
        )
        
        return train_set, valid_set, test_set
    
    def create_dataloaders(
        self,
        train_set: Dataset,
        valid_set: Dataset,
        test_set: Dataset,
        batch_size: int = 32,
        num_workers: int = 4,
        shuffle_train: bool = True
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        创建数据加载器
        
        Args:
            train_set: 训练集
            valid_set: 验证集
            test_set: 测试集
            batch_size: 批大小
            num_workers: 工作线程数
            shuffle_train: 是否打乱训练集
            
        Returns:
            (train_loader, valid_loader, test_loader)
        """
        train_loader = DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=shuffle_train,
            num_workers=num_workers,
            pin_memory=True
        )
        
        valid_loader = DataLoader(
            valid_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        return train_loader, valid_loader, test_loader