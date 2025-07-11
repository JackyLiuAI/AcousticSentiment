import torch
from torch.utils.data import Dataset
import numpy as np
import librosa
from typing import Dict, Tuple


class IEMOCAPDataset(Dataset):
    """PyTorch数据集类，用于加载处理后的IEMOCAP数据"""
    
    def __init__(
        self, 
        metadata: Dict,
        mode: str = "train",  # "train", "valid" or "test"
        feature_type: str = "mfcc",
        max_length: int = 400,
        sr: int = 16000,
        n_mfcc: int = 40,
        augment: bool = False
    ):
        """
        初始化数据集
        
        Args:
            metadata: 元数据字典
            mode: 数据集模式 ("train", "valid" or "test")
            feature_type: 特征类型 ("mfcc", "melspectrogram", "spectrogram")
            max_length: 最大帧数
            sr: 采样率
            n_mfcc: MFCC系数数量
            augment: 是否使用数据增强
        """
        self.metadata = metadata
        self.mode = mode
        self.feature_type = feature_type
        self.max_length = max_length
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.augment = augment
        
        # 情感标签映射
        self.emotion_map = {
            'neutral': 0,
            'happy': 1,
            'sad': 2,
            'angry': 3,
            'excited': 4,
            'frustrated': 5
        }
        
        # 过滤数据
        self.samples = [
            (data['audio_path'], data['text'], data['label'])
            for id_, data in metadata.items()
            if (mode == "train" and "train" in data['mode']) or
               (mode == "valid" and "valid" in data['mode']) or
               (mode == "test" and data['mode'] == "test")
        ]
        
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        audio_path, text, label = self.samples[idx]
        
        # 加载音频
        try:
            # 提取音频特征
            features = self._extract_features(audio_path)
            
            # 数据增强
            if self.augment and self.mode == "train":
                features = self._augment_features(features)
                
            # 转换为张量
            features = torch.FloatTensor(features)
            
            # 标签转换为索引
            label_idx = self.emotion_map[label]
            
            return features, label_idx, text
            
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            # 返回空数据
            return torch.zeros(1, self.max_length), 0, ""
    
    def _extract_features(self, audio_path: str) -> np.ndarray:
        """
        从音频文件中提取特征
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            特征数组
        """
        # 加载音频
        y, sr = librosa.load(audio_path, sr=self.sr)
        
        # 提取特征
        if self.feature_type == "mfcc":
            features = librosa.feature.mfcc(
                y=y, sr=sr, n_mfcc=self.n_mfcc
            )
        elif self.feature_type == "melspectrogram":
            features = librosa.feature.melspectrogram(
                y=y, sr=sr, n_mels=self.n_mfcc
            )
            features = librosa.power_to_db(features)
        else:  # spectrogram
            stft = librosa.stft(y)
            features = librosa.amplitude_to_db(abs(stft))
        
        # 标准化长度
        if features.shape[1] > self.max_length:
            features = features[:, :self.max_length]
        else:
            pad_width = self.max_length - features.shape[1]
            features = np.pad(
                features, 
                pad_width=((0, 0), (0, pad_width)), 
                mode='constant'
            )
            
        return features
    
    def _augment_features(self, features: np.ndarray) -> np.ndarray:
        """
        数据增强 - 添加噪声和时间拉伸
        
        Args:
            features: 原始特征
            
        Returns:
            增强后的特征
        """
        # 添加随机噪声
        noise = np.random.normal(0, 0.01, features.shape)
        features = features + noise
        
        # 随机时间扭曲
        if np.random.rand() > 0.5:
            time_warp = np.random.randint(1, 3)
            features = np.roll(features, time_warp, axis=1)
            
        return features
    
    def get_label_distribution(self) -> Dict:
        """获取标签分布统计"""
        dist = {label: 0 for label in self.emotion_map}
        
        for _, _, label in self.samples:
            dist[label] += 1
            
        return dist