import torch
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import librosa
import numpy as np

class Wav2VecFeatureExtractor:
    def __init__(self, model_name="facebook/wav2vec2-base-960h"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2Model.from_pretrained(model_name).to(self.device)
        self.model.eval()
    
    def extract_features(self, audio_path: str, sr: int = 16000) -> np.ndarray:
        """
        提取Wav2Vec2特征
        
        Args:
            audio_path: 音频文件路径
            sr: 采样率
            
        Returns:
            (seq_len, 768)形状的特征数组
        """
        # 加载音频
        waveform, _ = librosa.load(audio_path, sr=sr)
        
        # 处理音频
        inputs = self.processor(
            waveform, 
            sampling_rate=sr, 
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        # 提取特征
        with torch.no_grad():
            outputs = self.model(**inputs)
            features = outputs.last_hidden_state.squeeze(0).cpu().numpy()
        
        return features
    
    def extract_mean_features(self, audio_path: str, sr: int = 16000) -> np.ndarray:
        """
        提取平均后的Wav2Vec2特征
        
        Args:
            audio_path: 音频文件路径
            sr: 采样率
            
        Returns:
            (768,)形状的特征向量
        """
        features = self.extract_features(audio_path, sr)
        return np.mean(features, axis=0)