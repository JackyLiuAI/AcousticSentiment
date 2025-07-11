import os
from pydub import AudioSegment
import pandas as pd
from typing import Dict, List, Tuple
import json

class DatasetProcessor:
    """处理IEMOCAP数据集的核心类"""
    
    def __init__(self, base_path: str = "../Datasets/IEMOCAP"):
        self.base_path = base_path
        self.label_csv = os.path.join(base_path, "iemocap_6way_label.csv")
        self.valid_emotions = {'happy', 'sad', 'neutral', 'angry', 'excited', 'frustrated'}
        
    def process_dataset(self, overwrite: bool = False) -> Dict:
        """
        处理整个数据集，返回元数据
        
        Args:
            overwrite: 是否覆盖已处理的文件
            
        Returns:
            包含所有样本元数据的字典
        """
        metadata = {}
        sessions = [f"Session{i}" for i in range(1, 6)]
        
        # 加载标签信息
        label_df = pd.read_csv(self.label_csv)
        label_dict = dict(zip(label_df['id'], label_df['label']))
        mode_dict = dict(zip(label_df['id'], label_df['mode']))
        
        for session in sessions:
            session_path = os.path.join(self.base_path, session)
            if not os.path.exists(session_path):
                continue
                
            # 处理每个session
            session_metadata = self._process_session(
                session_path, 
                label_dict,
                mode_dict,
                overwrite
            )
            metadata.update(session_metadata)
            
        # 保存元数据
        self._save_metadata(metadata)
        return metadata
    
    def _process_session(
        self, 
        session_path: str,
        label_dict: Dict,
        mode_dict: Dict,
        overwrite: bool
    ) -> Dict:
        """
        处理单个session的数据
        
        Args:
            session_path: session文件夹路径
            label_dict: 标签字典 {id: label}
            mode_dict: 模式字典 {id: mode}
            overwrite: 是否覆盖已存在文件
            
        Returns:
            该session的元数据字典
        """
        metadata = {}
        trans_dir = os.path.join(session_path, "dialog", "transcriptions")
        wav_dir = os.path.join(session_path, "dialog", "wav")
        output_dir = os.path.join(session_path, "processed_audio")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 遍历所有转录文件
        for trans_file in os.listdir(trans_dir):
            if not trans_file.endswith(".txt"):
                continue
                
            audio_file = trans_file.replace(".txt", ".wav")
            audio_path = os.path.join(wav_dir, audio_file)
            
            if not os.path.exists(audio_path):
                continue
                
            # 读取转录文件内容
            with open(os.path.join(trans_dir, trans_file), "r") as f:
                lines = f.readlines()
                
            # 处理每一行
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                try:
                    # 解析行内容
                    id_part, time_part, *text_parts = line.split()
                    text = " ".join(text_parts).replace(":", "").strip()
                    
                    # 提取时间信息
                    time_str = time_part[1:-2]  # 去掉方括号和冒号
                    start_time, end_time = map(float, time_str.split("-"))
                    
                    # 检查是否有标签
                    label = label_dict.get(id_part)
                    if label not in self.valid_emotions:
                        continue
                        
                    # 构建输出路径
                    output_path = os.path.join(output_dir, f"{id_part}.wav")
                    
                    # 处理音频文件
                    if not os.path.exists(output_path) or overwrite:
                        self._cut_audio(audio_path, start_time, end_time, output_path)
                    
                    # 添加到元数据
                    metadata[id_part] = {
                        "audio_path": output_path,
                        "text": text,
                        "label": label,
                        "mode": mode_dict.get(id_part, "unknown"),
                        "duration": end_time - start_time
                    }
                    
                except Exception as e:
                    print(f"Error processing line '{line}': {e}")
                    
        return metadata
    
    def _cut_audio(
        self, 
        input_path: str, 
        start: float, 
        end: float, 
        output_path: str
    ) -> None:
        """
        切分音频文件
        
        Args:
            input_path: 输入音频路径
            start: 开始时间(秒)
            end: 结束时间(秒)
            output_path: 输出路径
        """
        try:
            audio = AudioSegment.from_wav(input_path)
            segment = audio[start*1000:end*1000]
            segment.export(output_path, format="wav")
        except Exception as e:
            print(f"Error cutting audio {input_path}: {e}")
    
    def _save_metadata(self, metadata: Dict, filename: str = "metadata.json") -> None:
        """
        保存元数据到JSON文件
        
        Args:
            metadata: 元数据字典
            filename: 保存文件名
        """
        save_path = os.path.join(self.base_path, filename)
        with open(save_path, "w") as f:
            json.dump(metadata, f, indent=2)
            
    def load_metadata(self, filename: str = "metadata.json") -> Dict:
        """
        从JSON文件加载元数据
        
        Args:
            filename: 元数据文件名
            
        Returns:
            元数据字典
        """
        load_path = os.path.join(self.base_path, filename)
        if not os.path.exists(load_path):
            return {}
            
        with open(load_path, "r") as f:
            return json.load(f)