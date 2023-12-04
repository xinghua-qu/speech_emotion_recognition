import torch
import torch.nn as nn
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

class SpeechEmotionModel(nn.Module):
    def __init__(self, num_classes, num_transformer_layers=0, d_model=512, nhead=8, whisper_model_name="openai/whisper-large-v3"):
        super(SpeechEmotionModel, self).__init__()
        self.num_classes = num_classes
        self.num_transformer_layers = num_transformer_layers

        # 设置设备和数据类型
        self.device = "cuda:7" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        # 加载 Whisper v3 Encoder 和处理器
        self.whisper_encoder = AutoModelForSpeechSeq2Seq.from_pretrained(
            whisper_model_name, torch_dtype=self.torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        ).to(self.device)
        self.processor = AutoProcessor.from_pretrained(whisper_model_name)

        whisper_d_model = self.whisper_encoder.config.hidden_size

        # 可选的维度匹配层
        self.dim_match_layer = nn.Linear(whisper_d_model, d_model) if whisper_d_model != d_model else None

        # Transformer 层
        if num_transformer_layers > 0:
            transformer_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
            self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=num_transformer_layers)
        else:
            self.transformer = None

        # 输出层
        self.output_layer = nn.Linear(d_model, num_classes)

    def forward(self, input_audio):
        # Assuming input_audio is shaped [batch_size, 1, audio_length]
        # Squeeze to remove the channel dimension
        if input_audio.ndim == 3:
            input_audio = input_audio.squeeze(1)
        # 处理输入音频
        processed_input = self.processor(input_audio, return_tensors="pt", sampling_rate=16000).input_values.to(self.device)
        print(processed_input.shape)
        exit(0)
        # 从 Whisper Encoder 获取特征表示
        whisper_output = self.whisper_encoder(processed_input).last_hidden_state

        # 维度匹配（如有必要）
        if self.dim_match_layer:
            whisper_output = self.dim_match_layer(whisper_output)

        # 通过 Transformer 层处理（如有）
        if self.transformer:
            whisper_output = self.transformer(whisper_output)

        # 特征聚合（平均池化）
        pooled_features = whisper_output.mean(dim=1)

        # 映射到情感类别
        output = self.output_layer(pooled_features)
        return output


if __name__=="__main__":
    model = SpeechEmotionModel(num_classes=8, num_transformer_layers=1, d_model=512, nhead=8, whisper_model_name="openai/whisper-large-v3")
    print(model)
    input_audio = torch.randn(4, 1, 16000)
    output = model(input_audio)
    print(output.shape)