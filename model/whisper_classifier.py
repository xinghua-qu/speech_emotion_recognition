import torch
import torch.nn as nn
from transformers import AutoModel

class SpeechEmotionModel(nn.Module):
    def __init__(self, num_classes, num_transformer_layers=0, d_model=512, nhead=8):
        super(SpeechEmotionModel, self).__init__()
        self.num_classes = num_classes
        self.num_transformer_layers = num_transformer_layers

        # Load Whisper v3 Encoder using Hugging Face
        self.whisper_encoder = AutoModel.from_pretrained("openai/whisper-base-v3")

        # Transformer Layers (if num_transformer_layers > 0)
        if num_transformer_layers > 0:
            transformer_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
            self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=num_transformer_layers)
        else:
            self.transformer = None

        # Output Layer
        self.output_layer = nn.Linear(d_model, num_classes)

    def forward(self, input_audio):
        # Get feature representation from Whisper Encoder
        features = self.whisper_encoder(input_audio).last_hidden_state

        # Process through Transformer Layers (if any)
        if self.transformer:
            features = self.transformer(features)

        # Map to emotion classes
        output = self.output_layer(features.mean(dim=1))
        return output
