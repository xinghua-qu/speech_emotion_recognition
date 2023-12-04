import torch
import torch.nn as nn

class EmotionRecognitionModel(nn.Module):
    def __init__(self, whisper_model, num_classes):
        super().__init__()
        self.whisper = whisper_model
        self.classifier = nn.Linear(whisper_model.config.hidden_size, num_classes)

    def forward(self, input_values):
        with torch.no_grad():
            features = self.whisper(input_values).last_hidden_state
        logits = self.classifier(features.mean(dim=1))
        return logits