import torch
from torch import nn
from transformers import WhisperModel, AutoFeatureExtractor


class SpeechEmotionClassifier(nn.Module):
    """Speech Emotion Classification model using Whisper encoder and Transformer layers."""

    def __init__(self, model_name='openai/whisper-large-v3', class_num=8):
        super(SpeechEmotionClassifier, self).__init__()
        self.whisper_model, self.feature_extractor = self.load_pretrained_whisper(model_name)
        self.whisper_encoder = self.whisper_model.encoder  # Use the encoder part of the Whisper model
        self.dimention = self.whisper_encoder.config.d_model
        self.nhead = self.whisper_encoder.config.nhead
        self.class_num = class_num 

        # Define additional transformer layers
        transformer_layer = nn.TransformerEncoderLayer(d_model=self.dimention, nhead=self.nhead)
        self.additional_transformer_layers = nn.TransformerEncoder(transformer_layer, num_layers=2)

        # Linear projection layer for classification (8 classes)
        self.classifier = nn.Linear(self.dimention, 8)
    
    def load_pretrained_whisper(self, model_name):
        """Load the pre-trained Whisper model."""
        model = WhisperModel.from_pretrained(model_name)
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        return model, feature_extractor
    
    def process_input(self, input_audio):
        """Process the input features to the model."""
        inputs = self.feature_extractor(input_audio, return_tensors="pt")
        input_features = inputs.input_features
        return input_features

    def forward(self, input_audio):
        """Forward pass for the model."""
        print(input_audio.shape)
        input_features = self.process_input(input_audio)
        
        # Pass input through the Whisper encoder
        encoder_output = self.whisper_encoder(input_features).last_hidden_state

        # Pass output through additional Transformer layers
        transformer_output = self.additional_transformer_layers(encoder_output)

        # Classification using the output of the last transformer layer
        logits = self.classifier(transformer_output[:, 0, :])
        return logits

def main():
    """Main function to instantiate and test the model."""
    # Instantiate the model
    model = SpeechEmotionClassifier(model_name="openai/whisper-large-v3", class_num=8)

    input_aduio = torch.rand(16000,4)  # Replace with actual input features

    # Forward pass
    logits = model(input_aduio)
    print(logits)

if __name__ == "__main__":
    main()
