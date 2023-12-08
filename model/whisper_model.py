import torch
from torch import nn
from transformers import WhisperModel, AutoFeatureExtractor
from torch.nn.functional import softmax

class SpeechEmotionClassifier(nn.Module):
    """Speech Emotion Classification model using Whisper encoder and Transformer layers."""

    def __init__(self, model_name='openai/whisper-large-v3', class_num=8):
        super(SpeechEmotionClassifier, self).__init__()
        self.whisper_model = self.load_pretrained_whisper(model_name)
        self.whisper_encoder = self.whisper_model.encoder  # Use the encoder part of the Whisper model
        self.dimention = self.whisper_encoder.config.d_model
        self.nhead = 8
        self.num_layers = 2
        self.class_num = class_num 
        self.set_trainable(trainable=True)  # Train the Whisper encoder

        # Define additional transformer layers
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=self.dimention,  
            nhead=self.nhead, 
            batch_first=True  # Set batch_first to True
            )
        self.additional_transformer_layers = nn.TransformerEncoder(transformer_layer, num_layers=self.num_layers)

        # Linear projection layer for classification (8 classes)
        self.classifier = nn.Linear(self.dimention, self.class_num)
    
    def load_pretrained_whisper(self, model_name):
        """Load the pre-trained Whisper model."""
        model = WhisperModel.from_pretrained(model_name)
        return model
    
    def set_trainable(self, trainable=True):
        """Set whether the model is trainable."""
        for param in self.whisper_encoder.parameters():
            param.requires_grad = trainable


    def forward(self, input_features):
        """Forward pass for the model."""
        # Pass input through the Whisper encoder
        with torch.no_grad():
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

    input_aduio = torch.rand(4, 128, 3000)  # Replace with actual input features

    # Forward pass
    logits = model(input_aduio)
    print(logits)

if __name__ == "__main__":
    main()
