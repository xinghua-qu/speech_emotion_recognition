import torch
from transformers import AutoFeatureExtractor, WhisperModel
from datasets import load_dataset

model = WhisperModel.from_pretrained("openai/whisper-base").encoder
feature_extractor = AutoFeatureExtractor.from_pretrained("openai/whisper-base")
ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
print(ds[0]['audio']['array'].shape)    
inputs = feature_extractor(ds[0]["audio"]["array"], return_tensors="pt")
print(inputs.input_features.shape)
input_features = inputs.input_features
last_hidden_state = model(input_features).last_hidden_state
print(last_hidden_state.shape)
# decoder_input_ids = torch.tensor([[1, 1]]) * model.config.decoder_start_token_id
# last_hidden_state = model(input_features, decoder_input_ids=decoder_input_ids).last_hidden_state
# list(last_hidden_state.shape)