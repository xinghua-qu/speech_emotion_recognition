import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from data import RAVDESSDataset



def main():
    processor = WhisperProcessor.from_pretrained("openai/whisper-large")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large")
    
    
if __name__ == "__main__":
    main()