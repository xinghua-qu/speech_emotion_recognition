import torch
import os
import torch.nn as nn
from transformers import Wav2Vec2ForSequenceClassification
import wandb
from model import SpeechEmotionClassifier

from data import RAVDESSDataset
from utils import read_config, split_dataset
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

def compute_accuracy(model, dataloader, device):
    """Compute accuracy on the given dataset."""
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for i, (_, data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            loss = criterion(outputs, target)
            total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == target).sum().item()
            total_samples += target.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_samples
    return accuracy, avg_loss

def w2v(config_file):
    wandb.login(key="6c2d72a2a160656cfd8ff15575bd8ef2019edacc")
    wandb.init(project="speech-emotion-wav2vec2")

    config = read_config(config_file)
    full_dataset = RAVDESSDataset(config.data_path, config.model_name)
    _, _, test_dataset = split_dataset(full_dataset, config)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    
    model = Wav2Vec2ForSequenceClassification.from_pretrained("ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition")
    model.to('cuda')

    accuracy = compute_accuracy(model, test_loader)
    wandb.log({"test_accuracy": accuracy})
    wandb.finish()
    
def whisper_infer(config_file):
    wandb.login(key="6c2d72a2a160656cfd8ff15575bd8ef2019edacc")
    wandb.init(project="speech-emotion-whisper-inference")

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    config = read_config(config_file)
    full_dataset = RAVDESSDataset(config.data_path, config.model_name)
    _, _, test_dataset = split_dataset(full_dataset, config)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    model = SpeechEmotionClassifier(config.model_name, config.class_num)
    model_path = f'./results/whisper_emotion_epoch_{config.epochs}_{config.model_name}/shanda.pth'
    state_dict = torch.load(model_path)
    # Remove the "module." prefix from the keys
    new_state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.to(device)

    accuracy, avg_loss = compute_accuracy(model, test_loader, device)
    print(accuracy, avg_loss)
    
    wandb.log({"test_accuracy": accuracy})
    wandb.log({"avg_loss": avg_loss})
    wandb.finish()

if __name__ == "__main__":
    model = 'whisper'
    
    if model == 'whisper':
        config_file = "config/whisper_based.yaml"
        whisper_infer(config_file)
    else:
        config_file = "config/w2v_infer.yaml"
        w2v(config_file)
