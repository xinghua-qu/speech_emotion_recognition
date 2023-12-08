import torch
from transformers import Wav2Vec2ForSequenceClassification
import wandb

from data import RAVDESSDataset
from utils import read_config, split_dataset

def compute_accuracy(model, dataloader):
    """Compute accuracy on the given dataset."""
    model.eval()
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for data, _, target in dataloader:
            data, target = data.to('cuda'), target.to('cuda')
            outputs = model(data).logits
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == target).sum().item()
            total_samples += target.size(0)

    accuracy = total_correct / total_samples
    return accuracy

def main(config_file):
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

if __name__ == "__main__":
    config_file = "config/w2v_infer.yaml"
    main(config_file)
