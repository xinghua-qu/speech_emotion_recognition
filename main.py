import torch
import torch.nn as nn
import torch.distributed as dist
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data.distributed import DistributedSampler
import wandb

from model import SpeechEmotionClassifier
from data import RAVDESSDataset
from utils import read_config, split_dataset  # Assuming split_dataset is a utility function you have for splitting the dataset
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def setup(rank, world_size):
    """Initializes the distributed backend."""
    os.environ['MASTER_ADDR'] = 'localhost'  # or the IP address of the master node
    os.environ['MASTER_PORT'] = '12355'      # any free port
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    """Clean up distributed training setup."""
    dist.destroy_process_group()


def train_model(model, dataloader, optimizer, criterion, scaler, rank, config, epoch):
    """Train the model for one epoch."""
    model.train()
    for i, (data, target) in enumerate(dataloader):
        data, target = data.to(rank), target.to(rank)

        optimizer.zero_grad()

        with autocast():
            outputs = model(data)
            print(outputs.shape, target.shape)
            loss = criterion(outputs, target)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Change here: Log every 1000 steps instead of every config.log_interval steps
        if i % 1000 == 0:
            wandb.log({"loss": loss.item(), "epoch": epoch, "batch": i})

def validate_model(model, dataloader, criterion, rank, config, epoch):
    """Validate the model for one epoch."""
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for i, (data, target) in enumerate(dataloader):
            data, target = data.to(rank), target.to(rank)
            outputs = model(data)
            loss = criterion(outputs, target)
            total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == target).sum().item()
            total_samples += target.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_samples
    wandb.log({"val_loss": avg_loss, "val_accuracy": accuracy, "epoch": epoch})

def main(rank, world_size, config_file):
    setup(rank, world_size)

    # Login to wandb with your API key
    wandb.login(key="6c2d72a2a160656cfd8ff15575bd8ef2019edacc")  # Replace with your actual API key

    config = read_config(config_file)
    full_dataset = RAVDESSDataset(config.data_path, config.model_name)
    train_dataset, val_dataset, test_dataset = split_dataset(full_dataset, config)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=False, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, sampler=val_sampler)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, sampler=test_sampler)
    
    model = SpeechEmotionClassifier(config.model_name, config.class_num)
    model.to(rank)  # Move model to the correct device
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()

    wandb.init(project="speech-emotion-classifier")

    for epoch in range(config.epochs):
        train_model(model, train_loader, optimizer, criterion, scaler, rank, config, epoch)
        validate_model(model, val_loader, criterion, rank, config, epoch)

    test_loss, test_accuracy = validate_model(model, test_loader, criterion, rank, config, epoch)
    wandb.log({"test_loss": test_loss, "val_accuracy": test_accuracy, "epoch": epoch})
    cleanup()

    wandb.finish()


if __name__ == "__main__":
    world_size = torch.cuda.device_count() 
    config_file = "config/base.yaml"
    torch.multiprocessing.spawn(main, args=(world_size, config_file), nprocs=world_size)
