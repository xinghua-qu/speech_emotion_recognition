"""This module contains the main code for training a Speech Emotion Classifier.

Author: Xinghua QU (quxinghua17@gmail.com)
"""

import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data.distributed import DistributedSampler
import wandb
import argparse
from model import SpeechEmotionClassifier
from data import RAVDESSDataset
from utils import read_config, split_dataset


def setup(rank, world_size):
    """Initializes the distributed backend."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    """Clean up distributed training setup."""
    dist.destroy_process_group()


def train_model(model, dataloader, optimizer, criterion, rank, config, epoch):
    """Train the model for one epoch."""
    model.train()
    total_correct = 0
    total_samples = 0
    for i, (_, data, target) in enumerate(dataloader):
        data, target = data.to(rank), target.to(rank)

        optimizer.zero_grad()
        
        # Compute output and loss
        outputs = model(data)
        loss = criterion(outputs, target)
        
        # Compute accuracy
        _, predicted = torch.max(outputs.data, 1)
        total_correct += (predicted == target).sum().item()
        total_samples += target.size(0)

        # Backpropagation
        loss.backward()
        optimizer.step()
        
        # Logging
        if rank == 0 and i % config.log_interval == 0:
            accuracy = total_correct / total_samples
            print(f"Epoch: {epoch}, steps: {i}, loss: {loss.item()}, accuracy: {accuracy}")
            wandb.log({"loss": loss.item(), "accuracy": accuracy, "epoch": epoch, "batch": i})

def validate_model(model, dataloader, criterion, rank, config, epoch, datatype):
    """Validate the model for one epoch."""
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for i, (_, data, target) in enumerate(dataloader):
            data, target = data.to(rank), target.to(rank)
            outputs = model(data)
            loss = criterion(outputs, target)
            total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == target).sum().item()
            total_samples += target.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_samples
    if rank == 0 and datatype == 'val':
        wandb.log({"val_loss": avg_loss, "val_accuracy": accuracy, "epoch": epoch})
    elif rank == 0 and datatype == 'test':
        wandb.log({"test_loss": avg_loss, "test_accuracy": accuracy, "epoch": epoch})


def main(rank, world_size, config_file, args):
    setup(rank, world_size)
    config = read_config(config_file)
    config.model_name = args.model_name
    config.epochs = args.epoch
    
    # Set the GPU ID if world_size is 1
    if world_size == 1:
        torch.cuda.set_device(args.gpu_id)

    if rank == 0:
        wandb.login(key="6c2d72a2a160656cfd8ff15575bd8ef2019edacc")
        run_name = f"shanda_speech_emotion_{config.model_name}_lr_{config.lr}_epochs_{config.epochs}"
        wandb.init(project="test-shanda_speech-emotion-whisper", name=run_name)

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
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    criterion = nn.CrossEntropyLoss()
    # scaler = GradScaler()

    for epoch in range(config.epochs):
        train_model(model, train_loader, optimizer, criterion, rank, config, epoch)
        validate_model(model, val_loader, criterion, rank, config, epoch, 'val')
        validate_model(model, test_loader, criterion, rank, config, epoch, 'test')
    
    validate_model(model, test_loader, criterion, rank, config, epoch, 'test')
    # Save the trained model
    if rank == 0:
        model_dir = f'./results/whisper_emotion_epoch_{config.epochs}_{config.model_name}'
        os.makedirs(model_dir, exist_ok=True)  # Create directory if it doesn't exist
        torch.save(model.state_dict(), f'{model_dir}/shanda.pth')
        wandb.finish()
    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a Speech Emotion Classifier.')
    parser.add_argument('--model_name', default='openai/whisper-large-v3', type=str, required=True, help='Name of the model to train.')
    parser.add_argument('--epoch', default=60, type=int, help='epoch for training.')
    args = parser.parse_args()
    world_size = torch.cuda.device_count() 
    # world_size = 1
    config_file = "config/whisper_based.yaml"
    torch.multiprocessing.spawn(main, args=(world_size, config_file, args), nprocs=world_size)
