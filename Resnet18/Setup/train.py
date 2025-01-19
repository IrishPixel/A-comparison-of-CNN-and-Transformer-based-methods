import argparse
import os
import torch
import data_setup, engine, model_builder, utils
import torchvision
import torch.nn as nn

from torchvision import transforms
from tensorboardX import SummaryWriter

#Setup arguments
def get_arg():
    parser = argparse.ArgumentParser(
        description='The script to train new models on normalized image tiles.')
    parser.add_argument(
        '-m', '--model', help="The model to be trained")
    parser.add_argument(
        '-bs', '--batch_size', help="Batch Size of the model")
    parser.add_argument(
        '-lr', '--learning_rate', help="Learning rage of the model")
    parser.add_argument(
        '-ep', '--num_epochs', help="How many times to train the model")
    parser.add_argument(
            '-cl', '--num_classes', help="Number of classes")
    parser.add_argument(
        '-tr', '--training_path', help="Path to the training data")
    parser.add_argument(
        '-ts', '--test_path', help="Path to the test data")
    parser.add_argument(
        '-op', '--output_path', help="Path to send trained model")
    parser.add_argument(
        '-n', '--name', help="The name of the outputed model (without .pth)")
    parser.add_argument(
        '-t', '--threads', help="Number of threads to be used")
    return parser.parse_args()

# Setup hyperparameters
args = get_arg()
NUM_EPOCHS = int(args.num_epochs)
BATCH_SIZE = int(args.batch_size)
HIDDEN_UNITS = 10
LEARNING_RATE = float(args.learning_rate)
num_classes = int(args.num_classes)

# Setup directories
train_dir = args.training_path
test_dir = args.test_path

# Setup target device
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(int(args.threads))

# Create transforms
data_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ToTensor()
])

# Create DataLoaders with help from data_setup.py
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir=train_dir,
    test_dir=test_dir,
    transform=data_transform,
    batch_size=BATCH_SIZE
)

# Create model with help from model_builder.py
model = torchvision.models.resnet18().to(device)

#Set up fully connected layer
model.fc == nn.Sequential(
    nn.Dropout(p=0.5),
    nn.Linear(model.fc.in_features, num_classes)
).to(device)

#Set loss and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),
                             lr=LEARNING_RATE,
                            weight_decay=1e-4)

#Set lr scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# Start training with help from engine.py
results = engine.train(model=model,
             train_dataloader=train_dataloader,
             test_dataloader=test_dataloader,
             loss_fn=loss_fn,
             optimizer=optimizer,
             scheduler=scheduler,
             epochs=NUM_EPOCHS,
             device=device,
             num_classes=num_classes)

# Save the model with help from utils.py
utils.save_model(model=model,
                 target_dir=args.output_path,
                 model_name=args.name + ".pth")

#Log output
writer = SummaryWriter(log_dir=args.output_path)
f_log = open(args.output_path + f"{args.name}.log", 'w')

for epoch in range(NUM_EPOCHS):
    train_acc = results['train_acc'][epoch]
    test_acc = results['test_acc'][epoch]
    train_loss = results['train_loss'][epoch]
    test_loss = results['test_loss'][epoch]

    if hasattr(train_acc, 'item'):
        train_acc = train_acc.item()
    if hasattr(test_acc, 'item'):
        test_acc = test_acc.item()
    if hasattr(train_loss, 'item'):
        train_loss = train_loss.item()
    if hasattr(test_loss, 'item'):
        test_loss = test_loss.item()

    writer.add_scalar('Accuracy/Train', train_acc, epoch)
    writer.add_scalar('Accuracy/Test', test_acc, epoch)
    writer.add_scalar('Loss/Train', train_loss, epoch)
    writer.add_scalar('Loss/Test', test_loss, epoch)
    
    log = (
    'epoch [{}/{}] ------ acc: train = {:.4f}, test = {:.4f}'.format(epoch+1, NUM_EPOCHS, train_acc, test_acc) + "\n"
    'epoch [{}/{}] ------ loss: train = {:.4f}, test = {:.4f}'.format(epoch+1, NUM_EPOCHS, train_loss, test_loss) + "\n"
    '================================\n\n'
    )
    print(log)

    f_log.write(log)
    f_log.flush()

f_log.close()
writer.close()
