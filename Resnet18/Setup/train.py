import argparse
import os
import torch
import data_setup, engine, model_builder, utils
import torchvision

from torchvision import transforms

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

# Setup directories
train_dir = args.training_path
test_dir = args.test_path

# Setup target device
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(int(args.threads))

# Create transforms
data_transform = transforms.Compose([
  transforms.Resize((64, 64)),
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

#Set loss and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),
                             lr=LEARNING_RATE)

# Start training with help from engine.py
results = engine.train(model=model,
             train_dataloader=train_dataloader,
             test_dataloader=test_dataloader,
             loss_fn=loss_fn,
             optimizer=optimizer,
             epochs=NUM_EPOCHS,
             device=device)

# Save the model with help from utils.py
utils.save_model(model=model,
                 target_dir=args.output_path,
                 model_name=args.name + ".pth")

#Plot loss curves
try:
    from helper_functions import plot_loss_curves
except:
    print("[INFO] Couldn't find helper_functions.py, downloading...")
    with open("helper_functions.py", "wb") as f:
        import requests
        request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py")
        f.write(request.content)
    from helper_functions import plot_loss_curves

# Plot the loss curves of our model
plot_loss_curves(results)