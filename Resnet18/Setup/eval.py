import argparse
import torch
import data_setup, engine, model_builder, utils
import torchvision
from torchvision import transforms
import logging
from sklearn.metrics import precision_recall_fscore_support

# Setup arguments
def get_arg():
    parser = argparse.ArgumentParser(
        description='Evaluate a trained model on a test dataset.')
    parser.add_argument(
        '-m', '--model', help="Path to the trained model file (.pth)")
    parser.add_argument(
        '-ts', '--test_path', help="Path to the test data")
    parser.add_argument(
        '-bs', '--batch_size', help="Batch Size of the model")
    parser.add_argument(
        '-t', '--threads', help="Number of threads to be used")
    parser.add_argument(
        '-l', '--log_path', help="Path to save the log file")
    return parser.parse_args()

# Setup arguments
args = get_arg()

# Setup hyperparameters
BATCH_SIZE = int(args.batch_size)
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(int(args.threads))

# Set up logging
logging.basicConfig(
    filename=args.log_path,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Create transforms
data_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# Create DataLoader for test data
_, test_dataloader, class_names = data_setup.create_dataloaders(
    test_dir=args.test_path,
    transform=data_transform,
    batch_size=BATCH_SIZE
)

# Load the trained model
model = torchvision.models.resnet18().to(device)  # Same architecture as during training
model.load_state_dict(torch.load(args.model))  # Load the model weights from the .pth file

# Set the model to evaluation mode
model.eval()

# Define the loss function (same as in training)
loss_fn = torch.nn.CrossEntropyLoss()

# Evaluate the model using engine.py
test_loss, test_acc, (per_class_precision, per_class_recall), all_preds, all_labels = engine.evaluation_step(
    model=model,
    dataloader=test_dataloader,
    loss_fn=loss_fn,
    device=device,
    num_classes=len(class_names)  # Modify the test_step to return predictions
)

#Log results
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
print("Per Class Metrics:")
for class_idx, (precision, recall) in enumerate(zip(per_class_precision, per_class_recall)):
    print(f"Class {class_idx}: Precision: {precision:.4f}, Recall: {recall:.4f}")

if args.log_path:
    with open(args.log_path, 'w') as log_file:
        log_file.write(f"Test Loss: {test_loss:.4f}\n")
        log_file.write(f"Test Accuracy: {test_acc:.4f}\n")
        log_file.write("Per_Class_Metrics:\n")
        for class_name, (precision, recall) in enumerate(zip(per_class_precision, per_class_recall)):
            log_file.write(f"Class {class_name}: Precision: {precision:.4f}, Recall: {recall:.4f}\n")
        log_file.write("Predicions and Labels:\n")
        log_file.write(f"Predicions: {all_preds}\n")
        log_file.write(f"Labels: {all_labels}\n")
