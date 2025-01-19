import torch

from tqdm.auto import tqdm
from typing import Dict, List, Tuple

def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer,
               device: torch.device,
               num_classes: int) -> Tuple[float, float]:

    model.train()

    train_loss, train_acc, train_precision, train_recall = 0,0,0,0
    all_preds, all_labels = [], []

    #Initialize counts for per-class precision
    class_tp = torch.zeros(num_classes, device=device)
    class_fp = torch.zeros(num_classes, device=device)
    class_fn = torch.zeros(num_classes, device=device)

    for batch, (X, y) in enumerate(dataloader):
        # Send data to target device
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate  and accumulate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item() 

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # Calculate and accumulate accuracy metric across all batches
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)

        all_preds.append(y_pred_class.cpu().tolist())
        all_labels.append(y.cpu().tolist())

        for class_idx in range(num_classes):
            class_tp[class_idx] += ((y_pred_class == class_idx) & (y == class_idx)).sum().item()
            class_fp[class_idx] += ((y_pred_class == class_idx) & (y != class_idx)).sum().item()
            class_fn[class_idx] += ((y_pred_class != class_idx) & (y == class_idx)).sum().item()

    train_precision = (class_tp.sum() / (class_tp.sum() + class_fp.sum() + 1e-8)).item()
    train_recall = (class_tp.sum() / (class_tp.sum() + class_fn.sum() + 1e-8)).item()

    per_class_precision = (class_tp / (class_tp + class_fp + 1e-8)).tolist()

    train_loss /= len(dataloader)
    train_acc /= len(dataloader)

    return train_loss, train_acc, train_precision, train_recall, per_class_precision, all_preds, all_labels  

def test_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module,
              device: torch.device,
              num_classes: int) -> Tuple[float, float]:

    model.eval() 

    # Setup test loss and test accuracy values
    test_loss, test_acc, test_precision, test_recall = 0, 0, 0, 0
    all_preds, all_labels = [], []

    class_tp = torch.zeros(num_classes, device=device)
    class_fp = torch.zeros(num_classes, device=device)
    class_fn = torch.zeros(num_classes, device=device)

    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            test_pred_logits = model(X)

            # 2. Calculate and accumulate loss
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            # Calculate and accumulate accuracy
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))
            
            all_preds.append(test_pred_labels.cpu().tolist())
            all_labels.append(y.cpu().tolist())

            for class_idx in range(num_classes):
                class_tp[class_idx] += ((test_pred_labels == class_idx) & (y == class_idx)).sum().item()
                class_fp[class_idx] += ((test_pred_labels == class_idx) & (y != class_idx)).sum().item()
                class_fn[class_idx] += ((test_pred_labels != class_idx) & (y == class_idx)).sum().item()

    test_precision = (class_tp.sum() / (class_tp.sum() + class_fp.sum() + 1e-8)).item()
    test_recall = (class_tp.sum() / (class_tp.sum() + class_fn.sum() + 1e-8)).item()

    per_class_precision = (class_tp / (class_tp + class_fp + 1e-8)).tolist()

    # Adjust metrics to get average loss and accuracy per batch 
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc, test_precision, test_recall, per_class_precision, all_preds, all_labels

def evaluation_step(model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        loss_fn: torch.nn.Module,
        device: torch.device,
        num_classes: int) -> Tuple[float, float, Tuple[List[float], List[float]], List[int], List[int]]:
    model.eval()
    test_loss, test_acc = 0.0, 0.0
    class_tp = torch.zeros(num_classes, device=device)
    class_fp = torch.zeros(num_classes, device=device)
    class_fn = torch.zeros(num_classes, device=device)
    all_preds, all_labels = [], []

    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            preds = model(X)
            loss = loss_fn(preds, y)
            test_loss += loss.item()

            pred_classes = preds.argmax(dim=1)
            test_acc += (pred_classes ==y).sum().item() / len(pred_classes)

            all_preds.extend(pred_classes.cpu().tolist())
            all_labels.extend(y.cpu().tolist())

            for class_idx in range(num_classes):
                class_tp[class_idx] += ((pred_classes ==class_idx) & (y == class_idx)).sum().item()
                class_fp[class_idx] += ((pred_classes ==class_idx) & (y != class_idx)).sum().item()
                class_fn[class_idx] += ((pred_classes !=class_idx) & (y == class_idx)).sum().item()

    test_loss /= len(dataloader)
    test_acc /= len(dataloader)

    per_class_precision = (class_tp / (class_tp + class_fp + 1e-8)).tolist()
    per_class_recall = (class_tp / (class_tp + class_fn + 1e-8)).tolist()

    return test_loss, test_acc, (per_class_precision, per_class_recall),all_preds, all_labels

def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          scheduler: torch.optim.lr_scheduler._LRScheduler,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device,
          num_classes: int) -> Dict[str, List]:

  # Create empty results dictionary
  results = {"train_loss": [],
      "train_acc": [],
      "test_loss": [],
      "test_acc": [],
      "train_precision": [],
      "train_recall": [],
      "test_precision": [],
      "test_recall": [],
      "train_per_class_precision": [],
      "test_per_class_precision": []
  }

  # Loop through training and testing steps for a number of epochs
  for epoch in tqdm(range(epochs)):
      train_loss, train_acc, train_precision, train_recall, train_per_class_precision, train_preds, train_labels = train_step(
              model=model,
              dataloader=train_dataloader,
              loss_fn=loss_fn,
              optimizer=optimizer,
              device=device,
              num_classes=num_classes)

      test_loss, test_acc, test_precision, test_recall, test_per_class_precision, test_preds, test_labels = test_step(
              model=model,
              dataloader=test_dataloader,
              loss_fn=loss_fn,
              device=device,
              num_classes=num_classes)

      # Scheduler step
      if scheduler is not None:
          scheduler.step()
      
      # Print out what's happening
      print(
          f"Epoch: {epoch+1} | "
          f"train_loss: {train_loss:.4f} | train_acc: {train_acc:.4f} | "
          f"train_precision: {train_precision:.4f} | train_recall: {train_recall:.4f} | "
          f"test_loss: {test_loss:.4f} | test_acc: {test_acc:.4f}"
          f"test_precision: {test_precision:.4f} | test_recall: {test_recall:.4f} | "
      )

      # Update results dictionary
      results["train_loss"].append(train_loss)
      results["train_acc"].append(train_acc)
      results["test_loss"].append(test_loss)
      results["test_acc"].append(test_acc)
      results["train_precision"].append(train_precision)
      results["test_precision"].append(test_precision)
      results["train_recall"].append(train_recall)
      results["test_recall"].append(test_recall)
      results["train_per_class_precision"].append(train_per_class_precision)
      results["test_per_class_precision"].append(test_per_class_precision)

  # Return the filled results at the end of the epochs
  return results
