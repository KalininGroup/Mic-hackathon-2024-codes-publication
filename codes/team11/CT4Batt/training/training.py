import torch
import torch.optim as optim
from utils.helper import visualize_predictions
import torch.nn as nn
from models.cnn_attention import CNN2D as CNN_model
from models.cnnKan_attention import CNN2DKan as CNN_model_w_Kan

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

# Training model
def train(config, train_loader, val_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    smooth_l1 = nn.SmoothL1Loss()
    iou_loss_fn = iou_loss

    # Model initialization
    model = CNN_model_w_Kan().to(device)
    #model = CNN_model().to(device)

    # Optimizer and Scheduler
    optimizer = optim.Adam(model.parameters(), lr=config["lr"], weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    # Training Loop
    num_epochs = config.get("num_epochs", 1000)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for batch_idx, (images_org, vertices) in enumerate(train_loader):
            images, vertx = images_org.to(device), vertices.to(device)
            optimizer.zero_grad()
            outputs = model(images)  # Forward pass

            # Ensure outputs and targets are floats
            outputs = outputs.float()
            vertx = vertx.float()

            # Calculate loss
            loss = smooth_l1(outputs, vertx) + iou_loss_fn(outputs, vertx)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()

        # Validation Loop
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for images_org, vertices in val_loader:
                optimizer.zero_grad()
                outputs = model(images)  # Forward pass

                # Ensure outputs and targets are floats
                outputs = outputs.float()
                vertx = vertx.float()

                # Calculate loss
                loss = smooth_l1(outputs, vertx) + iou_loss_fn(outputs, vertx)

                val_loss += loss.item()

        # Average Losses
        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        # Step the scheduler
        scheduler.step(avg_val_loss)

        # Print Training Progress
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}")

        # Visualization every 20 epochs
        if epoch % 200 == 0:
            visualize_predictions(outputs, vertx, images, idx=0)

    # Save the Model
    torch.save(model.state_dict(), 'final_model.pth')
    print("Model saved as 'final_model.pth'")
    return {"loss": avg_val_loss}

# Intersection over Union (IoU) loss for bounding box regression
def iou_loss(pred, target):
    """
    IoU Loss for bounding box regression.
    pred: Predicted bounding boxes, shape (N, 4)
    target: Ground truth bounding boxes, shape (N, 4)
    """
    x1_pred, y1_pred, x2_pred, y2_pred = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    x1_target, y1_target, x2_target, y2_target = target[:, 0], target[:, 1], target[:, 2], target[:, 3]

    # Calculate intersection
    x1_int = torch.max(x1_pred, x1_target)
    y1_int = torch.max(y1_pred, y1_target)
    x2_int = torch.min(x2_pred, x2_target)
    y2_int = torch.min(y2_pred, y2_target)

    int_area = torch.clamp(x2_int - x1_int, min=0) * torch.clamp(y2_int - y1_int, min=0)

    # Calculate union
    pred_area = (x2_pred - x1_pred) * (y2_pred - y1_pred)
    target_area = (x2_target - x1_target) * (y2_target - y1_target)
    union_area = pred_area + target_area - int_area

    # IoU
    iou = int_area / (union_area + 1e-6)
    loss = 1 - iou  # IoU loss
    return loss.mean()
    
# Load model function
def load_model(model_path='final_model.pth'):

    model = CNN_model_w_Kan().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

