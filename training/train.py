from tqdm import tqdm
import torch
import os

def train_step(
        model, 
        dataloader, 
        loss_fn, 
        optimizer, 
        device
):
    """The forward pass of a single epoch for training the model.

    Args:
    model - The model to be trained.
    dataloader - The training dataset.
    loss_fn - The combination of focal and Dice loss
    optimizer - Updates model's parameters to minimize loss function.
    device - Converts image and masks to run on CUDA along with model.

    Returns:
    The training loss of a single forward pass during training.
    """

    # Put model in training mode
    model.train()

    # Keeps track of total loss
    train_loss = 0

    for batch, (image, mask) in enumerate(tqdm(dataloader)):
        # Reset gradients after each batch to avoid accumulating gradients from previous run
        optimizer.zero_grad()
        
        # Convert image and mask to run on CUDA
        image, mask = image.to(device), mask.to(device)

        # Get model's predicted masks
        pred_mask = model(image)

        # Compare prediction with truth masks to get overall loss of batch
        loss = loss_fn(pred_mask, mask)

        # Accumulate loss
        train_loss += loss.item()
        
        # Backpropagation - computes gradients
        loss.backward()

        # Adjusts model's parameters with computed gradients
        optimizer.step()

    # Divide total loss by length of dataset to get average training loss
    train_loss /= len(dataloader)

    return train_loss

def val_step(
        model, 
        dataloader, 
        loss_fn, device
):
    """Evaluates the model on unseen data during training.

    Args:
    model - The model to be trained.
    dataloader - The training dataset.
    loss_fn - The combination of focal and Dice loss
    device - Converts image and masks to run on CUDA along with model.

    Returns:
    The validation loss value
    """
        
    # Put model on evaluation mode
    model.eval()

    val_loss = 0

    # Disable gradients
    with torch.no_grad():
        for batch, (images, masks) in enumerate(tqdm(dataloader)):
            images, masks = images.to(device), masks.to(device)

            pred_masks = model(images)
            loss = loss_fn(pred_masks, masks)
            val_loss += loss.item()

    val_loss /= len(dataloader)

    return val_loss

def train_model(
        model, 
        train_dataloader, 
        valid_dataloader, 
        loss_fn, 
        optimizer,
        scheduler,
        start_epoch, 
        epochs, 
        device, 
        path_dir,
        checkpoint_num
):
    """The training loop for the model.

    This function keeps track of training loss and validation loss for the entire training phase.
    It also does checkpointing, which saves the current epoch, model's state, optimizer's state, training and validation losses.
    
    Args:
    model - The model to be trained.
    train_dataloader - The training dataset
    valid_dataloader - The validation dataset
    loss_fn - Compares truth masks with model's predicted masks
    optimizer - Adjusts the model's parameters after backpropagation.
    scheduler - Keeps track of validation loss and adjusts model's learning rate if no improvement has been made during training.
    start_epoch - Starts training at specific epochs. Helpful for resuming training if paused or interrupted.
    epochs - The total number of epochs to run training for.
    device - Runs model, images, and masks on CUDA.
    path_dir - Directory to save checkpoints at
    checkpoint_num - Folder to save checkpoints at in path_dir.

    Returns:
    Two Python 1D lists that contain values for training and validation loss for the entire training phase.
    """

    # Create checkpoint directory if it doesn't exist yet for a new experiment
    if not os.path.exists(f"{path_dir}/checkpoint_{checkpoint_num}"):
        os.makedirs(f"{path_dir}/checkpoint_{checkpoint_num}", exist_ok=True)
    
    train_losses, val_losses = [], []

    # Keep track of lowest validation loss
    lowest_val_loss = float("inf")

    for epoch in range(start_epoch, epochs):
        # Get training and validation loss across each dataloader
        train_loss = train_step(model, train_dataloader, loss_fn, optimizer, device)
        val_loss = val_step(model, valid_dataloader, loss_fn, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Keep track of validation loss for scheduler to reduce LR if needed
        scheduler.step(val_loss)

        print(f'Epoch: {epoch + 1} | train_loss: {train_loss} | val_loss: {val_loss}')

        # Only save checkpoints if its the best state of the model currently
        if val_loss < lowest_val_loss:
            lowest_val_loss = val_loss
            
            # Checkpointing
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_losses": train_losses,
                "valid_losses": val_losses 
            }, f"{path_dir}/checkpoint_{checkpoint_num}/model_weights.pth")

            print("Saving best new checkpoint!")
        
        print()

    print("Training complete!")
    return train_losses, val_losses