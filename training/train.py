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
    model.train()

    train_loss = 0

    for batch, (image, mask) in enumerate(tqdm(dataloader)):
        optimizer.zero_grad()
        
        image, mask = image.to(device), mask.to(device)
        pred_mask = model(image)
    
        loss = loss_fn(pred_mask, mask)
        train_loss += loss.item()
        
        loss.backward()
        optimizer.step()

    train_loss /= len(dataloader)

    return train_loss

def val_step(
        model, 
        dataloader, 
        loss_fn, device
):
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
    if not os.path.exists(f"{path_dir}/checkpoint_{checkpoint_num}"):
        os.makedirs(f"{path_dir}/checkpoint_{checkpoint_num}", exist_ok=True)
    
    train_losses, val_losses = [], []
    lowest_val_loss = float("inf")

    for epoch in range(start_epoch, epochs):
        # Get training and validation loss across each dataloader
        train_loss = train_step(model, train_dataloader, loss_fn, optimizer, device)
        val_loss = val_step(model, valid_dataloader, loss_fn, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        scheduler.step(val_loss)

        print(f'Epoch: {epoch + 1} | train_loss: {train_loss} | val_loss: {val_loss}')

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