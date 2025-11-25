import torch
import torch.nn.functional as F
import gc
import csv
import os
import time
from datetime import datetime
from torch.utils.data import DataLoader
import torch.optim as optim
from torch import GradScaler
from termcolor import colored
from tqdm import tqdm

torch.set_float32_matmul_precision('high') # Relevant to not get annoyed by warnings

# Own Imports

from networks.deepflownet import DeepFlowNet
from datasets.GT3DDataset import GT3DDataset
from utils.subsample_utils import subsample_iter

def set_seeds(seed=42):
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using CUDA
    # Ensure deterministic behavior for certain operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def print_gpu_memory():
    
    if torch.cuda.is_available():
        # get current GPU from torch
        device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(device)

        # get memory stats
        total_memory = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
        allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)
        reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)
        free = total_memory - reserved

        # print the stats
        print(f"GPU: {device_name}")
        print(f"Total Memory: {total_memory:.2f} GB")
        print(f"Allocated: {allocated:.2f} GB")
        print(f"Reserved: {reserved:.2f} GB")
        print(f"Free: {free:.2f} GB")
    
    else:
        print("CUDA is not available.")

def train_hairflownet(
                hair_folder:str, 
                camera_folder:str,
                checkpoint_save_path:str,  
                multi:bool=False,
                load_checkpoint: bool = False,
                small:bool = True,
                checkpoint_path: str = "",
                ):
    
    # Use the same seed every time
    set_seeds(seed=42)
    
    log_dir = os.path.join(checkpoint_save_path, "logs")
    os.makedirs(log_dir, exist_ok=True)

    # Get current date and time
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")

    log_file = os.path.join(log_dir, f"training_losses_{timestamp}.log")
    csv_file = os.path.join(log_dir, f"training_losses_{timestamp}.csv")

    save_interval = 10000  # save every n subbatches
    checkpoint_dir = checkpoint_save_path 
    os.makedirs(checkpoint_dir, exist_ok=True)

    backup_step = 0 # counter for offbatch backups

    device = 'cuda' if torch.cuda.is_available() else 'cpu' # device for calculation

    V = 32 # num views (cameras)

    dataset = GT3DDataset(hair_folder, camera_folder, device=device)
    print(f"Dataset length: {len(dataset)}")
    gc.collect()
    torch.cuda.empty_cache()

    if 0:
        # take only a small dataset for sanity training
        small_split = 1.0
        num_val = int(len(dataset) * small_split)
        num_train = len(dataset) - num_val
        train_dataset, _ = torch.utils.data.random_split(dataset, [num_train, num_val])
        print(f"Train samples: {len(train_dataset)}")
    else:
        # Split dataset into training and validation
        val_split = 0.7
        num_val = int(len(dataset) * val_split)
        num_train = len(dataset) - num_val
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [num_train, num_val])

        print(f"Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

        val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=1,
            pin_memory=False,
        )

    if multi:
        dataloader = DataLoader(
            train_dataset,
            batch_size=1,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
            # prefetch_factor=1,
            # collate_fn=DeviceCollate(device)
        )
    else:
        dataloader = DataLoader(
            train_dataset,
            batch_size=1,
            shuffle=True,
            num_workers=1,
            pin_memory=False,
            # collate_fn=DeviceCollate(device)
        )

    # model & optimizer stuff
    model = DeepFlowNet(in_feat=1, token_dim=64, vit_heads=4, num_views=V, dir_lambda=0.2).to(device)
    model = torch.compile(model) # Compile model for faster training
    optimizer = optim.Adam(list(model.parameters()), lr=1e-3) # type: ignore #TODO: lr anpassen

    ############### Load checkpoint if available #####################################################

    start_epoch = 0
    if load_checkpoint and checkpoint_path != "" and os.path.exists(checkpoint_path) and checkpoint_path.startswith('checkpoint_epoch'):
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"]) # type: ignore
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1

        print(colored(f"Resuming training from epoch {start_epoch}", "green"))
    
    elif load_checkpoint and checkpoint_path != "" and os.path.exists(checkpoint_path) and checkpoint_path.startswith('model'):

        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"]) # type: ignore
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        current_epoch = checkpoint["epoch"]
        
        print(colored(f"Resuming training in epoch {current_epoch}", "green"))

    # Initialize CSV file with headers
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Batch", "Loss_Total", "CE" ,"conn_l1_attn", "conn_l1_res", "flow_consistency"])

    num_epochs = 100
    sub_batch_epochs = 100
    
    # Training loop
    for epoch in range(start_epoch, num_epochs):
        model.train() # type: ignore

        print(colored( f"New EPOCH:  {epoch}", "green"))

        scaler = GradScaler('cuda')

        for batch_idx, batch in enumerate(dataloader):
            print(colored( f"New Batch with Index (batch):  {batch_idx}", "magenta"))

            # print(f'Sanity Check | batch indexes: {batch.get_strand_idx_from_idx[batch_idx]} | prev_batch indexes: {dataset[batch_idx-1].get_strand_idx_from_idx[batch_idx]}') #TODO:

            # Reshape Images to fit on the GPU memory during training
            with torch.no_grad():
                imgs_raw = batch['flow_map'].squeeze(0).squeeze(0).to(device)
                print(f'IMGS Shape: {imgs_raw.shape}')

                scale = 0.33 
                new_h, new_w = int(imgs_raw.shape[-2] * scale), int(imgs_raw.shape[-1] * scale)
                imgs = F.interpolate(imgs_raw, size=(new_h, new_w), mode='bilinear', align_corners=False)

                del imgs_raw

                prev_imgs_raw = batch['prev_flow_map'].squeeze(0).squeeze(0).to(device)
                print(f'IMGS Shape: {prev_imgs_raw.shape}')
    
                new_h, new_w = int(prev_imgs_raw.shape[-2] * scale), int(prev_imgs_raw.shape[-1] * scale)
                prev_imgs = F.interpolate(prev_imgs_raw, size=(new_h, new_w), mode='bilinear', align_corners=False)

                del prev_imgs_raw

                print(f'Flow Map Shape: {imgs.shape} (Subsamples IMGS)')


            start_time = time.time()

            batch_log = []  # collect logs for this batch
            batch_losses = []

            for sub_batch_idx, clean in tqdm(enumerate(subsample_iter(batch, device, num_points=100, num_prev_points=1000))):

                data = {
                    'imgs': imgs.float().to(device),
                    'sample_coord': clean['sample_coord'].float(), # type: ignore
                    'pts_world': clean['pts_world'].float(), # type: ignore
                    'pts_view': clean['pts_view'].float() if clean['pts_view'] is not None else None,
                    'flow_view': clean['flow_view'].float(), # type: ignore
                    'real_flow_view': clean['real_flow_view'].float(), # type: ignore
                    'prev_imgs': prev_imgs.float(),  # reuse for sanity
                    'prev_sample_coord': clean['prev_sample_coord'].float(), # type: ignore
                    'prev_pts_world': clean['prev_pts_world'].float(), # type: ignore
                    'prev_pts_view': clean['prev_pts_view'].float() if clean['prev_pts_view'] is not None else None,
                    'gt_prev_pos': clean['gt_prev_pos'].float(), # type: ignore
                }

                del clean

                with torch.autocast("cuda"):

                    out = model(data)

                    loss = out['losses']
                    total_loss = sum(loss.values())

                    # Backward pass
                    #ignore this. It works 
                    scaler.scale(total_loss).backward()  # type: ignore

                # store logs in memory (probably a factor in the slowness)
                losses_str = ', '.join([f"{k}: {v:.4f}" for k, v in loss.items()])
                batch_log.append(f"SubBatch {sub_batch_idx}: {losses_str} | Total: {total_loss.item():.4f}\n") # type: ignore
                batch_losses.append((total_loss.item(), loss)) # type: ignore

                # print_gpu_memory()

                backup_step += 1

                # periodic checkpointing
                if backup_step % save_interval == 0:
                    ckpt_path = os.path.join(checkpoint_dir, f"model_step_{backup_step}.pt")
                    checkpoint = {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(), # type: ignore
                        "optimizer_state_dict": optimizer.state_dict(),
                    }
                    torch.save(checkpoint, ckpt_path)
                    print(f"Saved checkpoint at step {backup_step}: {ckpt_path}")

                if sub_batch_idx == sub_batch_epochs:
                    break

            mean_total_loss = sum(x[0] for x in batch_losses) / len(batch_losses)
            print(colored(f"[Batch {batch_idx}] Avg Total Loss: {mean_total_loss:.4f}", "red"))

            with open(log_file, "a") as f:
                f.write(f"\n=== Batch {batch_idx} Summary ===\n")
                f.writelines(batch_log)
                f.write(f"Average Total Loss: {mean_total_loss:.4f}\n\n")

            with open(csv_file, "a", newline="") as f:
                writer = csv.writer(f)
                # log only average per batch, not every sub-batch
                avg_sub_losses = {
                    k: sum(x[1][k].item() for x in batch_losses) / len(batch_losses)
                    for k in batch_losses[0][1].keys()
                }
                writer.writerow([epoch, batch_idx, mean_total_loss] + list(avg_sub_losses.values()))


            try:
                scaler.step(optimizer)   
                scaler.update()
                optimizer.zero_grad(set_to_none=True)          
            except RuntimeError as e:
                print("Optimizer step failed:", e)
                optimizer.zero_grad(set_to_none=True) #TODO: Maybe bad if error


        ################ Epoch Checkpoint ###############################

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(), # type: ignore
            "optimizer_state_dict": optimizer.state_dict(),
        }
        checkpoint_file = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
        torch.save(checkpoint, checkpoint_file)
        print(colored(f"Checkpoint saved at epoch {epoch}", "green")) # High viz logging

        # validation phase
        if 0: #skip for sanity checking
                
            
            model.eval() # type: ignore # Turn off certain Layers for Evaluation

            val_losses = []

            with torch.no_grad():
                for val_batch_idx, val_batch in enumerate(val_loader): # type: ignore
                    
                    imgs_raw = val_batch['flow_map'].squeeze(0).squeeze(0).to(device)
                    prev_imgs_raw = val_batch['prev_flow_map'].squeeze(0).squeeze(0).to(device)

                    # same preprocessing as in train
                    
                    scale = 0.25
                    new_h, new_w = int(imgs_raw.shape[-2] * scale), int(imgs_raw.shape[-1] * scale)
                    imgs = F.interpolate(imgs_raw, size=(new_h, new_w), mode='bilinear', align_corners=False)
                    prev_imgs = F.interpolate(prev_imgs_raw, size=(new_h, new_w), mode='bilinear', align_corners=False)

                    for sub_batch in subsample_iter(val_batch, device, num_points=100, num_prev_points=100000):
                        data = {
                            'imgs': imgs.float(),
                            'sample_coord': sub_batch['sample_coord'].float(), # type: ignore
                            'pts_world': sub_batch['pts_world'].float(), # type: ignore
                            'pts_view': sub_batch['pts_view'].float(), # type: ignore
                            'flow_view': sub_batch['real_flow_view'].float(), # type: ignore
                            'real_flow_view': sub_batch['real_flow_view'].float(), # type: ignore
                            'prev_imgs': prev_imgs.float(),
                            'prev_sample_coord': sub_batch['prev_sample_coord'].float(), # type: ignore
                            'prev_pts_world': sub_batch['prev_pts_world'].float(), # type: ignore
                            'prev_pts_view': sub_batch['prev_pts_view'].float(), # type: ignore
                            'gt_prev_pos': sub_batch['gt_prev_pos'].float(), # type: ignore
                        }

                        out = model(data)
                        loss_val = sum(out['losses'].values())
                        val_losses.append(loss_val.item()) # type: ignore


            # validation loss
            val_loss_mean = sum(val_losses) / len(val_losses)
            print(colored(f"Validation Loss (epoch {epoch}): {val_loss_mean:.4f}", "yellow"))

            # log to file
            with open(log_file, "a") as f:
                f.write(f"Validation Loss (epoch {epoch}): {val_loss_mean:.4f}\n")

            with open(csv_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([epoch, "val", val_loss_mean])

    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  Training completed !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")