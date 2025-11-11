import torch
import torch.nn.functional as F
from networks.deepflownet import MovementAttn
import gc
import csv
import os
import time
from datetime import datetime
from torch.utils.data import DataLoader
import torch.optim as optim
torch.set_float32_matmul_precision('high')
from torch import GradScaler

from termcolor import colored

from datasets.GT3DDataset import GT3DDataset
from training.backbone import print_gpu_memory

from utils.subsample_utils import canonicalize_and_subsample, canonicalize_and_subsample_iter

def train_mvsnet(hair_folder:str, 
                camera_folder:str,
                checkpoint_save_path:str,  
                multi:bool=False,
                load_checkpoint: bool = False,
                small:bool = True,
                checkpoint_path: str ="/home/student_korfhage/mvshair/data/checkpoints11/checkpoint_epoch_4.pt",
                 ):
    
    log_dir = os.path.join(checkpoint_save_path, "logs")
    os.makedirs(log_dir, exist_ok=True)

    # Get current date and time
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")

    # Define the log file path with timestamp
    log_file = os.path.join(log_dir, f"training_losses_{timestamp}.log")
    csv_file = os.path.join(log_dir, f"training_losses_{timestamp}.csv")

    save_interval = 100  # save every 20 batches
    checkpoint_dir = checkpoint_save_path 
    os.makedirs(checkpoint_dir, exist_ok=True)

    backup_step = 0

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    V = 32             # num views

    dataset = GT3DDataset(hair_folder, camera_folder, device=device)
    print(f"Dataset length: {len(dataset)}")
    gc.collect()
    torch.cuda.empty_cache()

    if small:
        small_split = 0.9
        num_val = int(len(dataset) * small_split)
        num_train = len(dataset) - num_val
        train_dataset, _ = torch.utils.data.random_split(dataset, [num_train, num_val])
        print(f"Train samples: {len(train_dataset)}")
    else:
        # Split dataset into train/val
        val_split = 0.1
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
            shuffle=True, #TODO: should be true
            num_workers=0,
            pin_memory=False,
            # prefetch_factor=1,
            # collate_fn=DeviceCollate(device)
        )
    else:
        dataloader = DataLoader(
            train_dataset,
            batch_size=1,
            shuffle=True, #TODO: should be true
            num_workers=1,
            pin_memory=False,
            # collate_fn=DeviceCollate(device)
        )

    # model & optimizer stuff
    model = MovementAttn(in_feat=1, token_dim=64, vit_heads=4, num_views=V, dir_lambda=0.7).to(device)
    model = torch.compile(model) 
    optimizer = optim.Adam(list(model.parameters()), lr=1e-2)

    start_epoch = 0
    if load_checkpoint and checkpoint_path is not "" and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        print(colored(f"Resuming training from epoch {start_epoch}", "green"))

    # Initialize CSV file with headers
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Batch", "Loss_Total", "CE" ,"conn_l1_attn", "conn_l1_res", "flow_consistency"])

    num_epochs = 1000
    sub_batch_epochs = 100
    for epoch in range(start_epoch, num_epochs):
        model.train()

        print(colored( f"New EPOCH:  {epoch}", "green"))

        scaler = GradScaler('cuda')

        for batch_idx, batch in enumerate(dataloader):
            print(colored( f"New Batch with Index (batch):  {batch_idx}", "magenta"))

            # print(f'Sanity Check | batch indexes: {batch.get_strand_idx_from_idx[batch_idx]} | prev_batch indexes: {dataset[batch_idx-1].get_strand_idx_from_idx[batch_idx]}') #TODO:

            with torch.no_grad():
                imgs_raw = batch['flow_map'].squeeze(0).squeeze(0).to(device)
                print(f'IMGS Shape: {imgs_raw.shape}')

                scale = 1.0   
                new_h, new_w = int(imgs_raw.shape[-2] * scale), int(imgs_raw.shape[-1] * scale)
                imgs = F.interpolate(imgs_raw, size=(new_h, new_w), mode='bilinear', align_corners=False)

                del imgs_raw

                prev_imgs_raw = batch['prev_flow_map'].squeeze(0).squeeze(0).to(device)
                print(f'IMGS Shape: {prev_imgs_raw.shape}')

                scale = 0.25    
                new_h, new_w = int(prev_imgs_raw.shape[-2] * scale), int(prev_imgs_raw.shape[-1] * scale)
                imgs = F.interpolate(prev_imgs_raw, size=(new_h, new_w), mode='bilinear', align_corners=False)

                del prev_imgs_raw

                print(f'Flow Map Shape: {imgs.shape} (Subsamples IMGS)')

            for i, chunk in enumerate(canonicalize_and_subsample_iter(batch, device, num_points=100, num_prev_points=100000)):
                print("chunk", i, {k: (None if v is None else tuple(v.shape)) for k, v in chunk.items()})
                if i >= 2:
                    break


            start_time = time.time()

            batch_log = []  # collect logs for this batch
            batch_losses = []

            for sub_batch_idx, sub_batch in enumerate(canonicalize_and_subsample_iter(batch, device, num_points=100, num_prev_points=100000)):

                clean = sub_batch

                data = {
                    'imgs': imgs.float().to(device),
                    'sample_coord': clean['sample_coord'].float(),
                    'pts_world': clean['pts_world'].float(),
                    'pts_view': clean['pts_view'].float() if clean['pts_view'] is not None else None,
                    'flow_view': clean['real_flow_view'].float(),
                    'real_flow_view': clean['real_flow_view'].float(),
                    'prev_imgs': imgs.float(),  # reuse for sanity
                    'prev_sample_coord': clean['prev_sample_coord'].float(),
                    'prev_pts_world': clean['prev_pts_world'].float(),
                    'prev_pts_view': clean['prev_pts_view'].float() if clean['prev_pts_view'] is not None else None,
                    'gt_prev_pos': clean['gt_prev_pos'].float(),
                }

                del clean

                with torch.autocast("cuda"):
                    out = model(data)

                    loss = out['losses']
                    total_loss = sum(loss.values())

                     # Backward pass
                    #ignore this. It works 
                    scaler.scale(total_loss).backward() #TODO: Really tho?

                # store logs in memory
                losses_str = ', '.join([f"{k}: {v:.4f}" for k, v in loss.items()])
                batch_log.append(f"SubBatch {sub_batch_idx}: {losses_str} | Total: {total_loss.item():.4f}\n")
                batch_losses.append((total_loss.item(), loss))

                


                print_gpu_memory()

                print(datetime.now())
                print(f'Batch Index (Global): {batch_idx}')
                print(f'Epoch: {epoch}')
                end_time = time.time()

                time_taken = end_time - start_time

                minutes = int(time_taken // 60)
                seconds = int(time_taken % 60)
                milliseconds = int((time_taken - int(time_taken)) * 1000)

                print(f'Time needed to run one backward: {minutes:02d}:{seconds:02d}.{milliseconds:03d}')

                backup_step += 1

                # periodic checkpointing
                if backup_step % save_interval == 0:
                    ckpt_path = os.path.join(checkpoint_dir, f"model_step_{backup_step}.pt")
                    checkpoint = {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                    }
                    torch.save(checkpoint, ckpt_path)
                    print(f"Saved checkpoint at step {backup_step}: {ckpt_path}")

                if sub_batch_idx == sub_batch_epochs:
                    break

          

                start_time = time.time()

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
            except RuntimeError as e:
                print("Optimizer step failed:", e)
                optimizer.zero_grad(set_to_none=True) #TODO: Maybe bad if error

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        checkpoint_file = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
        torch.save(checkpoint, checkpoint_file)
        print(colored(f"Checkpoint saved at epoch {epoch}", "green"))

        if not small:
                
            # validation phase
            model.eval()
            val_losses = []
            with torch.no_grad():
                for val_batch_idx, val_batch in enumerate(val_loader):
                    imgs_raw = val_batch['flow_map'].squeeze(0).squeeze(0).to(device)
                    prev_imgs_raw = val_batch['prev_flow_map'].squeeze(0).squeeze(0).to(device)

                    # same preprocessing as in train
                    scale = 0.25
                    new_h, new_w = int(imgs_raw.shape[-2] * scale), int(imgs_raw.shape[-1] * scale)
                    imgs = F.interpolate(imgs_raw, size=(new_h, new_w), mode='bilinear', align_corners=False)
                    prev_imgs = F.interpolate(prev_imgs_raw, size=(new_h, new_w), mode='bilinear', align_corners=False)

                    for sub_batch in canonicalize_and_subsample_iter(val_batch, device, num_points=100, num_prev_points=100000):
                        data = {
                            'imgs': imgs.float(),
                            'sample_coord': sub_batch['sample_coord'].float(),
                            'pts_world': sub_batch['pts_world'].float(),
                            'pts_view': sub_batch['pts_view'].float() if sub_batch['pts_view'] is not None else None,
                            'flow_view': sub_batch['real_flow_view'].float(),
                            'real_flow_view': sub_batch['real_flow_view'].float(),
                            'prev_imgs': prev_imgs.float(),
                            'prev_sample_coord': sub_batch['prev_sample_coord'].float(),
                            'prev_pts_world': sub_batch['prev_pts_world'].float(),
                            'prev_pts_view': sub_batch['prev_pts_view'].float() if sub_batch['prev_pts_view'] is not None else None,
                            'gt_prev_pos': sub_batch['gt_prev_pos'].float(),
                        }

                        out = model(data)
                        loss_val = sum(out['losses'].values())
                        val_losses.append(loss_val.item())


            # validation loss
            val_loss_mean = sum(val_losses) / len(val_losses)
            print(colored(f"Validation Loss (epoch {epoch}): {val_loss_mean:.4f}", "yellow"))

            # log to file
            with open(log_file, "a") as f:
                f.write(f"Validation Loss (epoch {epoch}): {val_loss_mean:.4f}\n")

            with open(csv_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([epoch, "val", val_loss_mean])

    print("==== Training completed ====")
