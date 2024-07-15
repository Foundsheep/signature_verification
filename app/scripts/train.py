import torch
from torch import nn
import torch.nn.functional as F
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).absolute().parent.parent))

from models.siamese_network import *
from configs import *
from datasets.korean_aihub_sentence_dataset import get_dataloader, KoreanTypographyDataset

# PyTorch TensorBoard support
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tqdm import tqdm


def get_instances(is_new):
    model = SiameseNetwork_OutputEmbedding()
    if LOSS == "BCELoss":
        loss_fn = nn.BCELoss()
    elif LOSS == "TripletMarginWithDistanceLoss":
        loss_fn = nn.TripletMarginWithDistanceLoss(distance_function=lambda x, y: 1.0 - F.cosine_similarity(x, y))

    if is_new:
        model = SiameseNetwork()
    else:
        from models import model

    model.to(DEVICE)
    optim = torch.optim.Adam(model.parameters())
    return model, loss_fn, optim


def train_one_epoch(model, train_loader, epoch_index, optimizer, loss_fn, tb_writer):

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(train_loader):
        img_1, img_2, label = data
        img_1 = img_1.to(DEVICE)
        img_2 = img_2.to(DEVICE)
        label = label.to(DEVICE)

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        out = model(img_1, img_2)
        
        # Compute the loss and its gradients
        try:
            loss = loss_fn(out, label)
        except:
            # 마지막 배치 데이터 수가 1개일 경우 label.size() == torch.size([])로 되어 에러 발생
            if label.dim() == 0:
                label = label.unsqueeze(dim=0)

            loss = loss_fn(out, label)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        last_loss = loss.item()
        if (i+1) % 10 == 0:
            print(f'  iteration [{i + 1}] loss: [{last_loss :.5f}]')
            tb_x = epoch_index * len(train_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
    return last_loss



def run(model, loss_fn, optim, train_dl, val_dl):

    # Initializing in a separate cell so we can easily add more epochs to the same run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(f'runs/sample_test_{timestamp}')

    best_vloss = 1_000_000.
    
    checkpoint_dir = Path(f"./model_checkpoints/{timestamp}")
    if not checkpoint_dir.exists():
        checkpoint_dir.mkdir(parents=True)
        print(f"[{str(checkpoint_dir)}] directory is made!")

    for epoch in tqdm(range(EPOCHS)):
        print(f'EPOCH {epoch+1}:')

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        last_loss = train_one_epoch(model=model,
                                    train_loader=train_dl,
                                    epoch_index=epoch,
                                    optimizer=optim,
                                    loss_fn=loss_fn,
                                    tb_writer=writer)



        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
        model.eval()

        # Disable gradient computation and reduce memory consumption.
        running_vloss = 0.0
        with torch.no_grad():
            for i, vdata in enumerate(val_dl):
                v_img_1, v_img_2, v_label = vdata

                v_img_1 = v_img_1.to(DEVICE)
                v_img_2 = v_img_2.to(DEVICE)
                v_label = v_label.to(DEVICE)

                v_out = model(v_img_1, v_img_2)
                vloss = loss_fn(v_out, v_label)
                running_vloss += vloss
                
        avg_vloss = running_vloss / (i + 1)
        print(f'LOSS train [{last_loss :.5f}] / valid [{avg_vloss :.5f}]')

        # Log the running loss averaged per batch
        # for both training and validation
        writer.add_scalars('Training vs. Validation Loss',
                        { 'Training' : last_loss, 'Validation' : avg_vloss },
                        epoch + 1)
        writer.flush()

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss or (epoch + 1) == EPOCHS:
            best_vloss = avg_vloss
            model_path = f'{str(checkpoint_dir)}/epoch_{str(epoch+1).zfill(4)}.pt'
            torch.save(model.state_dict(), model_path)

    print("================= END =================")

if __name__ == "__main__":
    model, loss_fn, optim = get_instances(is_new=True)
    train_dl = get_dataloader(root_dir=ROOT_DIR,
                              is_train=True,
                              is_sanity_check=None,
                              batch_size=BATCH_SIZE,
                              shuffle=True)
    val_dl = get_dataloader(root_dir=ROOT_DIR,
                            is_train=False,
                            is_sanity_check=None,
                            batch_size=BATCH_SIZE,
                            shuffle=True)
    run(model=model,
        loss_fn=loss_fn,
        optim=optim,
        train_dl=train_dl,
        val_dl=val_dl)