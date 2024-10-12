from utils import RunningAverage, calculate_iou, get_logger
import torch
from torch.utils.tensorboard import SummaryWriter
import os
from tqdm import tqdm

logger=get_logger('UNetTrainer')

def move_to_device(data, device):
    if isinstance(data, tuple) or isinstance(data, list):
        return tuple([move_to_device(x, device) for x in data])
    else:
        data = data.to(device)
        return data

def eval(model, loader, loss_criterion, device):
    model.eval()
    loss_scores = RunningAverage()
    iou_scores = RunningAverage()
    for inputs, labels in loader:
        inputs, labels = move_to_device(inputs, device), move_to_device(labels, device)
        outputs = model(inputs).squeeze()

        # Reverse sigmoid(during training, when calculate loss, sigmoid apllied)
        loss = loss_criterion(torch.logit(outputs, eps=1e-6), labels)

        predicts = torch.where(outputs > 0.5, torch.tensor(1, dtype=torch.int16, device=device), torch.tensor(0, dtype=torch.int16, device=device))
        iou = calculate_iou(labels, predicts)
        
        loss_scores.update(loss.item(), inputs.shape[0])
        iou_scores.update(iou.mean(), iou.shape[0])
    return loss_scores.avg, iou_scores.avg

class Trainer:
    num_epoch = 0
    best_val_iou = 0

    def __init__(self, model:torch.nn.Module, optimizer, lr_scheduler, loss_criterion, loaders, max_num_epochs, writer, logdir, device) -> None:
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss_criterion = loss_criterion
        self.loaders = loaders
        self.max_num_epochs = max_num_epochs
        self.writer = writer
        self.logdir = logdir
        self.device = device
        
        if not os.path.exists(logdir):
            os.mkdir(logdir)
    
    def train(self):
        self.model.train()
        train_losses = RunningAverage()
        for inputs, labels in self.loaders["Train"]:
            inputs, labels = move_to_device(inputs, self.device), move_to_device(labels, self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs).squeeze()
            loss = self.loss_criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            train_losses.update(loss.item(), inputs.shape[0])
        return train_losses.avg
    
    def fit(self):
        val_loss, val_iou  = eval(self.model, self.loaders["Val"], self.loss_criterion, device=self.device)
        print(f"Before training | Val Loss: {val_loss:.5f} | Val IoU Score: {val_iou:.4f}")
        if val_iou > 0.95 and val_iou < 0.99:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = 0.0002
        if val_iou > 0.99:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = 0.0001
        
        logger.info("Train begin...")
        for i in range(1, self.max_num_epochs + 1):
            train_loss = self.train()
            val_loss, val_iou  = eval(self.model, self.loaders["Val"], self.loss_criterion, device=self.device)

            logger.info(f"Epoch {i:02d}/{self.max_num_epochs} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f} | Val IoU Score: {val_iou:.4f}")
            
            # # 在每个 epoch 结束后，根据验证损失调整学习率
            # self.lr_scheduler.step(val_loss)

            Trainer.num_epoch += 1
            self.writer.add_scalar('Loss/Train', train_loss, Trainer.num_epoch)
            self.writer.add_scalar('Loss/Val', val_loss, Trainer.num_epoch)  # Log validation IoU
            self.writer.add_scalar('IoU/Val', val_iou, Trainer.num_epoch)  # Log validation IoU
            current_lr = self.optimizer.param_groups[0]['lr']
            self.writer.add_scalar('Learning_Rate', current_lr, Trainer.num_epoch)  # Log the learning rate

            if val_iou > Trainer.best_val_iou:
                Trainer.best_val_iou = val_iou
                logger.info(f"Best val iou: {Trainer.best_val_iou}, save checkpoint...")
                self.save_checkpoint()

            if val_iou > 0.99:
                break
        
        return train_loss, val_loss, val_iou
        # self.writer.close()
        # self.save_checkpoint(self.max_num_epochs)

    def save_checkpoint(self):
        state = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        checkpoint_name = 'checkpoint.pytorch'
        file_path = os.path.join(self.logdir, checkpoint_name)
        torch.save(state, file_path)
