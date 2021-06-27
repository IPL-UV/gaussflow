import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer


class FlowLearnerPlane(pl.LightningModule):
    def __init__(self, model:torch.nn.Module, base_dist, cfg):
        super().__init__()
        self.model = model
        self.base_dist = base_dist
        self.cfg = cfg

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        
        # pass to INN and get transformed variable z and log Jacobian determinant
        z, log_jac_det = self.model(batch)
        
        # calculate the negative log-likelihood of the model with a standard normal prior
        if self.cfg.loss_fn == "inn":
            loss = 0.5*torch.sum(z**2, 1) - log_jac_det
            loss = loss.mean() / z.shape[1]
        else:
            loss = self.base_dist.log_prob(z).sum(1) + log_jac_det
            loss = -loss.mean()
        
        logs = {'train_loss': loss}
        
        return {'loss': loss, 'log': logs}
    
    def validation_step(self, batch, batch_idx):
        
        # pass to INN and get transformed variable z and log Jacobian determinant
        z, log_jac_det = self.model(batch)
        
        # calculate the negative log-likelihood of the model with a standard normal prior
        if self.cfg.loss_fn == "inn":
            loss = 0.5*torch.sum(z**2, 1) - log_jac_det
            loss = loss.mean() / z.shape[1]
        else:
            loss = self.base_dist.log_prob(z).sum(1) + log_jac_det
            loss = -loss.mean()
        
        logs = {'valid_loss': loss}
        
        return {'loss': loss, 'log': logs}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.learning_rate, betas=(0.9, 0.999), weight_decay=0.0)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, 
            milestones=[
                        0.5 * self.cfg.num_epochs, 
                        0.75 * self.cfg.num_epochs,
                        0.9 * self.cfg.num_epochs,
                        ], 
            gamma=0.1)
        
        optimizers = [optimizer]
        lr_schedulers = {'scheduler': lr_scheduler}
        return optimizers, lr_schedulers

class CondFlowLearnerPlane(FlowLearnerPlane):
    def __init__(self, model:torch.nn.Module, cfg):
        super().__init__(model, cfg)



    def training_step(self, batch, batch_idx):
        
        x, y = batch[0], batch[1]
        
        # pass to INN and get transformed variable z and log Jacobian determinant
        z, log_jac_det = self.model(x, c=[y])
        
        # calculate the negative log-likelihood of the model with a standard normal prior
        loss = 0.5*torch.sum(z**2, 1) - log_jac_det
        loss = loss.mean() / z.shape[1]
        
        logs = {'train_loss': loss}
        
        return {'loss': loss, 'log': logs}
    
    def validation_step(self, batch, batch_idx):

        x, y = batch[0], batch[1]
        
        # pass to INN and get transformed variable z and log Jacobian determinant
        z, log_jac_det = self.model(x, c=[y])
        
        # calculate the negative log-likelihood of the model with a standard normal prior
        loss = 0.5*torch.sum(z**2, 1) - log_jac_det
        loss = loss.mean() / z.shape[1]
        
        logs = {'valid_loss': loss}
        
        return {'loss': loss, 'log': logs}