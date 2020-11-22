import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from models import DeepFakesModel
from datasets import DeepFakesDataModule

def main():
    model = DeepFakesModel()
    dataset = DeepFakesDataModule('./data/src_aligned', './data/dst_aligned')
    wandb_logger = WandbLogger()
    trainer = pl.Trainer(
        gpus=1,
        distributed_backend='dp',
        logger=wandb_logger, 
        fast_dev_run=False
    )
    trainer.fit(model, dataset)

if __name__ == '__main__':
    main()