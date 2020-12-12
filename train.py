import configs
from absl import app, flags
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from models import DeepFakesModel
from datasets import DeepFakesDataModule

FLAGS = flags.FLAGS


def main(argv=None):
    model = DeepFakesModel()
    dataset = DeepFakesDataModule(**FLAGS.flag_values_dict())
    wandb_logger = WandbLogger()
    trainer = pl.Trainer(
        gpus=1,
        logger=wandb_logger,
        distributed_backend='dp',
        fast_dev_run=False,
    )
    trainer.fit(model, dataset)

if __name__ == '__main__':
    app.run(main)