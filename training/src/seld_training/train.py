"""Main training script using PyTorch Lightning."""

from __future__ import annotations

from pathlib import Path

import lightning as L
import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

import structlog

from seld_training.model.resnet_conformer import ResNetConformer
from seld_training.loss.adpit_loss import ADPITLoss
from seld_training.data.dataset import SELDDataset
from seld_training.evaluation.seld_metrics import compute_seld_metrics

logger = structlog.get_logger(__name__)


class SELDLightningModule(L.LightningModule):
    """PyTorch Lightning module for SELD training with ADPIT loss."""

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.save_hyperparameters(cfg)

        model_cfg = cfg["model"]
        training_cfg = cfg["training"]

        self.model = ResNetConformer(
            in_channels=7,  # 4 log-mel + 3 IV
            num_classes=model_cfg["num_classes"],
            num_tracks=model_cfg["num_tracks"],
            conformer_layers=model_cfg["conformer_layers"],
            d_model=model_cfg["conformer_d_model"],
            num_heads=model_cfg["conformer_heads"],
            conv_kernel_size=model_cfg["conformer_conv_kernel"],
            use_se=model_cfg["se_block"],
        )

        self.criterion = ADPITLoss(
            num_tracks=model_cfg["num_tracks"],
            aux_lambda=training_cfg["adpit_lambda"],
        )

        self.lr = training_cfg["learning_rate"]
        self.weight_decay = training_cfg["weight_decay"]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        features, labels = batch
        predictions = self(features)

        # Align time dimensions
        min_t = min(predictions.shape[1], labels.shape[1])
        predictions = predictions[:, :min_t]
        labels = labels[:, :min_t]

        loss = self.criterion(predictions, labels)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: tuple, batch_idx: int) -> None:
        features, labels = batch
        predictions = self(features)

        min_t = min(predictions.shape[1], labels.shape[1])
        predictions = predictions[:, :min_t]
        labels = labels[:, :min_t]

        loss = self.criterion(predictions, labels)
        self.log("val/loss", loss, prog_bar=True, sync_dist=True)

        # Compute SELD metrics on first batch
        if batch_idx == 0:
            pred_np = predictions[0].detach().cpu().numpy()
            target_np = labels[0].detach().cpu().numpy()
            metrics = compute_seld_metrics(pred_np, target_np)
            for key, value in metrics.items():
                self.log(f"val/{key}", value, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


def main() -> None:
    """Main training entry point."""
    # Load config
    config_path = Path(__file__).parent / "config" / "default.yaml"
    cfg = OmegaConf.load(config_path)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    data_cfg = cfg_dict["data"]

    # Create datasets
    train_dataset = SELDDataset(
        data_dir=data_cfg["data_dir"],
        split="train",
        num_classes=cfg_dict["model"]["num_classes"],
        num_tracks=cfg_dict["model"]["num_tracks"],
    )
    val_dataset = SELDDataset(
        data_dir=data_cfg["data_dir"],
        split="val",
        num_classes=cfg_dict["model"]["num_classes"],
        num_tracks=cfg_dict["model"]["num_tracks"],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg_dict["training"]["batch_size"],
        shuffle=True,
        num_workers=data_cfg["num_workers"],
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg_dict["training"]["batch_size"],
        shuffle=False,
        num_workers=data_cfg["num_workers"],
        pin_memory=True,
    )

    # Create model
    model = SELDLightningModule(cfg_dict)

    # Callbacks
    callbacks = [
        L.pytorch.callbacks.ModelCheckpoint(
            monitor="val/seld_error",
            mode="min",
            save_top_k=3,
            filename="seld-{epoch:03d}-{val/seld_error:.4f}",
        ),
        L.pytorch.callbacks.EarlyStopping(
            monitor="val/seld_error",
            patience=20,
            mode="min",
        ),
        L.pytorch.callbacks.LearningRateMonitor(logging_interval="epoch"),
    ]

    # W&B logger
    wandb_cfg = cfg_dict.get("wandb", {})
    wandb_logger = None
    try:
        from lightning.pytorch.loggers import WandbLogger

        wandb_logger = WandbLogger(
            project=wandb_cfg.get("project", "seld-digital-twin"),
            entity=wandb_cfg.get("entity"),
            log_model=wandb_cfg.get("log_model", True),
        )
    except ImportError:
        logger.warning("wandb not available, using CSV logger")

    # Trainer
    trainer = L.Trainer(
        max_epochs=cfg_dict["training"]["max_epochs"],
        accelerator="auto",
        devices="auto",
        precision="16-mixed",
        gradient_clip_val=cfg_dict["training"]["gradient_clip_val"],
        accumulate_grad_batches=cfg_dict["training"]["accumulate_grad_batches"],
        callbacks=callbacks,
        logger=wandb_logger or True,
        log_every_n_steps=10,
    )

    trainer.fit(model, train_loader, val_loader)
    logger.info("training_complete")


if __name__ == "__main__":
    main()
