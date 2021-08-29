import segmentation_models_pytorch as smp
import torch.nn as nn
import torch


class PSPNet(nn.Module):
    def __init__(self, hparams: dict):
        super().__init__()
        self.model = smp.PSPNet(
            encoder_name=hparams["encoder_name"],
            encoder_weights=hparams["encoder_weights"],
            encoder_depth=hparams["encoder_depth"],
            psp_out_channels=hparams["psp_out_channels"],
            psp_dropout=hparams["psp_dropout"],
            in_channels=hparams["in_channels"],
            activation=hparams["activations"],
            classes=hparams["classes"],
        )
        
    def forward(self, x: torch.Tensor):
        return self.model(x)

