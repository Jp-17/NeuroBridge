"""InfoNCE contrastive loss for neural-CLIP alignment.

Implements symmetric InfoNCE loss between neural embeddings and CLIP embeddings,
following the approach used in MindEye and BrainDiffuser.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """Symmetric InfoNCE contrastive loss.

    Computes bidirectional contrastive loss between neural and CLIP embeddings:
    L = 0.5 * (L_neural_to_clip + L_clip_to_neural)

    Args:
        temperature: Temperature scaling for logits (default: 0.07)
        learnable_temperature: Whether to learn temperature (default: True)
    """

    def __init__(
        self,
        temperature: float = 0.07,
        learnable_temperature: bool = True,
    ):
        super().__init__()

        if learnable_temperature:
            self.log_temperature = nn.Parameter(
                torch.tensor(temperature).log()
            )
        else:
            self.register_buffer(
                "log_temperature", torch.tensor(temperature).log()
            )

    @property
    def temperature(self):
        return self.log_temperature.exp()

    def forward(
        self,
        neural_embeds: torch.Tensor,
        clip_embeds: torch.Tensor,
    ) -> dict:
        """Compute symmetric InfoNCE loss.

        Args:
            neural_embeds: L2-normalized neural embeddings (batch, dim)
            clip_embeds: L2-normalized CLIP embeddings (batch, dim)

        Returns:
            Dict with 'loss', 'loss_n2c', 'loss_c2n', 'accuracy'
        """
        batch_size = neural_embeds.shape[0]

        # Cosine similarity matrix
        logits = (neural_embeds @ clip_embeds.T) / self.temperature
        # logits shape: (batch, batch)

        # Labels: positive pairs are on the diagonal
        labels = torch.arange(batch_size, device=logits.device)

        # Bidirectional cross-entropy
        loss_n2c = F.cross_entropy(logits, labels)      # neural → clip
        loss_c2n = F.cross_entropy(logits.T, labels)     # clip → neural
        loss = 0.5 * (loss_n2c + loss_c2n)

        # Accuracy (top-1 retrieval)
        with torch.no_grad():
            preds_n2c = logits.argmax(dim=1)
            preds_c2n = logits.argmax(dim=0)
            acc_n2c = (preds_n2c == labels).float().mean()
            acc_c2n = (preds_c2n == labels).float().mean()
            accuracy = 0.5 * (acc_n2c + acc_c2n)

            # Top-5 retrieval accuracy
            _, topk_n2c = logits.topk(min(5, batch_size), dim=1)
            _, topk_c2n = logits.T.topk(min(5, batch_size), dim=1)
            top5_n2c = (topk_n2c == labels.unsqueeze(1)).any(dim=1).float().mean()
            top5_c2n = (topk_c2n == labels.unsqueeze(1)).any(dim=1).float().mean()
            top5_accuracy = 0.5 * (top5_n2c + top5_c2n)

        return {
            "loss": loss,
            "loss_n2c": loss_n2c.item(),
            "loss_c2n": loss_c2n.item(),
            "accuracy": accuracy.item(),
            "top5_accuracy": top5_accuracy.item(),
            "temperature": self.temperature.item(),
        }
