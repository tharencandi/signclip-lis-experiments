"""A3LIS finetuning task with DHN-NCE objective."""

import torch
import torch.nn.functional as F

from signclip.losses.dhn_nce import DHNNCELoss
from signclip.tasks.a3lis_finetune import fineTuneA3LIS


class fineTuneA3LISDHN(fineTuneA3LIS):
    """Reuse A3LIS finetune pipeline but optimize DHN-NCE in-batch."""

    def __init__(self, config, checkpoint_path=None):
        super().__init__(config, checkpoint_path=checkpoint_path)

        self.dhn_temperature = float(
            getattr(config.fairseq.optimization, "dhn_temperature", 0.07)
        )
        self.dhn_beta1 = float(getattr(config.fairseq.optimization, "dhn_beta1", 0.5))
        self.dhn_beta2 = float(getattr(config.fairseq.optimization, "dhn_beta2", 0.5))
        self.dhn_loss = DHNNCELoss(
            temperature=self.dhn_temperature,
            beta1=self.dhn_beta1,
            beta2=self.dhn_beta2,
        )
        print(
            "DHN-NCE enabled: "
            f"temperature={self.dhn_temperature}, "
            f"beta1={self.dhn_beta1}, beta2={self.dhn_beta2}"
        )

    def _batch_nce_and_sim(self, output, label_tensor):
        # Keep retrieval metrics on the same 147-class class-text bank.
        logit_scale = self._get_logit_scale()
        raw_video = F.normalize(output["pooled_video"], p=2, dim=-1)
        raw_text = F.normalize(output["pooled_text"], p=2, dim=-1)
        text_embeds_all = self.all_text_embeds.to(self.device)

        # DHN-NCE objective on in-batch pairs.
        loss = self.dhn_loss(raw_video, raw_text)

        # Global similarity for retrieval metrics.
        sim_matrix = (logit_scale * raw_video) @ text_embeds_all.t()
        return sim_matrix, loss
