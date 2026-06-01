"""
Train the A3LIS model without restoring a SignCLIP checkpoint.

This keeps the pretrained BERT backbone from the model factory, but starts the
run without loading the pretrained SignCLIP checkpoint weights. The InfoNCE
objective, dataset pipeline, and evaluation loop are the same as finetuning.
"""

from torch import optim

from signclip.tasks.a3lis_finetune import fineTuneA3LIS


class trainA3LIS(fineTuneA3LIS):
    """A3LIS training with pretrained BERT and no SignCLIP checkpoint restore."""

    def __init__(self, config):
        super().__init__(config, checkpoint_path=None)

        # Override finetune freezing: scratch training updates all model params.
        for param in self.model.parameters():
            param.requires_grad = True

        opt_name = getattr(config.fairseq.optimization, 'optimizer', 'adamw')
        lr = config.fairseq.optimization.lr[0] if hasattr(config.fairseq.optimization, 'lr') else 1e-4
        weight_decay = getattr(config.fairseq.optimization, 'weight_decay', 1e-2)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        if opt_name == 'adam':
            self.optimizer = optim.Adam(trainable_params, lr=lr, weight_decay=weight_decay)
        else:
            self.optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"[trainA3LIS] Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    def _refresh_all_text_embeds(self):
        # Keep retrieval metrics aligned with current text encoder weights.
        self.all_text_embeds = self._encode_all_class_texts()

    def train_step_with_metrics(self, skip_backprop=False):
        self._refresh_all_text_embeds()
        return super().train_step_with_metrics(skip_backprop=skip_backprop)

    def eval_with_metrics(self):
        self._refresh_all_text_embeds()
        return super().eval_with_metrics()