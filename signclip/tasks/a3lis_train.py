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

        # Reset freeze policy before selectively enabling trainable parts.
        for param in self.model.parameters():
            param.requires_grad = False

        # Trainable projection from visual features to token space.
        for param in self.model.video_encoder.videomlp.parameters():
            param.requires_grad = True

        # f_theta_v: train video-side Transformer blocks.
        for param in self.model.video_encoder.bert.encoder.parameters():
            param.requires_grad = True
        if hasattr(self.model.video_encoder.bert, 'pooler') and self.model.video_encoder.bert.pooler is not None:
            for param in self.model.video_encoder.bert.pooler.parameters():
                param.requires_grad = True

        # f_theta_t: train text-side Transformer blocks.
        # Keep embedding lookup frozen to match xt from frozen embeddings.
        for param in self.model.text_encoder.encoder.parameters():
            param.requires_grad = True
        if hasattr(self.model.text_encoder, 'pooler') and self.model.text_encoder.pooler is not None:
            for param in self.model.text_encoder.pooler.parameters():
                param.requires_grad = True

        # Keep learned temperature trainable as in contrastive setups.
        if hasattr(self.model, 'logit_scale'):
            self.model.logit_scale.requires_grad = True

        opt_name = getattr(config.fairseq.optimization, 'optimizer', 'adamw')
        lr = config.fairseq.optimization.lr[0] if hasattr(config.fairseq.optimization, 'lr') else 1e-4
        weight_decay = getattr(config.fairseq.optimization, 'weight_decay', 1e-2)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        assert len(trainable_params) > 0, "No trainable parameters found after freeze policy"
        if opt_name == 'adam':
            self.optimizer = optim.Adam(trainable_params, lr=lr, weight_decay=weight_decay)
        else:
            self.optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"[trainA3LIS] Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")