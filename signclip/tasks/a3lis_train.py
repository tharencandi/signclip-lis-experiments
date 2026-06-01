"""
Train the A3LIS model without restoring a SignCLIP checkpoint.

This keeps the pretrained BERT backbone from the model factory, but starts the
run without loading the pretrained SignCLIP checkpoint weights. The InfoNCE
objective, dataset pipeline, and evaluation loop are the same as finetuning.
"""

from signclip.tasks.a3lis_finetune import fineTuneA3LIS


class trainA3LIS(fineTuneA3LIS):
    """A3LIS training with pretrained BERT and no SignCLIP checkpoint restore."""

    def __init__(self, config):
        super().__init__(config, checkpoint_path=None)