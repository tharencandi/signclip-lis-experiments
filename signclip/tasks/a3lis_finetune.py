"""
Fine-tuning SignCLIP from frozen weights on the A3LIS dataset.

A3LIS: 10 signers, 147 sign classes, ~1470 videos (10 per class).
Split: 70/10/20 train/val/test by signer (signer-independent evaluation).

Loss: Global NCE over all 147 classes.
  - Text encoder is frozen → encode all 147 class texts once, cache.
  - Each training step: [N x 147] sim_matrix, CrossEntropyLoss.
  - Training and evaluation see the same 147-class space — directly
    optimises the retrieval metric instead of a 7-15 negative proxy.

Fine-tuning: videomlp (randomly init'd pose projection) + last 2 BERT layers
  + pooler of the video encoder. Text encoder frozen throughout.
"""

from tqdm import tqdm
import os
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader

from signclip.losses.nce import MMContraLoss
from signclip.tasks.retritask import RetriTask
from signclip.datasets.a3lis_dataset import A3LISDataset
from signclip.utils.pose_utils import preprocess_pose, MAX_FRAMES
from signclip.utils.metrics import compute_retrieval_metrics
from signclip.utils.samplers import BalancedBatchSampler
from signclip.utils.a3lis_paths import (
    POSES_ROOT,
    CSV_PATH,
    SPLIT_CONFIG_PATH,
    PRETOKENIZED_LABELS_PATH,
)

class CrossModalSupConLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, video_features, text_features, labels):
        # Calculate the standard similarity matrices
        logits_per_video = video_features @ text_features.t()
        logits_per_text = text_features @ video_features.t()

        # Create a mask that is 1 where labels match, and 0 everywhere else
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(video_features.device)

        # Compute log-softmax for both directions
        log_prob_video = F.log_softmax(logits_per_video, dim=1)
        log_prob_text = F.log_softmax(logits_per_text, dim=1)

        # Compute mean of log-likelihood over positive pairs
        loss_v = -(mask * log_prob_video).sum(1) / mask.sum(1)
        loss_t = -(mask * log_prob_text).sum(1) / mask.sum(1)

        # Return the average of both directions
        return (loss_v.mean() + loss_t.mean()) / 2

class VideoSupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, video_features, labels):
        device = video_features.device
        labels = labels.contiguous().view(-1, 1)
        
        mask = torch.eq(labels, labels.T).float().to(device)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(video_features.size(0)).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        
        similarity_matrix = torch.matmul(video_features, video_features.T) / self.temperature
        logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
        logits = similarity_matrix - logits_max.detach()
        
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)
        
        sum_mask = mask.sum(1)
        valid_anchors = sum_mask > 0 
        
        if not valid_anchors.any():
            return torch.tensor(0.0, device=device)
            
        loss = -(mask * log_prob).sum(1)[valid_anchors] / sum_mask[valid_anchors]
        return loss.mean()


class fineTuneA3LIS(RetriTask):

    def __init__(self, config, checkpoint_path=None):
        super().__init__(config)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.build_model()
        self.model.to(self.device)



        # Load pretrained SignCLIP weights for fine-tuning
        if checkpoint_path is not None:
            state = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            if isinstance(state, dict) and 'model' in state:
                sd = state['model']
            elif isinstance(state, dict) and 'model_state_dict' in state:
                sd = state['model_state_dict']
            else:
                sd = state

            # Strip the 'mmmodel.' prefix to match current model structure
            stripped_sd = {k.replace('mmmodel.', ''): v for k, v in sd.items()}

            missing, unexpected = self.model.load_state_dict(stripped_sd, strict=False)
            print(f"Checkpoint loaded — missing keys: {len(missing)}, unexpected: {len(unexpected)}")
            if missing:
                print(f"  Missing (not in checkpoint): {missing[:5]}")
            if unexpected:
                print(f"  Unexpected (not in model): {unexpected[:5]}")

        # Freeze text encoder entirely
        for param in self.model.text_encoder.parameters():
            param.requires_grad = False

        # For video encoder: always train videomlp (pose-specific projection, randomly init'd)
        for param in self.model.video_encoder.parameters():
            param.requires_grad = False  # freeze all first

        for param in self.model.video_encoder.videomlp.parameters():
            param.requires_grad = True   # always train — was randomly initialized

        # Unfreeze last N bert layers + pooler
        NUM_LAYERS = 2  # start here, increase if underfitting
        for layer in self.model.video_encoder.bert.encoder.layer[-NUM_LAYERS:]:
            for param in layer.parameters():
                param.requires_grad = True

        for param in self.model.video_encoder.bert.pooler.parameters():
            param.requires_grad = True

        if hasattr(self.model, 'logit_scale'):
            self.model.logit_scale.requires_grad = True
            # Use pretrained logit_scale value — do NOT override it

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
        # USED FOR EVERYTHING ELSE
        '''
        opt_name = getattr(config.fairseq.optimization, 'optimizer', 'adamw')
        lr = config.fairseq.optimization.lr[0] if hasattr(config.fairseq.optimization, 'lr') else 1e-4
        weight_decay = getattr(config.fairseq.optimization, 'weight_decay', 1e-2)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        assert len(trainable_params) > 0, "No trainable parameters found — check that video_encoder weights have requires_grad=True"
        if opt_name == 'adam':
            self.optimizer = optim.Adam(trainable_params, lr=lr, weight_decay=weight_decay)
        else:
            self.optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)

        self.nce_loss = MMContraLoss() # SignClip loss
        #self.supcon_loss = VideoSupConLoss(temperature = 0.07)
        #self.supcon_weight = 0.2

        max_text_len = getattr(config.dataset, 'max_len', 64)
        '''
        # USED FOR CROSS-ENTROPY
        '''
        opt_name = getattr(config.fairseq.optimization, 'optimizer', 'adamw')
        lr = config.fairseq.optimization.lr[0] if hasattr(config.fairseq.optimization, 'lr') else 1e-4
        weight_decay = getattr(config.fairseq.optimization, 'weight_decay', 1e-2)
        # 1. Build the Classification Head (Assuming 768 is the SignCLIP video embedding dimension)
        self.classifier_head = nn.Linear(768, 149).to(self.device)
        self.ce_loss = nn.CrossEntropyLoss()
        # 2. Gather trainable parameters and ADD the new head to them!
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        trainable_params.extend(list(self.classifier_head.parameters()))
        # 3. Initialize Optimizer
        if opt_name == 'adam':
            self.optimizer = optim.Adam(trainable_params, lr=lr, weight_decay=weight_decay)
        else:
            self.optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
        max_text_len = getattr(config.dataset, 'max_len', 64)
        '''
        # USED FOR SIGNCLIP LOSS + CROSS-ENTROPY
        '''
        # 1. The Multimodal Text-Video Loss
        self.nce_loss = MMContraLoss()
        # 2. The Classification Head & Loss (768 embedding dim, 149 classes)
        self.classifier_head = nn.Linear(768, 149).to(self.device)
        self.ce_loss = nn.CrossEntropyLoss()
        # 3. Define the blending weight (Keep NCE as the primary focus, CE as secondary)
        self.ce_weight = 0.5  
        # 4. Gather parameters and inject the new head
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        trainable_params.extend(list(self.classifier_head.parameters()))
        opt_name = getattr(config.fairseq.optimization, 'optimizer', 'adamw')
        lr = config.fairseq.optimization.lr[0] if hasattr(config.fairseq.optimization, 'lr') else 1e-4
        weight_decay = getattr(config.fairseq.optimization, 'weight_decay', 1e-2)
        if opt_name == 'adam':
            self.optimizer = optim.Adam(trainable_params, lr=lr, weight_decay=weight_decay)
        else:
            self.optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
        max_text_len = getattr(config.dataset, 'max_len', 64)
        '''

        # USED FOR GLOBAL NCE
        '''
        # Remove all external loss initializations (no MMContraLoss, no CE head)
        opt_name = getattr(config.fairseq.optimization, 'optimizer', 'adamw')
        lr = config.fairseq.optimization.lr[0] if hasattr(config.fairseq.optimization, 'lr') else 1e-4
        weight_decay = getattr(config.fairseq.optimization, 'weight_decay', 1e-2)
        # Gather only the standard model parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        if opt_name == 'adam':
            self.optimizer = optim.Adam(trainable_params, lr=lr, weight_decay=weight_decay)
        else:
            self.optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
        max_text_len = getattr(config.dataset, 'max_len', 64)
        '''

        # USED FOR GLOBAL NCE + DECOUPLED SUP-CON
        opt_name = getattr(config.fairseq.optimization, 'optimizer', 'adamw')
        lr = config.fairseq.optimization.lr[0] if hasattr(config.fairseq.optimization, 'lr') else 1e-4
        weight_decay = getattr(config.fairseq.optimization, 'weight_decay', 1e-2)
        # Gather only the standard model parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        if opt_name == 'adam':
            self.optimizer = optim.Adam(trainable_params, lr=lr, weight_decay=weight_decay)
        else:
            self.optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
        # The alpha weight for Decoupled SupCon (Default 0.5 per the paper)
        self.dcl_weight = 0.5
        max_text_len = getattr(config.dataset, 'max_len', 64)

        self.train_dataset = A3LISDataset(
            POSES_ROOT, CSV_PATH,
            split_config_path=SPLIT_CONFIG_PATH,
            split_filter='train',
            max_text_len=max_text_len,
            pretokenized_labels_path=PRETOKENIZED_LABELS_PATH
        )
        self.val_dataset = A3LISDataset(
            POSES_ROOT, CSV_PATH,
            split_config_path=SPLIT_CONFIG_PATH,
            split_filter='val',
            max_text_len=max_text_len,
            pretokenized_labels_path=PRETOKENIZED_LABELS_PATH
        )

        batch_size = getattr(config.fairseq.dataset, 'batch_size', 16)
        num_workers = getattr(config.fairseq.dataset, 'num_workers', 0)

        # BALANCED BATCH SAMPLER
        
        if len(self.train_dataset) > 0:
            # Create the sampler (P=4 classes, K=4 instances = batch size of 16)
            balanced_sampler = BalancedBatchSampler(self.train_dataset, n_classes=8, n_samples=2)
            
            self.train_data = DataLoader(
                self.train_dataset,
                # NOTE: When using batch_sampler, you MUST remove batch_size, shuffle, and drop_last!
                batch_sampler=balanced_sampler,
                collate_fn=self.train_pose_collate_fn,
                num_workers=num_workers,
            )
        else:
            self.train_data = None
        
        # STANDARD BATCHES
        '''
        if len(self.train_dataset) > 0:
            self.train_data = DataLoader(
                self.train_dataset,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=self.train_pose_collate_fn,
                num_workers=num_workers,
                drop_last=False
            )
        else:
            self.train_data = None
        '''
        

        self.val_data = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self.val_pose_collate_fn,
            num_workers=num_workers
        ) if len(self.val_dataset) > 0 else None

        # Pre-compute all class text embeddings (frozen encoder — done once).
        # These stay on CPU until move_to_device is called per-batch.
        self._all_class_caps, self._all_class_cmasks = \
            self._build_all_class_tokens(self.train_dataset, max_text_len)
        print(f"Pre-computing text embeddings for {self._all_class_caps.size(0)} classes...")
        self.all_text_embeds = self._encode_all_class_texts()  # [num_classes x D]
        print(f"  done. Shape: {self.all_text_embeds.shape}")


    # ------------------------------------------------------------------
    # Global NCE helpers
    # ------------------------------------------------------------------

    def _build_all_class_tokens(self, dataset, max_text_len):
        """Build (caps, cmasks) for every class in label_idx order.

        Uses the align_processor so the token format is identical to
        what individual samples receive at training time.
        """
        num_classes = len(dataset.meta_processor.label_map)  # 147
        align_proc = dataset.align_processor

        # idx_to_italian: reverse of label_map (italian -> idx)
        idx_to_italian = {v: k for k, v in dataset.meta_processor.label_map.items()}
        # english_map: italian_label -> [list of english labels]; use first label
        english_map = dataset.meta_processor.english_map

        all_caps   = torch.zeros(num_classes, max_text_len, dtype=torch.long)
        all_cmasks = torch.zeros(num_classes, max_text_len, dtype=torch.long)

        for idx in range(num_classes):
            italian = idx_to_italian[idx]
            english = english_map.get(italian, [italian])[0]
            # Reuse align_processor's _build_caps so format is exactly consistent
            if align_proc.pretokenized_labels and english in align_proc.pretokenized_labels:
                raw_ids = align_proc.pretokenized_labels[english]['raw_ids']
            else:
                text = f"<en> <lis> {english}"
                raw_ids = align_proc.tokenizer(text, add_special_tokens=False)["input_ids"]
            caps, cmasks = align_proc._build_caps(raw_ids)
            all_caps[idx]   = caps
            all_cmasks[idx] = cmasks

        return all_caps, all_cmasks

    @torch.no_grad()
    def _encode_all_class_texts(self):
        """Run all class (caps, cmasks) through the frozen text encoder.
        Returns normalised embeddings on CPU, shape [num_classes x D].
        """
        self.model.eval()
        caps   = self._all_class_caps.to(self.device)
        cmasks = self._all_class_cmasks.to(self.device)
        # Process in mini-batches to avoid OOM
        chunk = 64
        parts = []
        for i in range(0, caps.size(0), chunk):
            pooled = self.model.forward_text(caps[i:i+chunk], cmasks[i:i+chunk])
            parts.append(F.normalize(pooled, p=2, dim=-1).cpu())
        return torch.cat(parts, dim=0)  # [num_classes x D] on CPU

    def _get_logit_scale(self):
        """Safely retrieve the model's learned temperature scale."""
        if hasattr(self.model, 'logit_scale'):
            return self.model.logit_scale.exp()
        return 14.28  # default: 1 / 0.07

    # Original one, gemini says this:
    # I am also fixing a tiny math bug in your original code here. 
    # Your original code multiplied logit_scale into both embeddings 
    # before passing them to the loss. When the loss takes the dot product, 
    # that accidentally squares the temperature scale! 
    # The standard way is to multiply the scale into the embeddings only once.
    '''
    def _batch_nce_and_sim(self, output, label_tensor):
        """Batch-local bidirectional NCE loss (MMContraLoss) + 147-class sim_matrix.

        Uses MMContraLoss from nce.py — identical to the SignCLIP pretraining objective.
        Embeddings are L2-normalised and scaled by logit_scale before the loss,
        consistent with how the pretrained model was trained.
        Metrics sim_matrix is computed against all 147 frozen class embeddings.
        """
        logit_scale       = self._get_logit_scale()
        video_embeds      = logit_scale * F.normalize(output["pooled_video"], p=2, dim=-1)  # [N x D]
        text_embeds_batch = logit_scale * F.normalize(output["pooled_text"],  p=2, dim=-1)  # [N x D]
        text_embeds_all   = self.all_text_embeds.to(self.device)                            # [147 x D]

        loss = self.nce_loss(video_embeds, text_embeds_batch)

        # 147-class sim_matrix for retrieval metrics only (logit_scale already baked in)
        sim_matrix = video_embeds @ text_embeds_all.t()  # [N x 147]
        return sim_matrix, loss
    '''
    # For Supervised contrastive loss
    '''
    def _batch_nce_and_sim(self, output, label_tensor):
        logit_scale       = self._get_logit_scale()
        
        # 1. Normalize the embeddings (do NOT multiply logit_scale here yet)
        raw_video_embeds  = F.normalize(output["pooled_video"], p=2, dim=-1)
        raw_text_embeds   = F.normalize(output["pooled_text"],  p=2, dim=-1)
        text_embeds_all   = self.all_text_embeds.to(self.device)

        # 2. Multiply by the scale just before passing to the loss
        scaled_video = logit_scale * raw_video_embeds
        scaled_text = logit_scale * raw_text_embeds

        # 3. Calculate SupCon Loss using the labels!
        loss = self.supcon_loss(scaled_video, scaled_text, label_tensor)

        # 4. Compute 147-class sim_matrix for retrieval metrics
        sim_matrix = scaled_video @ text_embeds_all.t()
        
        return sim_matrix, loss
    '''
    # Original + SupCon as a regulizer term
    '''
    def _batch_nce_and_sim(self, output, label_tensor):
        logit_scale       = self._get_logit_scale()
        
        # 1. Base L2 normalization
        raw_video_embeds  = F.normalize(output["pooled_video"], p=2, dim=-1)
        raw_text_embeds   = F.normalize(output["pooled_text"],  p=2, dim=-1)
        text_embeds_all   = self.all_text_embeds.to(self.device)

        # 2. Scale features for the multimodal NCE loss
        scaled_video = logit_scale * raw_video_embeds
        scaled_text  = logit_scale * raw_text_embeds

        # Loss 1: Standard Text-to-Video Contrastive Loss
        loss_nce = self.nce_loss(scaled_video, scaled_text)

        # Loss 2: Video-to-Video Supervised Contrastive Regularizer 
        loss_supcon = self.supcon_loss(raw_video_embeds, label_tensor)

        # Blended Loss output
        total_loss = loss_nce + (self.supcon_weight * loss_supcon)

        # Compute 147-class similarity matrix for tracking evaluation metrics
        sim_matrix = scaled_video @ text_embeds_all.t()
        
        return sim_matrix, total_loss
    '''
    # CROSS-ENTROPY
    '''
    def _batch_nce_and_sim(self, output, label_tensor):
        # 1. Get the raw video embedding
        raw_video_embeds = output["pooled_video"] 
        
        # 2. Pass it through our new classification head to get the 147 bucket scores
        logits = self.classifier_head(raw_video_embeds)
        
        # 3. Calculate standard Cross-Entropy Loss
        loss = self.ce_loss(logits, label_tensor)
        
        # 4. Use the logits as the "similarity matrix" so the evaluation metrics still work
        sim_matrix = logits
        
        return sim_matrix, loss
    '''
    # Original signclip + cross-entropy
    '''
    def _batch_nce_and_sim(self, output, label_tensor):
        logit_scale       = self._get_logit_scale()
        
        # --- TASK 1: MULTIMODAL ALIGNMENT (NCE) ---
        # Normalize and scale the embeddings for the contrastive text mapping
        raw_video_embeds  = F.normalize(output["pooled_video"], p=2, dim=-1)
        raw_text_embeds   = F.normalize(output["pooled_text"],  p=2, dim=-1)
        text_embeds_all   = self.all_text_embeds.to(self.device)

        scaled_video = logit_scale * raw_video_embeds
        scaled_text  = logit_scale * raw_text_embeds

        loss_nce = self.nce_loss(scaled_video, scaled_text)

        # --- TASK 2: LINEAR SEPARABILITY (CE) ---
        # Pass the unnormalized video features into the classification head
        logits = self.classifier_head(output["pooled_video"])
        loss_ce = self.ce_loss(logits, label_tensor)

        # --- COMBINE ---
        total_loss = loss_nce + (self.ce_weight * loss_ce)

        # CRITICAL: Return the Multimodal sim_matrix (Video vs Text)
        # so the evaluation metrics track true Direct Retrieval accuracy!
        sim_matrix = scaled_video @ text_embeds_all.t()
        
        return sim_matrix, total_loss
    '''
    # Global NCE
    '''
    def _batch_nce_and_sim(self, output, label_tensor):
        logit_scale = self._get_logit_scale()
        
        # 1. L2 Normalize both modalities
        raw_video_embeds = F.normalize(output["pooled_video"], p=2, dim=-1)
        text_embeds_all  = self.all_text_embeds.to(self.device)

        # 2. Scale the video and compute Global Logits against ALL 149 classes
        scaled_video = logit_scale * raw_video_embeds
        global_logits = scaled_video @ text_embeds_all.t()  # Shape: [16, 149]

        # 3. Global NCE Loss
        loss_nce = F.cross_entropy(global_logits, label_tensor)
        
        # Return the global_logits as the sim_matrix so metrics are calculated perfectly
        return global_logits, loss_nce
    '''

    # Global NCE + Decoupled Sup-Con
    def _batch_nce_and_sim(self, output, label_tensor):
        logit_scale = self._get_logit_scale()
        device = output["pooled_video"].device
        
        # 1. Normalize embeddings
        raw_video = F.normalize(output["pooled_video"], p=2, dim=-1)
        text_embeds_all = self.all_text_embeds.to(device)

        # ==========================================
        # LOSS 1: GLOBAL NCE (Text-to-Video Alignment)
        # ==========================================
        scaled_video = logit_scale * raw_video
        global_logits = scaled_video @ text_embeds_all.t()
        loss_nce = F.cross_entropy(global_logits, label_tensor)

        # ==========================================
        # LOSS 2: DECOUPLED SUPCON (DCL)
        # ==========================================
        labels = label_tensor.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float()
        
        # self_mask: 1 on diagonal (the video itself)
        self_mask = torch.eye(labels.shape[0], device=device)
        
        # pos_mask (P_i): Same class, excluding self
        pos_mask = mask - self_mask
        
        # neg_mask: Different classes ONLY (The "Decoupled" magic!)
        neg_mask = 1.0 - mask
        
        # Base similarity matrix scaled by temperature (tau=0.07)
        tau = 0.07 
        sim_matrix = torch.matmul(raw_video, raw_video.T) / tau
        
        # Denominator: Sum of exp over NEGATIVES only
        exp_sim = torch.exp(sim_matrix) * neg_mask
        denominator = exp_sim.sum(dim=1, keepdim=True)
        
        # Log Prob: <v_i, v_p> - log(denominator)
        log_prob = sim_matrix - torch.log(denominator + 1e-6)
        
        # Mean over positive pairs
        sum_pos = pos_mask.sum(dim=1)
        valid_anchors = sum_pos > 0
        
        if valid_anchors.any():
            loss_dcl = -(pos_mask * log_prob).sum(dim=1)[valid_anchors] / sum_pos[valid_anchors]
            loss_dcl = loss_dcl.mean()
        else:
            loss_dcl = torch.tensor(0.0, device=device)

        # ==========================================
        # COMBINED OBJECTIVE
        # ==========================================
        total_loss = loss_nce + (self.dcl_weight * loss_dcl)

        # Return global_logits as sim_matrix for perfect metric tracking
        return global_logits, total_loss



    def _retrieval_metrics(self, sim_matrix, label_tensor):
        return compute_retrieval_metrics(sim_matrix, label_tensor)

    # ------------------------------------------------------------------
    # Training / evaluation
    # ------------------------------------------------------------------

    def train_step_with_metrics(self, skip_backprop=False):
        self.model.train()
        total_loss = 0
        all_ranks = []
        recall_at_1 = recall_at_5 = recall_at_10 = 0
        total_samples = 0


        pbar = tqdm(self.train_data, desc="Train", dynamic_ncols=True, leave=False)
        for batch in pbar:
            batch = self.move_to_device(batch)
            output = self.model(batch['caps'], batch['cmasks'], batch['pose'], batch['vmasks'])

            sim_matrix, loss = self._batch_nce_and_sim(output, batch['label'])
            total_loss += loss.item()

            if not skip_backprop:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            r1, r5, r10, medK, ranks = self._retrieval_metrics(sim_matrix, batch['label'])
            # Accumulate weighted by batch size for correct epoch-level averages
            n = sim_matrix.size(0)
            recall_at_1  += r1  * n / 100.
            recall_at_5  += r5  * n / 100.
            recall_at_10 += r10 * n / 100.
            all_ranks.extend(ranks)
            total_samples += n

            avg_loss = total_loss / (pbar.n + 1)
            pbar.set_postfix({
                "loss": f"{avg_loss:.4f}",
                "R@1": f"{r1:.1f}%",
                "R@5": f"{r5:.1f}%",
                "medK": f"{medK:.1f}"
            })

        avg_loss = total_loss / len(self.train_data)
        medianK  = float(torch.median(torch.tensor(all_ranks, dtype=torch.float)).item()) if all_ranks else 0
        r1  = 100. * recall_at_1  / total_samples if total_samples > 0 else 0
        r5  = 100. * recall_at_5  / total_samples if total_samples > 0 else 0
        r10 = 100. * recall_at_10 / total_samples if total_samples > 0 else 0
        return avg_loss, r1, r5, r10, medianK

    def eval_with_metrics(self):
        self.model.eval()
        total_loss = 0
        all_ranks = []
        recall_at_1 = recall_at_5 = recall_at_10 = 0
        total_samples = 0

        with torch.no_grad():
            pbar = tqdm(self.val_data, desc="Val", dynamic_ncols=True, leave=False)
            for batch in pbar:
                batch = self.move_to_device(batch)
                output = self.model(batch['caps'], batch['cmasks'], batch['pose'], batch['vmasks'])

                sim_matrix, loss = self._batch_nce_and_sim(output, batch['label'])
                total_loss += loss.item()

                r1, r5, r10, medK, ranks = self._retrieval_metrics(sim_matrix, batch['label'])
                n = sim_matrix.size(0)
                recall_at_1  += r1  * n / 100.
                recall_at_5  += r5  * n / 100.
                recall_at_10 += r10 * n / 100.
                all_ranks.extend(ranks)
                total_samples += n

                avg_loss = total_loss / (pbar.n + 1)
                pbar.set_postfix({
                    "loss": f"{avg_loss:.4f}",
                    "R@1": f"{r1:.1f}%",
                    "R@5": f"{r5:.1f}%",
                    "medK": f"{medK:.1f}"
                })

        avg_loss = total_loss / len(self.val_data)
        medianK  = float(torch.median(torch.tensor(all_ranks, dtype=torch.float)).item()) if all_ranks else 0
        r1  = 100. * recall_at_1  / total_samples if total_samples > 0 else 0
        r5  = 100. * recall_at_5  / total_samples if total_samples > 0 else 0
        r10 = 100. * recall_at_10 / total_samples if total_samples > 0 else 0
        return avg_loss, r1, r5, r10, medianK

    def train_loop(self, num_epochs, save_path=None):
        best_val_r1 = 0
        for epoch in range(num_epochs):
            train_loss, r1, r5, r10, medK   = self.train_step_with_metrics()
            val_loss,  vr1, vr5, vr10, vmedK = self.eval_with_metrics()
            print(
                f"Epoch {epoch+1}/{num_epochs} | "
                f"Train — Loss: {train_loss:.4f} R@1: {r1:.2f}% R@5: {r5:.2f}% medK: {medK:.1f} | "
                f"Val   — Loss: {val_loss:.4f}  R@1: {vr1:.2f}% R@5: {vr5:.2f}% medK: {vmedK:.1f}"
            )
            if save_path and vr1 > best_val_r1:
                best_val_r1 = vr1
                torch.save({'model_state_dict': self.model.state_dict()}, save_path)
                print(f"  ✓ New best model saved (Val R@1: {vr1:.2f}%)")

    # ------------------------------------------------------------------
    # DataLoader utilities
    # ------------------------------------------------------------------

    def train_pose_collate_fn(self, batch):
        return self.pose_collate_fn(batch, augment=True)

    def val_pose_collate_fn(self, batch):
        return self.pose_collate_fn(batch, augment=False)

    def pose_collate_fn(self, batch, augment=False):
        if len(batch) == 0:
            return {
                'pose':   torch.empty(0),
                'vmasks': torch.empty(0, dtype=torch.bool),
                'label':  torch.empty(0, dtype=torch.long),
                'caps':   torch.empty(0),
                'cmasks': torch.empty(0),
            }
        poses, lengths = [], []
        for item in batch:
            # Use the canonical preprocessing pipeline (shoulder normalisation,
            # face-contour filtering, leg hiding, NaN→0) so that fine-tuning
            # and inference receive the same input distribution.
            p = preprocess_pose(item['pose'], max_frames=MAX_FRAMES, augment=augment)[0]  # (T, 609)
            poses.append(p)
            lengths.append(p.shape[0])

        max_len = max(lengths)
        padded_poses = torch.stack([
            torch.cat([p, torch.zeros(max_len - p.shape[0], p.shape[1])], dim=0)
            if p.shape[0] < max_len else p
            for p in poses
        ])
        vmasks = torch.zeros(padded_poses.shape[:2], dtype=torch.bool)
        for i, l in enumerate(lengths):
            vmasks[i, :l] = True

        return {
            'pose':   padded_poses,
            'vmasks': vmasks,
            'label':  torch.tensor([item['label'] for item in batch], dtype=torch.long),
            'caps':   torch.stack([item['caps']   for item in batch]),
            'cmasks': torch.stack([item['cmasks'] for item in batch]),
        }

    def move_to_device(self, batch):
        return {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}