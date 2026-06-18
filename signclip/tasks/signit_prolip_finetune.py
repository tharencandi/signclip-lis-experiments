"""
ProLIP-style fine-tuning for SignIT SignCLIP.

Key idea:
- Freeze the text encoder and the full visual backbone.
- Unfreeze only the final visual projector (VideoTokenMLP.linear2).
- Train with global CE against frozen class text embeddings.
- Add squared-distance regularization to keep projector close to pretrained weights:

    loss = CE(sim(video, text_all), label) + lambda * ||W - W0||_F^2
"""

from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader

from signclip.tasks.retritask import RetriTask
from signclip.datasets.signit_dataset import SignITDataset
from signclip.utils.pose_utils import preprocess_pose, MAX_FRAMES
from signclip.utils.metrics import compute_retrieval_metrics
from signclip.utils.signit_paths import POSES_ROOT


class fineTuneSignITProLIP(RetriTask):

    def __init__(self, config, checkpoint_path=None):
        super().__init__(config)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.build_model()
        self.model.to(self.device)

        if checkpoint_path is not None:
            state = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            if isinstance(state, dict) and 'model' in state:
                sd = state['model']
            elif isinstance(state, dict) and 'model_state_dict' in state:
                sd = state['model_state_dict']
            else:
                sd = state

            stripped_sd = {k.replace('mmmodel.', ''): v for k, v in sd.items()}
            missing, unexpected = self.model.load_state_dict(stripped_sd, strict=False)
            print(f"Checkpoint loaded - missing keys: {len(missing)}, unexpected: {len(unexpected)}")
            if missing:
                print(f"  Missing (not in checkpoint): {missing[:5]}")
            if unexpected:
                print(f"  Unexpected (not in model): {unexpected[:5]}")

        for param in self.model.parameters():
            param.requires_grad = False

        if not hasattr(self.model, 'video_encoder') or not hasattr(self.model.video_encoder, 'videomlp'):
            raise AttributeError("Model does not expose video_encoder.videomlp needed for ProLIP")
        if not hasattr(self.model.video_encoder.videomlp, 'linear2'):
            raise AttributeError("video_encoder.videomlp has no linear2 layer for ProLIP")

        self.projector: nn.Module = self.model.video_encoder.videomlp.linear2
        for param in self.projector.parameters():
            param.requires_grad = True

        if hasattr(self.model, 'logit_scale'):
            self.model.logit_scale.requires_grad = True

        self._projector_init = {
            name: p.detach().cpu().clone()
            for name, p in self.projector.named_parameters()
        }

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

        opt_name = getattr(config.fairseq.optimization, 'optimizer', 'adamw')
        lr = config.fairseq.optimization.lr[0] if hasattr(config.fairseq.optimization, 'lr') else 1e-4
        weight_decay = getattr(config.fairseq.optimization, 'weight_decay', 1e-2)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        assert len(trainable_params) > 0, "No trainable parameters found for ProLIP"
        if opt_name == 'adam':
            self.optimizer = optim.Adam(trainable_params, lr=lr, weight_decay=weight_decay)
        else:
            self.optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)

        self.prolip_lambda = float(getattr(config.fairseq.optimization, 'prolip_lambda', 1.0))
        self.prolip_lambda_mode = getattr(config.fairseq.optimization, 'prolip_lambda_mode', 'inv_n')

        max_text_len = getattr(config.dataset, 'max_len', 64)
        self.train_dataset = SignITDataset(
            POSES_ROOT,
            split_filter='train',
            max_text_len=max_text_len,
        )
        self.val_dataset = SignITDataset(
            POSES_ROOT,
            split_filter='val',
            max_text_len=max_text_len,
        )

        if len(self.train_dataset) == 0:
            raise RuntimeError(
                f"SignIT train dataset is empty. Check POSES_ROOT={POSES_ROOT} and split labels embedded in filenames."
            )
        if len(self.val_dataset) == 0:
            raise RuntimeError(
                f"SignIT val dataset is empty. Check POSES_ROOT={POSES_ROOT} and split labels embedded in filenames."
            )

        batch_size = getattr(config.fairseq.dataset, 'batch_size', 16)
        num_workers = getattr(config.fairseq.dataset, 'num_workers', 0)

        self.train_data = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=self.train_pose_collate_fn,
            num_workers=num_workers,
            drop_last=False,
        )
        self.val_data = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self.val_pose_collate_fn,
            num_workers=num_workers,
        )

        self._all_class_caps, self._all_class_cmasks = self._build_all_class_tokens(self.train_dataset, max_text_len)
        print(f"Pre-computing text embeddings for {self._all_class_caps.size(0)} classes...")
        self.all_text_embeds = self._encode_all_class_texts()
        print(f"  done. Shape: {self.all_text_embeds.shape}")

        self.num_classes = self.all_text_embeds.size(0)
        self._lambda_eff = self._compute_effective_lambda()
        print(
            f"ProLIP regularization: lambda_base={self.prolip_lambda:.6g}, "
            f"mode={self.prolip_lambda_mode}, lambda_eff={self._lambda_eff:.6g}"
        )

    def _compute_effective_lambda(self) -> float:
        if len(self.train_dataset) == 0:
            return self.prolip_lambda
        shots = max(1.0, float(len(self.train_dataset)) / float(max(1, self.num_classes)))
        mode = str(self.prolip_lambda_mode).lower()
        if mode == 'inv_n2':
            return self.prolip_lambda / (shots * shots)
        if mode == 'constant':
            return self.prolip_lambda
        return self.prolip_lambda / shots

    def _build_all_class_tokens(self, dataset, max_text_len):
        num_classes = len(dataset.meta_processor.label_map)
        align_proc = dataset.align_processor

        idx_to_italian = {v: k for k, v in dataset.meta_processor.label_map.items()}
        english_map = dataset.meta_processor.english_map

        all_caps = torch.zeros(num_classes, max_text_len, dtype=torch.long)
        all_cmasks = torch.zeros(num_classes, max_text_len, dtype=torch.long)

        for idx in range(num_classes):
            italian = idx_to_italian[idx]
            english = english_map.get(italian, [italian])[0]
            text = f"<en> <lis> {english}"
            raw_ids = align_proc.tokenizer(text, add_special_tokens=False)["input_ids"]
            caps, cmasks = align_proc._build_caps(raw_ids)
            all_caps[idx] = caps
            all_cmasks[idx] = cmasks

        return all_caps, all_cmasks

    @torch.no_grad()
    def _encode_all_class_texts(self):
        self.model.eval()
        caps = self._all_class_caps.to(self.device)
        cmasks = self._all_class_cmasks.to(self.device)
        chunk = 64
        parts = []
        for i in range(0, caps.size(0), chunk):
            pooled = self.model.forward_text(caps[i:i + chunk], cmasks[i:i + chunk])
            parts.append(F.normalize(pooled, p=2, dim=-1).cpu())
        return torch.cat(parts, dim=0)

    def _get_logit_scale(self):
        if hasattr(self.model, 'logit_scale'):
            return self.model.logit_scale.exp()
        return 14.28

    def _prolip_regularizer(self):
        reg = next(self.projector.parameters()).new_zeros(())
        for name, param in self.projector.named_parameters():
            ref = self._projector_init[name].to(param.device)
            reg = reg + (param - ref).pow(2).sum()
        return reg

    def _batch_loss_and_sim(self, output, label_tensor):
        logit_scale = self._get_logit_scale()
        video_embeds = F.normalize(output['pooled_video'], p=2, dim=-1)
        text_embeds_all = self.all_text_embeds.to(self.device)

        sim_matrix = logit_scale * (video_embeds @ text_embeds_all.t())

        ce_loss = F.cross_entropy(sim_matrix, label_tensor)
        reg_loss = self._prolip_regularizer()
        loss = ce_loss + (self._lambda_eff * reg_loss)
        return sim_matrix, loss, ce_loss.detach(), reg_loss.detach()

    def _retrieval_metrics(self, sim_matrix, label_tensor):
        return compute_retrieval_metrics(sim_matrix, label_tensor)

    def train_step_with_metrics(self, skip_backprop=False):
        self.model.train()
        total_loss = 0.0
        total_ce = 0.0
        total_reg = 0.0
        all_ranks = []
        recall_at_1 = recall_at_5 = recall_at_10 = 0.0
        total_samples = 0

        pbar = tqdm(self.train_data, desc='Train', dynamic_ncols=True, leave=False)
        for batch in pbar:
            batch = self.move_to_device(batch)
            output = self.model(batch['caps'], batch['cmasks'], batch['pose'], batch['vmasks'])

            sim_matrix, loss, ce_loss, reg_loss = self._batch_loss_and_sim(output, batch['label'])
            total_loss += float(loss.item())
            total_ce += float(ce_loss.item())
            total_reg += float(reg_loss.item())

            if not skip_backprop:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            r1, r5, r10, medK, ranks = self._retrieval_metrics(sim_matrix, batch['label'])
            n = sim_matrix.size(0)
            recall_at_1 += r1 * n / 100.0
            recall_at_5 += r5 * n / 100.0
            recall_at_10 += r10 * n / 100.0
            all_ranks.extend(ranks)
            total_samples += n

            avg_loss = total_loss / (pbar.n + 1)
            avg_ce = total_ce / (pbar.n + 1)
            avg_reg = total_reg / (pbar.n + 1)
            pbar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'ce': f'{avg_ce:.4f}',
                'reg': f'{avg_reg:.2e}',
                'R@1': f'{r1:.1f}%',
                'R@5': f'{r5:.1f}%',
                'medK': f'{medK:.1f}',
            })

        avg_loss = total_loss / len(self.train_data)
        medianK = float(torch.median(torch.tensor(all_ranks, dtype=torch.float)).item()) if all_ranks else 0
        r1 = 100.0 * recall_at_1 / total_samples if total_samples > 0 else 0
        r5 = 100.0 * recall_at_5 / total_samples if total_samples > 0 else 0
        r10 = 100.0 * recall_at_10 / total_samples if total_samples > 0 else 0
        return avg_loss, r1, r5, r10, medianK

    def eval_with_metrics(self):
        self.model.eval()
        total_loss = 0.0
        all_ranks = []
        recall_at_1 = recall_at_5 = recall_at_10 = 0.0
        total_samples = 0

        with torch.no_grad():
            pbar = tqdm(self.val_data, desc='Val', dynamic_ncols=True, leave=False)
            for batch in pbar:
                batch = self.move_to_device(batch)
                output = self.model(batch['caps'], batch['cmasks'], batch['pose'], batch['vmasks'])

                sim_matrix, loss, _, _ = self._batch_loss_and_sim(output, batch['label'])
                total_loss += float(loss.item())

                r1, r5, r10, medK, ranks = self._retrieval_metrics(sim_matrix, batch['label'])
                n = sim_matrix.size(0)
                recall_at_1 += r1 * n / 100.0
                recall_at_5 += r5 * n / 100.0
                recall_at_10 += r10 * n / 100.0
                all_ranks.extend(ranks)
                total_samples += n

                avg_loss = total_loss / (pbar.n + 1)
                pbar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'R@1': f'{r1:.1f}%',
                    'R@5': f'{r5:.1f}%',
                    'medK': f'{medK:.1f}',
                })

        avg_loss = total_loss / len(self.val_data)
        medianK = float(torch.median(torch.tensor(all_ranks, dtype=torch.float)).item()) if all_ranks else 0
        r1 = 100.0 * recall_at_1 / total_samples if total_samples > 0 else 0
        r5 = 100.0 * recall_at_5 / total_samples if total_samples > 0 else 0
        r10 = 100.0 * recall_at_10 / total_samples if total_samples > 0 else 0
        return avg_loss, r1, r5, r10, medianK

    def train_pose_collate_fn(self, batch):
        return self.pose_collate_fn(batch, augment=True)

    def val_pose_collate_fn(self, batch):
        return self.pose_collate_fn(batch, augment=False)

    def pose_collate_fn(self, batch, augment=False):
        if len(batch) == 0:
            return {
                'pose': torch.empty(0),
                'vmasks': torch.empty(0, dtype=torch.bool),
                'label': torch.empty(0, dtype=torch.long),
                'caps': torch.empty(0),
                'cmasks': torch.empty(0),
            }

        poses = []
        lengths = []
        for item in batch:
            pose = preprocess_pose(item['pose'], max_frames=MAX_FRAMES, augment=augment)[0]
            poses.append(pose)
            lengths.append(pose.shape[0])

        max_len = max(lengths)
        padded_poses = torch.stack([
            torch.cat([pose, torch.zeros(max_len - pose.shape[0], pose.shape[1])], dim=0)
            if pose.shape[0] < max_len else pose
            for pose in poses
        ])
        vmasks = torch.zeros(padded_poses.shape[:2], dtype=torch.bool)
        for i, length in enumerate(lengths):
            vmasks[i, :length] = True

        return {
            'pose': padded_poses,
            'vmasks': vmasks,
            'label': torch.tensor([item['label'] for item in batch], dtype=torch.long),
            'caps': torch.stack([item['caps'] for item in batch]),
            'cmasks': torch.stack([item['cmasks'] for item in batch]),
        }

    def move_to_device(self, batch):
        return {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}