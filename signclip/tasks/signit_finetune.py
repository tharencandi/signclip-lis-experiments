"""Fine-tuning SignCLIP on the SignIT dataset with Global NCE only.

Training regime summary:
- Class imbalance is handled with a WeightedRandomSampler using per-sample
    weights proportional to 1 / class_count(label), so rare classes are sampled
    more often and class exposure is balanced in expectation across training.
- Data augmentation strength is class-aware and inversely proportional to class
    count. Rarer classes receive stronger temporal/spatial/noise perturbations
    (and higher flip probability) to reduce overfitting when the same few
    examples are reused.
"""

from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, WeightedRandomSampler

from signclip.tasks.retritask import RetriTask
from signclip.datasets.signit_dataset import SignITDataset
from signclip.utils.pose_utils import preprocess_pose, MAX_FRAMES
from signclip.utils.metrics import compute_retrieval_metrics
from signclip.utils.signit_paths import POSES_ROOT


class fineTuneSignIT(RetriTask):

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

        for param in self.model.text_encoder.parameters():
            param.requires_grad = False

        for param in self.model.video_encoder.parameters():
            param.requires_grad = False

        for param in self.model.video_encoder.videomlp.parameters():
            param.requires_grad = True

        num_layers = 2
        for layer in self.model.video_encoder.bert.encoder.layer[-num_layers:]:
            for param in layer.parameters():
                param.requires_grad = True

        for param in self.model.video_encoder.bert.pooler.parameters():
            param.requires_grad = True

        if hasattr(self.model, 'logit_scale'):
            self.model.logit_scale.requires_grad = True

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"Trainable parameters: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

        opt_name = getattr(config.fairseq.optimization, 'optimizer', 'adamw')
        lr = config.fairseq.optimization.lr[0] if hasattr(config.fairseq.optimization, 'lr') else 1e-4
        weight_decay = getattr(config.fairseq.optimization, 'weight_decay', 1e-2)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        if opt_name == 'adam':
            self.optimizer = optim.Adam(trainable_params, lr=lr, weight_decay=weight_decay)
        else:
            self.optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)

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

        # Build class-frequency stats used for both weighted sampling and
        # augmentation intensity scaling (rarer classes get stronger aug).
        self.label_counts = self._compute_label_counts(self.train_dataset)
        self.mean_label_count = float(sum(self.label_counts.values())) / float(len(self.label_counts))

        sample_weights = self._compute_sample_weights(self.train_dataset, self.label_counts)
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )

        min_count = min(self.label_counts.values())
        max_count = max(self.label_counts.values())
        print(
            "Train class count stats: "
            f"classes={len(self.label_counts)}, min={min_count}, max={max_count}, "
            f"mean={self.mean_label_count:.2f}"
        )

        batch_size = getattr(config.fairseq.dataset, 'batch_size', 16)
        num_workers = getattr(config.fairseq.dataset, 'num_workers', 0)

        self.aug_sigma_temporal = getattr(config.dataset, 'aug_sigma_temporal', 0.2)
        self.aug_sigma_spatial = getattr(config.dataset, 'aug_sigma_spatial', 0.2)
        self.aug_sigma_noise = getattr(config.dataset, 'aug_sigma_noise', 0.001)
        self.aug_p_flip = getattr(config.dataset, 'aug_p_flip', 0.2)
        self.aug_strength_max = getattr(config.dataset, 'aug_strength_max', 3.0)

        self.train_data = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
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

    @staticmethod
    def _compute_label_counts(dataset):
        counts = {}
        for sample in dataset.meta_processor.samples:
            label_idx = dataset.meta_processor.label_map[sample['label']]
            counts[label_idx] = counts.get(label_idx, 0) + 1
        return counts

    @staticmethod
    def _compute_sample_weights(dataset, label_counts):
        weights = []
        for sample in dataset.meta_processor.samples:
            label_idx = dataset.meta_processor.label_map[sample['label']]
            weights.append(1.0 / float(label_counts[label_idx]))
        return torch.tensor(weights, dtype=torch.double)

    def _augmentation_kwargs_for_label(self, label_idx):
        count = max(1, int(self.label_counts.get(int(label_idx), 1)))
        # Rare classes (small count) receive stronger augmentation.
        strength = self.mean_label_count / float(count)
        strength = min(max(strength, 1.0), float(self.aug_strength_max))
        return {
            'sigma_temporal': float(self.aug_sigma_temporal) * strength,
            'sigma_spatial': float(self.aug_sigma_spatial) * strength,
            'sigma_noise': float(self.aug_sigma_noise) * strength,
            'p_flip': min(float(self.aug_p_flip) * strength, 0.9),
        }

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

    def _batch_nce_and_sim(self, output, label_tensor):
        logit_scale = self._get_logit_scale()
        raw_video_embeds = F.normalize(output['pooled_video'], p=2, dim=-1)
        text_embeds_all = self.all_text_embeds.to(self.device)
        scaled_video = logit_scale * raw_video_embeds
        global_logits = scaled_video @ text_embeds_all.t()
        loss_nce = F.cross_entropy(global_logits, label_tensor)
        return global_logits, loss_nce

    def _retrieval_metrics(self, sim_matrix, label_tensor):
        return compute_retrieval_metrics(sim_matrix, label_tensor)

    def train_step_with_metrics(self, skip_backprop=False):
        self.model.train()
        total_loss = 0.0
        all_ranks = []
        recall_at_1 = recall_at_5 = recall_at_10 = 0.0
        total_samples = 0

        pbar = tqdm(self.train_data, desc='Train', dynamic_ncols=True, leave=False)
        for batch in pbar:
            batch = self.move_to_device(batch)
            output = self.model(batch['caps'], batch['cmasks'], batch['pose'], batch['vmasks'])

            sim_matrix, loss = self._batch_nce_and_sim(output, batch['label'])
            total_loss += float(loss.item())

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
            pbar.set_postfix({
                'loss': f'{avg_loss:.4f}',
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

                sim_matrix, loss = self._batch_nce_and_sim(output, batch['label'])
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
            aug_kwargs = self._augmentation_kwargs_for_label(item['label']) if augment else {}
            pose = preprocess_pose(
                item['pose'],
                max_frames=MAX_FRAMES,
                augment=augment,
                **aug_kwargs,
            )[0]
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
