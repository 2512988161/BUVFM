"""Video dataset for VideoVAEPlus pretraining.

Reuses the video loading and augmentation from mae_dataset.py but without
masking — VAE needs full video frames for reconstruction.
"""

import os
import random
import numpy as np
import torch
from PIL import Image
from torchvision import transforms


# ---------------------------------------------------------------------------
# Video loader (decord)
# ---------------------------------------------------------------------------

def get_video_loader():
    from decord import VideoReader, cpu

    def _loader(video_path):
        return VideoReader(video_path, num_threads=1, ctx=cpu(0))

    return _loader


# ---------------------------------------------------------------------------
# Augmentation transforms
# ---------------------------------------------------------------------------

class GroupMultiScaleCrop:
    """Multi-scale crop with fixed-position crop offsets.
    Scales: [1, 0.875, 0.75, 0.66] of the shorter side.
    """

    def __init__(self, input_size, scales=None):
        self.scales = scales if scales is not None else [1, 0.875, 0.75, 0.66]
        self.input_size = input_size if not isinstance(input_size, int) else [input_size, input_size]
        self.interpolation = Image.BILINEAR

    def __call__(self, img_tuple):
        img_group, label = img_tuple
        im_size = img_group[0].size
        crop_w, crop_h, offset_w, offset_h = self._sample_crop_size(im_size)
        crop_img_group = [
            img.crop((offset_w, offset_h, offset_w + crop_w, offset_h + crop_h))
            for img in img_group
        ]
        ret_img_group = [
            img.resize((self.input_size[0], self.input_size[1]), self.interpolation)
            for img in crop_img_group
        ]
        return (ret_img_group, label)

    def _sample_crop_size(self, im_size):
        image_w, image_h = im_size[0], im_size[1]
        base_size = min(image_w, image_h)
        crop_sizes = [int(base_size * x) for x in self.scales]
        crop_h = [self.input_size[1] if abs(x - self.input_size[1]) < 3 else x for x in crop_sizes]
        crop_w = [self.input_size[0] if abs(x - self.input_size[0]) < 3 else x for x in crop_sizes]

        pairs = []
        for i, h in enumerate(crop_h):
            for j, w in enumerate(crop_w):
                if abs(i - j) <= 1:
                    pairs.append((w, h))

        crop_pair = random.choice(pairs)
        w_offset, h_offset = self._sample_fix_offset(image_w, image_h, crop_pair[0], crop_pair[1])
        return crop_pair[0], crop_pair[1], w_offset, h_offset

    @staticmethod
    def _sample_fix_offset(image_w, image_h, crop_w, crop_h):
        w_step = (image_w - crop_w) // 4
        h_step = (image_h - crop_h) // 4
        offsets = [
            (0, 0), (4 * w_step, 0), (0, 4 * h_step), (4 * w_step, 4 * h_step),
            (2 * w_step, 2 * h_step),
            (0, 2 * h_step), (4 * w_step, 2 * h_step),
            (2 * w_step, 4 * h_step), (2 * w_step, 0),
            (1 * w_step, 1 * h_step), (3 * w_step, 1 * h_step),
            (1 * w_step, 3 * h_step), (3 * w_step, 3 * h_step),
        ]
        return random.choice(offsets)


class Stack:
    def __init__(self, roll=False):
        self.roll = roll

    def __call__(self, img_tuple):
        img_group, label = img_tuple
        if img_group[0].mode == 'L':
            return (np.concatenate([np.expand_dims(x, 2) for x in img_group], axis=2), label)
        elif img_group[0].mode == 'RGB':
            if self.roll:
                return (np.concatenate([np.array(x)[:, :, ::-1] for x in img_group], axis=2), label)
            else:
                return (np.concatenate(img_group, axis=2), label)


class ToTorchFormatTensor:
    def __init__(self, div=True):
        self.div = div

    def __call__(self, pic_tuple):
        pic, label = pic_tuple
        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic).permute(2, 0, 1).contiguous()
        else:
            img = torch.as_tensor(pic.tobytes(), dtype=torch.uint8)
            img = img.view(pic.size[1], pic.size[0], len(pic.mode))
            img = img.transpose(0, 1).transpose(0, 2).contiguous()
        return (img.float().div(255.) if self.div else img.float(), label)


class GroupNormalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor_tuple):
        tensor, label = tensor_tuple
        rep_mean = self.mean * (tensor.size()[0] // len(self.mean))
        rep_std = self.std * (tensor.size()[0] // len(self.std))
        for t, m, s in zip(tensor, rep_mean, rep_std):
            t.sub_(m).div_(s)
        return (tensor, label)


# ---------------------------------------------------------------------------
# Data augmentation wrapper (no masking)
# ---------------------------------------------------------------------------

class DataAugmentationForVideoVAE:
    """Multi-scale crop + normalize for VAE pretraining (no masking needed)."""

    def __init__(self, input_size=224, num_frames=16, mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225)):
        self.input_mean = list(mean)
        self.input_std = list(std)
        self.num_frames = num_frames

        normalize = GroupNormalize(self.input_mean, self.input_std)
        self.transform = transforms.Compose([
            GroupMultiScaleCrop(input_size),
            Stack(roll=False),
            ToTorchFormatTensor(div=True),
            normalize,
        ])

    def __call__(self, images):
        process_data, _ = self.transform(images)
        # T*C, H, W -> T, C, H, W -> C, T, H, W
        process_data = process_data.view(
            (self.num_frames, 3) + process_data.size()[-2:]
        ).transpose(0, 1)
        return process_data

    def __repr__(self):
        return f"DataAugmentationForVideoVAE(transform={self.transform})"


# ---------------------------------------------------------------------------
# Video dataset for VAE pretraining
# ---------------------------------------------------------------------------

class VideoVAEPretrainDataset(torch.utils.data.Dataset):
    """Video dataset for VAE pretraining — .mp4 files via decord."""

    def __init__(self, root, setting, new_length=16, new_step=4, transform=None,
                 num_segments=1, temporal_jitter=False):
        super().__init__()
        self.root = root
        self.setting = setting
        self.new_length = new_length
        self.new_step = new_step
        self.skip_length = self.new_length * self.new_step
        self.transform = transform
        self.num_segments = num_segments
        self.temporal_jitter = temporal_jitter

        self.video_loader = get_video_loader()
        self.clips = self._make_dataset(root, setting)
        if len(self.clips) == 0:
            raise RuntimeError(f"Found 0 video clips in: {root}")

    def _make_dataset(self, root, setting):
        if not os.path.exists(setting):
            raise RuntimeError(f"Setting file {setting} does not exist.")
        clips = []
        with open(setting) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.rsplit(" ", 2)
                if len(parts) < 3:
                    continue
                clip_path = os.path.join(root, parts[0])
                start_idx = int(parts[1])
                total_frame = int(parts[2])
                clips.append((clip_path, start_idx, total_frame))
        return clips

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, index):
        try:
            video_path, start_idx, total_frame = self.clips[index]
            decord_vr = self.video_loader(video_path)
            duration = len(decord_vr)

            segment_indices, skip_offsets = self._sample_train_indices(duration)
            frame_id_list = self.get_frame_id_list(duration, segment_indices, skip_offsets)
            video_data = decord_vr.get_batch(frame_id_list).asnumpy()
            images = [
                Image.fromarray(video_data[vid, :, :, :]).convert("RGB")
                for vid in range(len(frame_id_list))
            ]
        except Exception as e:
            print(f"Failed to load video from {video_path}: {e}")
            index = random.randint(0, len(self.clips) - 1)
            return self.__getitem__(index)

        if self.transform is not None:
            process_data = self.transform((images, None))
        else:
            process_data = torch.from_numpy(np.stack(images)).permute(3, 0, 1, 2).float() / 255.
        return process_data

    def _sample_train_indices(self, num_frames):
        average_duration = (num_frames - self.skip_length + 1) // self.num_segments
        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)), average_duration)
            offsets = offsets + np.random.randint(average_duration, size=self.num_segments)
        elif num_frames > max(self.num_segments, self.skip_length):
            offsets = np.sort(np.random.randint(num_frames - self.skip_length + 1, size=self.num_segments))
        else:
            offsets = np.zeros((self.num_segments,))

        if self.temporal_jitter:
            skip_offsets = np.random.randint(self.new_step, size=self.skip_length // self.new_step)
        else:
            skip_offsets = np.zeros(self.skip_length // self.new_step, dtype=int)
        return offsets + 1, skip_offsets

    def get_frame_id_list(self, duration, indices, skip_offsets):
        frame_id_list = []
        for seg_ind in indices:
            offset = int(seg_ind)
            for i, _ in enumerate(range(0, self.skip_length, self.new_step)):
                if offset + skip_offsets[i] <= duration:
                    frame_id = offset + skip_offsets[i] - 1
                else:
                    frame_id = offset - 1
                frame_id_list.append(frame_id)
                if offset + self.new_step < duration:
                    offset += self.new_step
        return frame_id_list


def build_vae_pretraining_dataset(args):
    """Build pretraining dataset from parsed arguments."""
    transform = DataAugmentationForVideoVAE(
        input_size=args.input_size,
        num_frames=args.num_frames,
    )
    dataset = VideoVAEPretrainDataset(
        root=args.data_root,
        setting=args.data_path,
        new_length=args.num_frames,
        new_step=args.sampling_rate,
        transform=transform,
        num_segments=args.num_segments,
        temporal_jitter=args.temporal_jitter,
    )
    print("Data Aug = %s" % str(transform))
    return dataset
