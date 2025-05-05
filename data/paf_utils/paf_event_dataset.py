import os
import xml.etree.ElementTree as ET
import json
import string

import torch
from torch.utils.data import Dataset

from data.paf_utils.aedat_utils import getDVSeventsDavis
from data.paf_utils.voxels import events_to_voxel
from data.utils.types import DataType
from data.genx_utils.labels import SparselyBatchedObjectLabels
from data.genx_utils.labels import ObjectLabelBase, ObjectLabels, SparselyBatchedObjectLabels


class PAFEventDetectionDataset(Dataset):
    def __init__(self,
                 train_folder: str,
                 raw_folder: str,
                 classnames_json: str,
                 num_bins: int,
                 H: int,
                 W: int,
                 sae_window_ms: float = 20.0,
                 transform=None):
        """
        Args:
            train_folder: path containing `frames/` and `labels/`
            raw_folder: path containing `*.aedat`
            classnames_json: list of class names
            num_bins, H, W: for voxel grid
            sae_window_ms: SAE frame interval (20 ms by default)
            transform: optional torch transforms on voxel
        """
        self.raw_folder = raw_folder
        self.label_dir = os.path.join(train_folder, 'labels')
        self.frame_dir = os.path.join(train_folder, 'frames')
        with open(classnames_json, 'r') as f:
            self.classnames = json.load(f)
        self.class2idx = {c: i for i, c in enumerate(self.classnames)}

        # 1) build letter→raw_key map: e.g. {'a': '1', 'b':'2', …}
        raw_files = sorted(f for f in os.listdir(raw_folder) if f.endswith('.aedat'))
        letters   = string.ascii_lowercase
        self.letter_map = {
            letters[i]: os.path.splitext(raw_files[i])[0]
            for i in range(min(len(raw_files), len(letters)))
        }

        # 2) collect all samples by scanning labels/*.xml
        self.samples = []
        for fn in os.listdir(self.label_dir):
            if not fn.endswith('.xml'):
                continue
            stem = os.path.splitext(fn)[0]  # e.g. 'd951'
            video_letter = stem[0]           # 'd'
            frame_idx    = int(stem[1:])     # 951
            if video_letter not in self.letter_map:
                # no corresponding raw
                continue
            raw_key = self.letter_map[video_letter]            # e.g. '4'
            aedat   = os.path.join(raw_folder, raw_key + '.aedat')
            xml     = os.path.join(self.label_dir, fn)
            png     = os.path.join(self.frame_dir, stem + '.png')
            # store the 20 ms window start time = frame_idx * sae_window_ms
            t0 = frame_idx * sae_window_ms * 1e3   # microseconds
            t1 = t0 + sae_window_ms * 1e3
            self.samples.append({
                'aedat': aedat,
                'xml':   xml,
                'png':   png,
                't0':    int(t0),
                't1':    int(t1),
            })

        self.num_bins = num_bins
        self.H, self.W  = H, W
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        # --- load events in [t0,t1)
        ts, xs, ys, ps = getDVSeventsDavis(
            sample['aedat'],
            numEvents=1e10,
            startTime=sample['t0']
        )
        # truncate at t1
        mask = [t < sample['t1'] for t in ts]
        xs = [x for x, m in zip(xs, mask) if m]
        ys = [y for y, m in zip(ys, mask) if m]
        ps = [p for p, m in zip(ps, mask) if m]
        ts = [t for t, m in zip(ts, mask) if m]

        voxel_np = events_to_voxel(xs, ys, ts, ps,
                                self.num_bins, self.H, self.W)

        # 2) convert to Tensor
        voxel = torch.from_numpy(voxel_np).float()

        if self.transform:
            voxel = self.transform(voxel)

        # --- parse XML for boxes & labels
        tree = ET.parse(sample['xml'])
        boxes, labels = [], []
        for obj in tree.findall('object'):
            cls  = obj.find('name').text
            bnd = obj.find('bndbox')
            xmin = float(bnd.find('xmin').text)
            ymin = float(bnd.find('ymin').text)
            xmax = float(bnd.find('xmax').text)
            ymax = float(bnd.find('ymax').text)
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class2idx[cls])

        boxes  = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        # return (voxel_grid, image, target) or just (voxel, target)
        return voxel, boxes, labels


def paf_event_collate(batch):
    # stack voxels
    voxels = torch.stack([b[0] for b in batch], dim=0)
    B, C, D, H, W = voxels.shape  # adapt dims if needed

    ev_sequence = [ voxels[:, :, d] for d in range(D) ]

    is_first = torch.ones(B, dtype=torch.bool)

    # build sparse object labels per batch
    obj_labels_list = []
    for _, boxes, labels in batch:
        if boxes.numel() == 0:
            empty_lbl = ObjectLabels(
                object_labels = torch.empty((0, len(ObjectLabelBase._str2idx)), dtype=torch.float32),
                input_size_hw = (H, W)
            )
            obj_labels_list.append(empty_lbl)
            continue
        N = boxes.shape[0]
        # create label tensor [N, 7]
        lbl = torch.zeros((N, len(ObjectLabelBase._str2idx)), dtype=torch.float32)
        lbl[:, ObjectLabelBase._str2idx['t']] = 0
        lbl[:, ObjectLabelBase._str2idx['x']] = boxes[:, 0]
        lbl[:, ObjectLabelBase._str2idx['y']] = boxes[:, 1]
        lbl[:, ObjectLabelBase._str2idx['w']] = boxes[:, 2] - boxes[:, 0]
        lbl[:, ObjectLabelBase._str2idx['h']] = boxes[:, 3] - boxes[:, 1]
        lbl[:, ObjectLabelBase._str2idx['class_id']] = labels
        lbl[:, ObjectLabelBase._str2idx['class_confidence']] = 1.0

        obj_labels = ObjectLabels(object_labels=lbl, input_size_hw=(H, W))
        obj_labels_list.append(obj_labels)

    sparse_labels = SparselyBatchedObjectLabels(obj_labels_list)
    return {
        'data': {
            DataType.EV_REPR: ev_sequence,
            DataType.OBJLABELS_SEQ: sparse_labels,
            DataType.IS_FIRST_SAMPLE: is_first,
        }
    }