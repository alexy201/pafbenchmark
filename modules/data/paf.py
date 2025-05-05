import pytorch_lightning as pl
from torch.utils.data import DataLoader
from data.paf_utils.paf_event_dataset import PAFEventDetectionDataset, paf_event_collate

class PAFEventDataModule(pl.LightningDataModule):
    def __init__(self, train_folder, raw_folder, classnames_json,
                 num_bins, H, W, train_bs, val_bs, num_workers):
        super().__init__()
        self.train_folder = train_folder
        self.raw_folder = raw_folder
        self.classnames = classnames_json
        self.num_bins, self.H, self.W = num_bins, H, W
        self.train_bs, self.val_bs = train_bs, val_bs
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.train_ds = PAFEventDetectionDataset(
            self.train_folder, self.raw_folder, self.classnames,
            self.num_bins, self.H, self.W
        )
        self.val_ds = self.train_ds  # or split

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.train_bs,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=paf_event_collate,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.val_bs,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=paf_event_collate,
            pin_memory=True
        )
