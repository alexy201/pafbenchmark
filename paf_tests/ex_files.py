import os, sys
# adjust this to point to your project root:
project_root = os.path.abspath(os.path.join(__file__, '..', '..'))
sys.path.insert(0, project_root)
from data.paf_utils.paf_event_dataset import PAFEventDetectionDataset

train_folder = '/home/as4296/palmer_scratch/vit/paf/train'
raw_folder   = '/home/as4296/palmer_scratch/vit/paf/raw'
classnames   = '/home/as4296/palmer_scratch/vit/paf/train/classes.json'

ds = PAFEventDetectionDataset(
    train_folder=train_folder,
    raw_folder=raw_folder,
    classnames_json=classnames,
    num_bins=8,
    H=260, W=346
)

print("Found in raw:", sorted(os.path.splitext(f)[0] for f in os.listdir(raw_folder) if f.endswith('.aedat')))
print("Found in label:", sorted(os.path.splitext(f)[0] for f in os.listdir(os.path.join(train_folder, 'labels')) if f.endswith('.xml')))
print("Dataset keys:", ds.samples)
print("Dataset length:", len(ds))
