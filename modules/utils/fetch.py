from omegaconf import DictConfig

from modules.data.genx import DataModule as genx_data_module
from modules.detection_module import Module as det_module
from modules.data.paf import PAFEventDataModule

def fetch_data_module(config: DictConfig):
    batch_size_train = config.batch_size.train
    batch_size_eval = config.batch_size.eval
    num_workers_generic = config.hardware.get('num_workers', None)
    num_workers_train = config.hardware.num_workers.get('train', num_workers_generic)
    num_workers_eval = config.hardware.num_workers.get('eval', num_workers_generic)
    dataset_str = config.dataset.name
    if dataset_str == 'paf_event':
        train_workers = int(config.hardware.num_workers.train)
        eval_workers  = int(config.hardware.num_workers.eval)
        return PAFEventDataModule(
            train_folder=config.dataset.train_folder,
            raw_folder=config.dataset.raw_folder,
            classnames_json=config.dataset.classnames,
            num_bins=config.dataset.num_bins,
            H=config.dataset.height,
            W=config.dataset.width,
            train_bs=config.batch_size.train,
            val_bs=config.batch_size.eval,
            num_workers=eval_workers
        )

    if dataset_str in {'gen1', 'gen4'}:
        dataset_config = config.dataset
        return genx_data_module(dataset_config,
                                num_workers_train=num_workers_train,
                                num_workers_eval=num_workers_eval,
                                batch_size_train=batch_size_train,
                                batch_size_eval=batch_size_eval)
    raise NotImplementedError

def fetch_model_module(config: DictConfig):
    # not implement judge
    return det_module(config)
