# PAFBenchmark event‐based detection training
python train.py \
  model=rnndet \
  encoder_model=encoder \
  dataset=paf_event \
  +experiment/paf_event="base.yaml" \
  dataset.train_folder=/home/as4296/palmer_scratch/vit/paf/train \
  dataset.raw_folder=/home/as4296/palmer_scratch/vit/paf/raw \
  dataset.classnames=/home/as4296/palmer_scratch/vit/paf/train/classes.json \
  dataset.num_bins=8 \
  dataset.height=260 \
  dataset.width=346 \
  hardware.gpus=[0] \
  batch_size.train=10 \
  batch_size.eval=10 \
  hardware.num_workers.train=4 \
  hardware.num_workers.eval=4 \
  validation.val_check_interval=10000 \
  validation.check_val_every_n_epoch=null \
  training.max_steps=300000 \
  training.learning_rate=0.0003 \
  encoder_model.encoder.num_blocks=[2,2,2,2] \
  logging.train.high_dim.enable=false \
  logging.validation.high_dim.enable=false

