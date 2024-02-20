CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --master_port=20335 --nproc_per_node=1 train_distillation.py --distributed=True
