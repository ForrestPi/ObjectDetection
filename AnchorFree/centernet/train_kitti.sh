python3 main.py ddd --exp_id subcnn --dataset kitti \
--kitti_split subcnn \
--batch_size 4 \
--master_batch 7 \
--num_epochs 700 \
--lr_step 45,60 \
--resume \
--gpus 0