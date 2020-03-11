#
#
#
#  this script train CenterNet 101 on custom coco dataset
#  custom means we using coco format but select our
#  custom categories, such as we only using 3 classes
#  inside coco. change inside ./datasets/dataset/coco_custom.py
#  also raw data soft link under ./data/coco_custom/images and ./data/coco_custom/annotations/instance_train2017.json




python3 main.py ctdet --exp_id coco_custom_res101_3classes \
--dataset coco_custom \
--arch res_101 \
--batch_size 4 \
--num_epochs 900 \
--master_batch 5 \
--val_intervals -1 \
--lr 3.75e-4 \
--gpus 1
