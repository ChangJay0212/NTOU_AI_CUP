cd ../
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9 \
python train.py experiments/seg_detector/TBrain_resnet50_deform_thre.yaml \
--num_gpus 10 \
--num_workers 2 \
--batch_size 128 \
--resume model/model_epoch_384_minibatch_48000
