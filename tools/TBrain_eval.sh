cd ../
CUDA_VISIBLE_DEVICES=2 \
python eval.py experiments/seg_detector/TBrain_resnet50_deform_thre.yaml \
--resume model/totaltext_resnet50 \
--polygon \
--box_thresh 0.6
---------------------------------------------------------------------------------------------------
res50 :


CUDA_VISIBLE_DEVICES=0 python eval1.py experiments/seg_detector/TBrain_resnet50_deform_thre.yaml --resume /home/mmplab603/下載/DB-20210611T085515Z-001/res50/model_epoch_967_minibatch_60000 --box_thresh 0.5 --result_dir AICUP_result 
-------------------------------------------------------------------------------------------------
res152:

CUDA_VISIBLE_DEVICES=0 python eval1.py experiments/seg_detector/TBrain_resnet152_deform_thre.yaml --resume /home/mmplab603/下載/DB-20210611T085515Z-001/res152/model_epoch_662_minibatch_84000 --box_thresh 0.5 --result_dir AICUP_result




