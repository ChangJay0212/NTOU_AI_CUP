# 簡介
這現論文-"Real-time Scene Text Detection with Differentiable Binarization"的project，使用PyTorch. [此論文](https://arxiv.org/abs/1911.08947) 提出了一個可以在任意形狀的場景中即時偵測文字的文字檢測器，目前這個檢測器是在現有的檢測器中效果最好的。

程式碼繼承(inherited)自[DB]([https://github.com/Megvii-CSG/MegReader](https://github.com/MhLiao/DB)).


## 安裝

### Requirements:
- Python3
- PyTorch >= 1.2 
- GCC >= 4.9 (This is important for PyTorch)
- CUDA >= 9.0 (10.1 is recommended)


```bash=
  # first, make sure that your conda is setup properly with the right environment
  # for that, check that `which conda`, `which pip` and `which python` points to the
  # right path. From a clean conda env, this is what you need to do

  conda create --name DB -y
  conda activate DB

  # this installs the right pip and dependencies for the fresh python
  conda install ipython pip

  # python dependencies
  pip install -r requirement.txt

  # install PyTorch with cuda-10.1
  conda install pytorch torchvision cudatoolkit=10.1 -c pytorch

  # clone repo
  git clone https://github.com/MhLiao/DB.git
  cd DB/

  # build deformable convolution opertor
  # make sure your cuda path of $CUDA_HOME is the same version as your cuda in PyTorch
  # make sure GCC >= 4.9
  # you need to delete the build directory before you re-build it.
  echo $CUDA_HOME
  cd assets/ops/dcn/
  python setup.py build_ext --inplace

```

## Models
下載已經訓練好的Models [Baidu Drive](https://pan.baidu.com/s/1vxcdpOswTK6MxJyPIJlBkA) (download code: p6u3), [Google Drive](https://drive.google.com/open?id=1T9n0HTP3X3Y_nJ0D1ekMhCQRHntORLJG).
```
  pre-trained-model-synthtext   -- used to finetune models, not for evaluation
  td500_resnet18
  td500_resnet50
  totaltext_resnet18
  totaltext_resnet50
```

## Datasets
數據集的根目錄可以是```DB/datasets/```

Download the converted ground-truth and data list [Baidu Drive](https://pan.baidu.com/s/1BPYxcZnLXN87rQKmz9PFYA) (download code: mz0a), [Google Drive](https://drive.google.com/open?id=12ozVTiBIqK8rUFWLUrlquNfoQxL2kAl7). Datasets的圖片都可以在它們的官網上取得。

## Testing
### 預備 dataset
範例-test images的路徑:
```
  datasets/total_text/train_images
  datasets/total_text/train_gts
  datasets/total_text/train_list.txt
  datasets/total_text/test_images
  datasets/total_text/test_gts
  datasets/total_text/test_list.txt
```
在```base_totaltext.yaml```設定 data list 路徑

### Config file
**The YAML files with the name of ```base*.yaml``` should not be used as the training or testing config file directly.**

### Demo
使用單張圖片，範例:

```CUDA_VISIBLE_DEVICES=0 python demo.py experiments/seg_detector/totaltext_resnet18_deform_thre.yaml --image_path datasets/total_text/test_images/img10.jpg --resume path-to-model-directory/totaltext_resnet18 --polygon --box_thresh 0.7 --visualize```

結果會顯示在 `demo_results`.

### Evaluate the performance
Note that we do not provide all the protocols for all benchmarks for simplification. The embedded evaluation protocol in the code is modified from the protocol of ICDAR 2015 dataset while support arbitrary-shape polygons. It almost produces the same results as the pascal evaluation protocol in Total-Text dataset. 

The `img651.jpg` in the test set of Total-Text contains exif info for a 90° rotation thus the gt does not match the image. You should read and re-write this image to get normal results. The converted image is also provided in the dataset links. 

The following command can re-implement the results in the paper:

```
CUDA_VISIBLE_DEVICES=0 python eval.py experiments/seg_detector/totaltext_resnet18_deform_thre.yaml --resume path-to-model-directory/totaltext_resnet18 --polygon --box_thresh 0.7

CUDA_VISIBLE_DEVICES=0 python eval.py experiments/seg_detector/totaltext_resnet50_deform_thre.yaml --resume path-to-model-directory/totaltext_resnet50 --polygon --box_thresh 0.6

CUDA_VISIBLE_DEVICES=0 python eval.py experiments/seg_detector/td500_resnet18_deform_thre.yaml --resume path-to-model-directory/td500_resnet18 --box_thresh 0.5

CUDA_VISIBLE_DEVICES=0 python eval.py experiments/seg_detector/td500_resnet50_deform_thre.yaml --resume path-to-model-directory/td500_resnet50 --box_thresh 0.5

# short side 736, which can be changed in base_ic15.yaml
CUDA_VISIBLE_DEVICES=0 python eval.py experiments/seg_detector/ic15_resnet18_deform_thre.yaml --resume path-to-model-directory/ic15_resnet18 --box_thresh 0.55

# short side 736, which can be changed in base_ic15.yaml
CUDA_VISIBLE_DEVICES=0 python eval.py experiments/seg_detector/ic15_resnet50_deform_thre.yaml --resume path-to-model-directory/ic15_resnet50 --box_thresh 0.6

# short side 1152, which can be changed in base_ic15.yaml
CUDA_VISIBLE_DEVICES=0 python eval.py experiments/seg_detector/ic15_resnet50_deform_thre.yaml --resume path-to-model-directory/ic15_resnet50 --box_thresh 0.6
```

```box_thresh``` can be used to balance the precision and recall, which may be different for different datasets to get a good F-measure. ```polygon``` is only used for arbitrary-shape text dataset. The size of the input images are defined in ```validate_data->processes->AugmentDetectionData``` in ```base_*.yaml```.

### Evaluate the speed 
Set ```adaptive``` to ```False``` in the yaml file to speedup the inference without decreasing the performance. The speed is evaluated by performing a testing image for 50 times to exclude extra IO time.

```CUDA_VISIBLE_DEVICES=0 python eval.py experiments/seg_detector/totaltext_resnet18_deform_thre.yaml --resume path-to-model-directory/totaltext_resnet18 --polygon --box_thresh 0.7 --speed```

Note that the speed is related to both to the GPU and the CPU since the model runs with the GPU and the post-processing algorithm runs with the CPU.

## Training
Check the paths of data_dir and data_list in the base_*.yaml file. For better performance, you can first per-train the model with SynthText and then fine-tune it with the specific real-world dataset.

```CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py path-to-yaml-file --num_gpus 4```

You can also try distributed training (**Note that the distributed mode is not fully tested. I am not sure whether it can achieves the same performance as non-distributed training.**)

```CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 train.py path-to-yaml-file --num_gpus 4```

## Third-party implementations
* Keras implementation: [xuannianz/DifferentiableBinarization](https://github.com/xuannianz/DifferentiableBinarization)
* DB is included in [OpenCV](https://github.com/opencv/opencv/blob/master/doc/tutorials/dnn/dnn_text_spotting/dnn_text_spotting.markdown)
* DB is included in [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)

## Citing the related works

Please cite the related works in your publications if it helps your research:

     @inproceedings{liao2020real,
      author={Liao, Minghui and Wan, Zhaoyi and Yao, Cong and Chen, Kai and Bai, Xiang},
      title={Real-time Scene Text Detection with Differentiable Binarization},
      booktitle={Proc. AAAI},
      year={2020}
    }


    
