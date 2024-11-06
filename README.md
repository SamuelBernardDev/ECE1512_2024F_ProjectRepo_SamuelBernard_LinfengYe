# Abstract
In this report, we examine dataset distillation, a modern technique designed to reduce the training costs of deep neural networks
by condensing large, annotated datasets of natural images into smaller, synthetic datasets. This technique aims to maintain, or
even enhance, model performance in terms of test accuracy compared to models trained on the original, full-scale datasets. The
distilled dataset contains images generated to mimic essential features of the original data, effectively reducing redundancy and
preserving critical information required for accurate model training.

For the first part of the report, we focus on DataDam, a state-of-the-art distillation method that uses spatial attention matching to
selectively capture and recreate the most relevant image features. DataDam’s approach leverages spatial attention matching to align
key visual elements between original and distilled images, leading to efficient dataset distillation without significant information
loss. Compared to other methods in the literature, DataDam offers notable advancements in accuracy and training efficiency due
to its attention-based matching mechanism, which ensures that the synthetic images retain meaningful spatial characteristics and
similar distribution of the original dataset. We also discuss potential applications for DataDam.

For the second part of the report, we introduce two other state-of-the-art distillation methods and compare them with the
Attention Matching algorithm (DataDam). As such, one of these methods is the Prioritize Alignment in Dataset Distillation (PAD)
and the other is Condense Dataset by Aligning Features (CAFE). PAD introduces filtering mechanisms at both the information
extraction and embedding phases, thus addressing the dataset misalignment issue that arises in DataDam during distillation. By
sorting samples by difficulty and selecting high-level features in the embedding stage, PAD allows higher-quality synthetic images
and datasets. Alternatively, CAFE aims to solve gradient dominance in gradient matching techniques by introducing layer-wise
feature alignment and dynamic bi-level optimization. Using CAFE prevents overfitting and ensures a balanced representation of
class-specific features. In this analysis, we report the performance results from DataDam, PAD, and CAFE and compare the testing
accuracy of different models trained on their synthetic datasets initialized from both real images and Gaussian noise.

# Results Section
<img width="665" alt="image" src="https://github.com/user-attachments/assets/8057359b-2a6a-43af-a78f-d3ce6c4ebe39">

<img width="647" alt="image" src="https://github.com/user-attachments/assets/a654f918-5b97-4773-8ddc-e76a0bbab9a5">

# Running the experiment
## run.sh for DataDam:
DataDam MNIST with Random Initialization
```
python main_DataDAM.py  --dataset MNIST --model ConvNetD3 --num_eval 20 \
                       --Iteration 1000  --lr_img 0.1 --lr_net 0.01 --ipc 10 --init noise \
                       --data_path dataset --save_path NoiseMnistresultsSmallLr
```

DataDam MHIST with Random Initialization
```
python main_DataDAM.py  --dataset MHIST --model ConvNetD3 --num_eval 20 \
                       --Iteration 1000  --lr_img 0.1 --lr_net 0.01 --ipc 20 --init noise \
                       --data_path dataset --save_path NoiseMnistresultsSmallLr
```
*Change the --init command to real for real image initialization*

DataDam MNIST for Cross-Architecture
```
python cross_arch.py --dataset MNIST --epoch_eval_train 1000 --dd NoiseMnistresultsSmallLr/res_DataDAM_MNIST_ConvNetD3_10ipc_.pt --data_path dataset
```
```
python cross_arch.py --dataset MNIST --epoch_eval_train 1000 --dd ../Datadamzipped_Sam_MNIST_init_real/NoiseMnistresultsSmallLr/res_DataDAM_MNIST_ConvNetD3_10ipc_.pt --data_path dataset
```

DataDam MNIST for Continuous Learning
```
python CL_DM_DATADAM.py --dataset MNIST --model ConvNetD3 --ipc 10 --datadam 1 --dd res_DataDAM_MNIST_ConvNetD3_10ipc_Randominit.pt
```

## run.sh for CAFE:
`python distill.py  --dataset MNIST  --model ConvNet  --ipc 10 --Iteration 1000`

## run.sh for PAD:
```
cd buffer
python buffer_CL.py --dataset=MNIST --model=ConvNet --train_epochs=10 --num_experts=10 \
                    --zca --buffer_path=../buffer_storage/ --data_path=../dataset/ \
                    --sort_method="CIFAR10_GraNd" --rho_max=0.01 --rho_min=0.01 --alpha=0.3 \
                    --lr_teacher=0.01 --mom=0. --batch_train=256 --init_ratio=0.75 --add_end_epoch=20 \
                    --rm_epoch_first=40 --rm_epoch_second=60 --rm_easy_ratio_first=0.1 \
                    --rm_easy_ratio_second=0.2
```
```
cd ../distill
python PAD_depth.py --cfg ../configs/MNIST/ConvIN/IPC10.yaml
```
