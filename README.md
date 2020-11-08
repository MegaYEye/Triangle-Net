## Classification

1. Prepare data
    
    For ModelNet 40, download dataset from 
    https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip

    Next, extract it to ```<code folder>/data/modelnet40_ply_hdf5_2048``` folder.

    For ScanObjectNN, please get the download link according to the instruction from this link (https://hkust-vgd.github.io/scanobjectnn/). Then, extract ```training_objectdataset_augmentedrot_scale75.h5``` and ```test_objectdataset_augmentedrot_scale75.h5``` to ```<code folder>/data/ScanObjectNN_nobg```. 

2.  training

    for training on ModelNet40 with reconstruction network:
    
    ```
    python train_recon.py
    ```
    
    for training  on ModelNet40 without reconstruction network, of which the training is faster at a cost of minor accuracy drop:
    
    ```
    python train_wo_recon.py
    ```
    
    For both of the training configuration, ```--n_points``` can specify the number of points. 

    For training on ScanObjectNN:

    ```
    python train_scanobjects.py
    ```

## Segmentation

1. Prepare data
   
    Download dataset from: https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip
   
    Next, extract it to ```<code folder>/data/shapenetcore_partanno_segmentation_benchmark_v0_normal``` folder

2. Preprocessing
   
   To accelerate disk IO, we save the dataset as npy files:
    ```
    python segment_data_preprocess.py
    ```
3. Training
   ```
   python train_partseg.py
   ```


## Comparison experiment

We refer the following code for comparison experiments

[PointNet & PointNet++] https://github.com/yanx27/Pointnet_Pointnet2_pytorch

[DGCNN] https://github.com/WangYueFt/dgcnn   

[RI-CONV] https://github.com/hkust-vgd/riconv

[3DmFV] https://github.com/sitzikbs/3DmFV-Net


