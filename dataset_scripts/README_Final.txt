I. Things to do before dataset preperation:

Step 1: Check if you have all Raw (.pcd file) and labelled pointclouds (.ply file) of both synthetic and real world data in /home/teal-sutd/Documents/Open3D-ML/jeremy_ws/Point_clouds/Raw_and_Labelled_data

Step 2: Check if you have the training dataset file in /home/teal-sutd/Documents/Open3D-ML/jeremy_ws/Point_clouds/Training. You should see two folders: "dataset" and "dataset_archive". The folder "dataset" refers to the current dataset that the semantic model pipeline is using, while the "dataset_archive_" folder refers to past datasets that were used in training. 

Step 3: Check if you have the following scripts in the /home/teal-sutd/Documents/Open3D-ML/jeremy_ws/Point_clouds folder. 
	a. recommend_spatial_split_params.py: Used to compute spatial_split params based on raw point cloud params. User gives folder directory and it will check all ply files inside to compute the params in terminal. 
	b. spatial_grid_splitting.py: Used to perform spatial split of raw point clouds (ply format) into SemanticKITTI format (bins and labels) used for training. 
	c. random_point_split.py: Used to perform random point split of raw point clouds (ply format) into SemanticKITTI format (bins and labels) used for training. 
	d. validation_selected_stratified.py: Used to perform validation split of splitted point clouds while ensuring class balance in both training and validation folders after split. 
	e. validation_select_custom.py: Custom validation split script that make a small validation set while protecting rare classes in training.
	f. oversample_v3.py: Used to oversample rare classes of point clouds as part of data augmentation to increase the inference accuracy of rare classes. (Takes in bin and labels files and performs oversampling) 
	g. compute_weights_ver4.py: Used to compute the class weights used in kpconv_semantickitti.yml and randlanet_semantickitti.yml. 
	h. export_predicted_ply_FINAL.py: Used to export predicted results (bins and labels) into ply format to view in CloudCompare. It also applies the cloud compare labels and colors. 
	i. ply_to_semantic.py: Used to convert raw point clouds from LIDAR (.pcd) format into CloudCompare readable format - ply mesh (.ply). 
----------------------------------
II. Dataset Preperation

Step 1. Convert raw point cloud data (.pcd) into ply mesh format (.ply), for labelling in CloudCompare.
conda activate open3d_cuda128
cd PointClouds
python3 ply_to_semantic.py

Step 2. Perform labelling in CloudCompare, and save color_scale.xml file from your label. 

Step 3: Export the labelled point cloud data in ply mesh format (.ply), using BINARY format to save space. 

Step 4. Choose a point cloud splitting method, either Spatial Grid Split or Random Point Split. For all files do adjust the INPUT_DIR and OUTPUT_DIR when required. 
	a. Spatial Point Split
		i. First run recommended_split_params.py to compute what is the recommended spatial split params to use.
			cd PointClouds
			python3 recommended_split_params.py
		
		ii. Run spatial_grid_splitting.py to perform spatial split of the raw point cloud params into many bin and labels files. Use only real point cloud data for this. 
		Remember to adjust INPUT_DIR to point to real point cloud data location. 
			python3 spatial_grid_splitting.py
		
		iii. Run validation_selected_stratified.py to create your validation dataset of using real point cloud data. The validation dataset should only contain real point 	cloud data.  
			python3 validation_selected_stratified.py
		
		iv. Repeat steps i and ii on the synthetic dataset.
		Remember to adjust INPUT_DIR to point to synthetic point cloud data location. 

			python3 recommended_split_params.py
			python3 spatial_grid_splitting.py
			python3 validation_selected_stratified.py
		
		v. Perform oversample_v3.py on the synthetic and real world point cloud sequences. 
		Remember to adjust INPUT_DIR to point to "dataset" point cloud data location. 
			python3 oversample_v3.py
		
		vi. Use compute_weights_ver4.py to compute the class weights that you would use for training.
			python3 compute_weights_ver4.py
		
		vii. After u have done steps i to vi, you have successfully created your training dataset and you can skip to "III. Training Procedures"
		
		
	b. Random Point Split
		i. Run random_point_split.py to perform spatial split of the raw point cloud params into many bin and labels files. Use only real point cloud data for this. Adjust the params as u train and see which is best.
			cd PointClouds
			python3 random_point_split.py
		
		ii. Run validation_selected_stratified.py to create your validation dataset of using real point cloud data. The validation dataset should only contain real point cloud data. 
			python3 validation_selected_stratified.py
		
		iii. Repeat steps a and b on the synthetic dataset.

			python3 random_point_split.py
			python3 validation_selected_stratified
		
		iv. Perform oversample_v3.py on the synthetic and real world point cloud sequences. 
			python3 oversample_v3.py 
			
		v. Use compute_weights_ver4.py to compute the class weights that you would use for training.
			python3 compute_weights_ver4.py
			
		vi. After u have done steps i to vi, you have successfully created your training dataset and you can skip to "III. Training Procedures"
		
	
---------------------------------	
III. Training Procedure

Step 1: Open a 4 terminals terminal and run these 
conda activate open3d_cuda128

Step 2: On two terminal Go into the open3d-ml directory
cd /home/teal-sutd/Documents/Open3D-ML

Step 3: On two other terminals, run these commands seperately:
watch nvidia-smi  
htop

These commands are used to monitor GPU and CPU usage respectively. 

Step 4: Run the kpconv or randlanet training pipeline 

Randlanet: 
cd /home/jeremychia/Documents/Open3D-ML
python scripts/run_pipeline.py torch   -c /home/jeremychia/Documents/Open3D-ML/ml3d/configs/randlanet_semantickitti.yml   --dataset.dataset_path /home/jeremychia/Documents/Point_clouds/Training  --pipeline SemanticSegmentation   --dataset.use_cache True   --device cuda   --device_ids 0

#multi process train
python launch_train.py torch   -c /home/teal-sutd/jeremy_main_ws/dataset06/randlanet_semantickitti.yml   --dataset.dataset_path /home/teal-sutd/jeremy_main_ws/dataset06   --pipeline SemanticSegmentation   --dataset.use_cache True   --pipeline.num_workers 12   --device cuda   --device_ids 0

KPconv:
cd /home/jeremychia/Documents/Open3D-ML
python scripts/run_pipeline.py torch   -c /home/jeremychia/Documents/Open3D-ML/ml3d/configs/modified/kpconv_semantickitti.yml  --dataset.dataset_path /home/jeremychia/Documents/Point_clouds/Training  --pipeline SemanticSegmentation   --dataset.use_cache True   --device cuda   --device_ids 0	

Step 4a: Run the tensorboard to see the status
cd /home/jeremychia/Documents/Open3D-ML
tensorboard --logdir <path_to_log_file> --port 6006
tensorboard --logdir /home/jeremychia/Documents/Open3D-ML/train_log/00001_KPFCNN_SemanticKITTI_torch --port 6006
example: tensorboard --logdir ./train_log/00003_KPFCNN_SemanticKITTI_torch --port 6006


/home/jeremychia/Documents/Open3D-ML/train_log/00001_KPFCNN_SemanticKITTI_tensorboard --logdir ./train_log/00001_KPFCNN_SemanticKITTI_torch --port 6006

Point Transformer:

python scripts/run_pipeline.py torch -c ml3d/configs/pointtransformer_s3dis.yml --dataset.dataset_path /home/jeremychia/Documents/Point_clouds/Training/dataset_s3dis/Stanford3dDataset_v1.2_Aligned_Version

tensorboard --logdir /home/jeremychia/Documents/Open3D-ML/train_log/00001_PointTransformer_S3DIS_torch --port 6006


tensorboard --logdir /home/jeremychia/Documents/Open3D-ML/train_log_ft185/00001_PointTransformer_S3DIS_torch --port 6007



torch

Step 5: After training run the test pipeline. 

Randlanet 
python scripts/run_pipeline.py torch -c /home/jeremychia/Documents/Open3D-ML/ml3d/configs/randlanet_semantickitti.yml --split test --dataset.dataset_path /home/jeremychia/Documents/Point_clouds/Training --pipeline SemanticSegmentation --dataset.use_cache True --device cuda --device_ids 0

python scripts/run_pipeline.py torch \
  -c /home/teal-sutd/jeremy_main_ws/dataset06/randlanet_semantickitti.yml \
  --split predict \
  --ckpt_path /home/teal-sutd/Documents/Open3D-ML/logs/RandLANet_SemanticKITTI_torch/checkpoint/ckpt_00246.pth \
  --dataset.dataset_path /home/teal-sutd/jeremy_main_ws/dataset06 \
  --predict_seq testfullpcd \
  --device cuda --device_ids 0
  




KPconv 
python scripts/run_pipeline.py torch -c /home/jeremychia/Documents/Open3D-ML/ml3d/configs/modified/kpconv_semantickitti.yml --split test --dataset.dataset_path /home/jeremychia/Documents/Point_clouds/Training --pipeline SemanticSegmentation --dataset.use_cache True --device cuda --device_ids 0

python scripts/run_pipeline.py torch \
  -c /home/jeremychia/Documents/Open3D-ML/ml3d/configs/modified/kpconv_semantickitti.yml \
  --split predict \
  --ckpt_path /home/jeremychia/Documents/Open3D-ML/logs/KPFCNN_SemanticKITTI_torch/checkpoint/ckpt_00339.pth\
  --dataset.dataset_path /home/jeremychia/Documents/Point_clouds/Training \
  --predict_seq testfullpcd_oversampled2_d2 \
  --device cuda --device_ids 0
  
python scripts/run_pipeline.py torch \
  -c /home/jeremychia/Documents/Open3D-ML/ml3d/configs/modified/kpconv_semantickitti.yml \
  --split predict \
  --ckpt_path /home/jeremychia/Documents/Open3D-ML/logs/KPFCNN_SemanticKITTI_torch/checkpoint/ckpt_00426.pth\
  --dataset.dataset_path /home/jeremychia/Documents/Point_clouds/Training \
  --predict_seq testfullpcd \
  --device cuda --device_ids 0  


python scripts/run_pipeline.py torch \
  -c /home/jeremychia/Documents/Open3D-ML/ml3d/configs/modified/kpconv_semantickitti.yml \
  --split predict \
  --ckpt_path /home/jeremychia/Documents/Point_clouds/Training/dataset08_models/KPFCNN_SemanticKITTI_torch/checkpoint/ckpt_00399.pth \
  --dataset.dataset_path /home/jeremychia/Documents/Point_clouds/Training \
  --predict_seq testfullpcd\
  --device cuda --device_ids 0

Point Transformer:

conda activate o3dml_cuda128
   

python convert_ply_to_s3dis.py \
    --input /home/jeremychia/Documents/Point_clouds/Training/dataset_s3dis/Stanford3dDataset_v1.2_Aligned_Version/testset2/ \
    --dataset-path /home/jeremychia/Documents/Point_clouds/Training/dataset_s3dis/Stanford3dDataset_v1.2_Aligned_Version \
    --test-area 3

python scripts/run_pipeline.py torch \
    -c ml3d/configs/pointtransformer_s3dis.yml \
    --split test \
    --ckpt_path logs/PointTransformer_S3DIS_torch/checkpoint/ckpt_00194.pth \
    --dataset.dataset_path /home/jeremychia/Documents/Point_clouds/Training/dataset_s3dis/Stanford3dDataset_v1.2_Aligned_Version \
    --dataset.test_area_idx 3 \
    --dataset.num_points 40960 \
    --pipeline.test_batch_size 1 \
    --model.max_voxels 10000 \
    --model.voxel_size 0.08 \
    --device cuda
    
cd /home/jeremychia/Documents/Open3D-ML
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128 \
python -u scripts/run_pipeline.py torch \
  -c ml3d/configs/pointtransformer_s3dis.yml \
  --split test \
  --ckpt_path logs/PointTransformer_S3DIS_torch/checkpoint/ckpt_00194.pth \
  --dataset.dataset_path /home/jeremychia/Documents/Point_clouds/Training/dataset_s3dis/Stanford3dDataset_v1.2_Aligned_Version \
  --dataset.test_area_idx 3 \
  --dataset.num_points 8192 \
  --model.voxel_size 0.16 \
  --pipeline.test_batch_size 1 \
  --device cuda

cd /home/jeremychia/Documents/Open3D-ML
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128 \
python -u scripts/run_pipeline.py torch \
  -c ml3d/configs/pointtransformer_s3dis.yml \
  --split test \
  --ckpt_path logs/PointTransformer_S3DIS_torch/checkpoint/ckpt_00180.pth \
  --dataset.dataset_path /home/jeremychia/Documents/Point_clouds/Training/dataset_s3dis/Stanford3dDataset_v1.2_Aligned_Version \
  --dataset.test_area_idx 3 \
  --dataset.num_points 8192 \
  --pipeline.test_batch_size 1 \
  --model.voxel_size 0.16 \
  --device cuda



tail -f /home/jeremychia/Documents/Open3D-ML/logs/PointTransformer_S3DIS_torch/log_test_2026-02-26_14-07-00.txt

python /home/jeremychia/Documents/Point_clouds/merge_split_predictions_s3dis.py \
  --pred-dir /home/jeremychia/Documents/Open3D-ML/test/S3DIS \
  --meta-dir /home/jeremychia/Documents/Point_clouds/Training/dataset_s3dis/Stanford3dDataset_v1.2_Aligned_Version/original_pkl_meta \
  --out-dir /home/jeremychia/Documents/Open3D-ML/test/S3DIS/merged


python /home/jeremychia/Documents/Point_clouds/npy_to_ply_s3dis.py \
  --pred-dir /home/jeremychia/Documents/Open3D-ML/test/S3DIS/merged \
  --dataset-path /home/jeremychia/Documents/Point_clouds/Training/dataset_s3dis/Stanford3dDataset_v1.2_Aligned_Version \
  --out-dir /home/jeremychia/Documents/Point_clouds/predicted_ply_merged


Step 6: After training, use the prediction pipeline in open3d-ml and use export_predicted_ply_FINAL.py to convert the predicted bin and labels back into ply for viewing in CloudCompare.

For RandLA-net and KPConv
python3 export_predicted_ply_FINAL.py

For PointTransformer

python /home/jeremychia/Documents/Point_clouds/npy_to_ply_s3dis.py \
  --pred-dir /home/jeremychia/Documents/Open3D-ML/test/S3DIS/merged \
  --dataset-path /home/jeremychia/Documents/Point_clouds/Training/dataset_s3dis/Stanford3dDataset_v1.2_Aligned_Version \
  --out-dir /home/jeremychia/Documents/Point_clouds/predicted_ply_merged


Fine Tuning Training

Point_transformer.py
python /home/jeremychia/Documents/Point_clouds/finetune_weights_pointtransformer.py   --config /home/jeremychia/Documents/Open3D-ML/ml3d/configs/pointtransformer_s3dis.yml   --mode ce   --ce-power 1.0    --ce-mult "12:0.85,1:1.08,4:1.03,0:0.95"

python /home/jeremychia/Documents/Point_clouds/finetune_weights_pointtransformer.py   --config /home/jeremychia/Documents/Open3D-ML/ml3d/configs/pointtransformer_s3dis.yml   --mode ce   --ce-power 1.0    --ce-mult "1:1.05,7:0.97"

python /home/jeremychia/Documents/Point_clouds/finetune_weights_pointtransformer.py   --config /home/jeremychia/Documents/Open3D-ML/ml3d/configs/pointtransformer_s3dis.yml   --mode ce   --ce-power 0.95    --ce-mult "11:0.995,7:0.995,10:1.02"

python /home/jeremychia/Documents/Point_clouds/finetune_weights_pointtransformer.py   --config /home/jeremychia/Documents/Open3D-ML/ml3d/configs/pointtransformer_s3dis.yml   --mode ce   --ce-power 1.0]    --ce-mult "11:0.98,7:0.98,10:1.04"


-----------------------------------
IV. Additional Information

Training class types

CloudCompare No | Object Class
0.000000 	"Unlabelled" 
1.000000 	"Wall" 
3.000000 	"Staircase"
4.000000 	"Fixed_Obstacles" 
6.000000 	"Safety_Barriers_And_Signs" 
7.000000 	"Temporary_Utilities" 
8.000000 	"Scaffold_Structure" 
10.000000 	"Large_Materials" 
11.000000 	"Stored_Equipment" 
12.000000 	"Mobile_Machines_And_Vehicles"
13.000000 	"Movable_Objects"
14.000000 	"Containers_And_Pallets" 

Train ID | Object Class
0 	  "Unlabelled" 
1	  "Wall" 
2	  "Staircase"
3	  "Fixed_Obstacles" 
4	  "Safety_Barriers_And_Signs" 
5	  "Temporary_Utilities" 
6	  "Scaffold_Structure" 
7	  "Large_Materials" 
8	  "Stored_Equipment" 
9	  "Mobile_Machines_And_Vehicles"
10	  "Movable_Objects"
11	  "Containers_And_Pallets" 
12	  "Small_Tools"

learning_map:
  0: 0     # Unlabelled / background
  1: 1     # Wall
  3: 2     # Staircase
  4: 3     # Fixed_Obstacles

  5: 4     # Temporary_Ramps
  6: 5     # Safety_Barriers_And_Signs
  7: 6     # Temporary_Utilities
  8: 7     # Scaffold_Structure

  9: 8     # Semi-Fixed_Obstacles
  10: 9    # Large_Materials
  11: 10   # Stored_Equipment
  12: 11   # Mobile_Machines_And_Vehicles
  13: 12   # Movable_Objects
  14: 13   # Containers_And_Pallets
  15: 14   # Small_Tools
  17: 15   # Portable_Objects


Omitted classes from training

CloudCompare No | Object Class
2.000000 	"Drains/Canals" 
5.000000 	"Temporary_Ramps" 
9.000000 	"Semi-Fixed_Obstacles"
15.000000 	"Small_Tools" (to include when dataset is sufficient) 
16.000000 	"Debris_And_Loose_Packaging"
17.000000 	"Portable_Objects" (to include when dataset is sufficient)
18.000000 	"Unclassified_Items"

----------------
Original

0.000000 "Unlabelled" 1
1.000000 "Wall" 2
2.000000 "Drains/Canals" x
3.000000 "Staircase" x
4.000000 "Fixed_Obstacles" 3```````1	`
5.000000 "Temporary_Ramps" x 
6.000000 "Safety_Barriers_And_Signs" 4
7.000000 "Temporary_Utilities" 5
8.000000 "Scaffold_Structure" 6
9.000000 "Semi-Fixed_Obstacles" x
10.000000 "Large_Materials" 7 
11.000000 "Stored_Equipment" 8 
12.000000 "Mobile_Machines_And_Vehicles"9 
13.000000 "Movable_Objects"10
14.000000 "Containers_And_Pallets" 11
15.000000 "Small_Tools" x
16.000000 "Debris_And_Loose_Packaging"x  
17.000000 "Portable_Objects" 12
18.000000 "Unclassified_Items" x 
