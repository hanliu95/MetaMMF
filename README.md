# MetaMMF
This is our experiment codes for the paper:

Dynamic Multimodal Fusion via Meta-Learning Towards Micro-Video Recommendation

## Environment settings
* Python 3.7
* Pytorch 1.7.0+cu101
* PyTorch Geometric 1.7.2
* Numpy 1.19.5
* Pandas 1.3.5

## File specification
* data_load.py : loads the raw data in path `./dataset_sample`, and the results are saved in path `./pro_data`.
* data_triple.py : obtains the triplets for model training, and the results are saved in path `./pro_triple`.
* GCN_model.py : implements the model framework of MetaMMF_GCN.
* Fusion_model.py : implements the model framework of MetaMMF.
* meta_layer.py : the regular layer of MetaMMF.
* cpd_layer.py : the simplified layer of MetaMMF.
* model_train.py : the training process of model.
* model_test.py : the testing process of model.
