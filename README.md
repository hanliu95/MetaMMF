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

## Usage
* Execution sequence

  The execution sequence of codes is as follows: data_load.py--->data_triple.py--->model_train.py--->model_test.py
  
* Execution results

  During the execution of file model_train.py, the epoch, iteration, and training loss will be printed as the training process:
  
    ```
[1, 500] loss: 0.21953
[1, 1000] loss: 0.15628
[1, 1500] loss: 0.13521
[1, 2000] loss: 0.12699
[1, 2500] loss: 0.12122
[1, 3000] loss: 0.11738
  ...
  ```
