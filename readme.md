# ImageRetrieval-Pytroch
[![python version](https://img.shields.io/badge/python-3.6%2B-brightgreen)]()
[![coverage](https://img.shields.io/badge/coverage-100%25-orange)]()

image retrieval

## Table of Contents

- [Structure](#structure)
- [Usage](#usage)
- [Config_file](#config_file)




## structure
```
├── config                     
│     ├── retrieval.yaml           图片检索的相关配置
├── data                           
│     ├── images                   默认图片检索库目录，可在retrieval.yaml中更改
│     ├── output                   检索结果的存放目录
│     ├── query_images             默认存放需要检索的图片，可在retrieval.yaml文件中更改
├── models                           
│     ├── image_retrieval_best.pth             模型权重文件
├── modules 
│     ├── __init__.py                    
│     ├── datasets 
│     │       ├── data_helpers.py                  和dataset相关的一些工具函数
│     │       ├── generic_dataset.py               dataset
│     ├── layers 
│     │       ├── loss.py                  
│     │       ├── functional.py                 
│     │       ├── normalization.py                 标准化类
│     ├── lshash                                   lsh相关文件 
│     ├── networks 
│     │       ├── retrieval_net.py                 检索模型代码    
│     ├── solver 
│     │       ├── feature_extractor.py             特征提取类
│     │       ├── image_retriever.py               图像检索类
│     │       ├── model_initializer.py             模型初始化类 
│     ├── train_log 
│     │       ├── ckpt                         a directory to save checkpoint
│     │       ├── log                          a directory to save log    
├── utils                      一些有用的工具函数   
├── retrieval_demo.py          demo文件，代码入口      

```     
## usage
use default params (all parameters setted in model_config file)
```
python retrieval_demo.py --model_config configs/retrieval.yaml 
```
use custom params
```
python retrieval_demo.py --model_config configs/retrieval.yaml --query_image XXX --image_gallery XXX --query_number XXX 
# custom params
# --query_image: str, 可以单个图片文件地址，或包含一批图片的文件目录
# --image_gallery: str, 图片库目录  
# --query_number: int, 返回结果的数量

```

## config_file
```yaml         
image_gallery: 'data/images'                # 图片库地址
query_image: "data/query_images"            # query图片地址 
query_number: 10                            # 返回结果的数量
checkpoint: 'models/image_retrieval_best.pth.pth'          # 模型权重地址
out_similar_dir: 'data/output/'                            # 检索结果的输出目录

lsh_config:
  hash_size: 0                    # hash值长度
  input_dim: 2048                 # 特征维度   
  num_hash_tables: 1              # hash表数量

feature_path: "data/test_feature.pkl"      # 图片库特征的存放地址
lsh_index_path: "data/test_lsh.pkl"        # lsh索引的存放地址
```

## Contributing

## License

