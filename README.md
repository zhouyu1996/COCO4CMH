## MS COCO dataset for cross-modal hashing 


##### 0. update

 I fixed a bug in our project.
 The bug is that the _np.random.randint()_ may produce the same index to conduct our dataset.
 we can use this _random.sample()_ to select different number from a array.

##### 1. introduction  

 This repository is created to handle the original MS COCO-2014 dataset for cross-modal retrieval task.


##### 2. configurations

 You can download the original MS COCO dataset from [here](https://cocodataset.org/#home)

 For cross-modal retrieval, we need images (image modality), annotations (text modality) and labels (supervised information).

 The configuration of our final dataset is shown as follows (you can surely change the partition by change some parameters in _coco_data.py_):




| Configuration tabel  |            images             |   texts    |       labels        |
| :------------------: | :---------------------------: | :--------: | :-----------------: |
|  training instance   |             5000              |  5000      |        000        |
| test(query) instance |             2500              |    2500    |        2500         |
| retrieval(database) instance  |             40000             |   40000    |        40000        |
|    representation    | 256 &times; 256 &times; 3 RGB | 2000-d BOW | 80-d one-hot vector |



##### 3. copyright

 This is part of my research for my master degree in Chongqing University.

 This code is free for educational, academic research and non-profit purposes. 

 Not for commercial/industrial activities. 



##### 4. contact

 Welcome to contact with Yu Zhou at 18990848997@163.com.  

 If you are a Chinese, you can surely write an e-mail in Chinese. 

 If you find the code useful in your research, please consider citing the related work: 

```
 We will update bib information after our paper is published. 
```
