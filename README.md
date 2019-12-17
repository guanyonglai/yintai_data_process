2019-12-12 15:28:53  
001_move_data_to_20.py     把数据从移动硬盘拷贝到我们的服务器  
002_extract_data.py        依据json文件抠出行人框提取行人轨迹  
003_detect_pytorch.py      检测代码，用去剔除脏图：宽高比不合适、人体置信度小于0.7(这里很耗时)  
004_FeatExtract_batch.py   MGN网络提取图片特征  
005_label_data_process.py  对兼职同学标注的数据进行处理(处理数据本身不难，难的是对兼职标错的数据进行修正)  
