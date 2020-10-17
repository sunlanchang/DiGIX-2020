# DiGIX-2020
2020 华为DIGIX 比赛文档

# Usage

1. 首次使用直接`git clone到本地即可`
1. 在自己文件夹创建`.gitignore`，将数据等文件夹写入`.gitignore`，不要同步数据，只同步代码。
1. 在每次写代码前要`git pull`，拉取一下队友的代码再开始写。
1. 写完代码后，使用`git add .`添加到本地缓存区，提交代码前使用`git commit -m '日志内容...'`，添加本次代码内容的日志，使用`git push`提交到GitHub，若出现merge冲突可能需要手工解决代码合并冲突。

# Introduction

```bash
$ cd zlh
$ tree -L 1
.
├── base
├── ctb_baseline0913.ipynb
├── data_reprocess_step1.py      # 得到最原始的特征one-hot特征,data_reprocess要依次运行
├── data_reprocess_step2.ipynb
├── data_reprocess_step3.ipynb
├── data_reprocess_step3.py
├── data_reprocess_step4.ipynb
├── data_reprocess_step5.ipynb
├── get_sample_data.ipynb        # 抽样一小部分数据用来测试代码
├── testmulti_process_0911.py    # 人造的数据集用于测试代码
└── test_w2v_feature.py
```

**数据介绍**
train_data.csv给定的特征包含了
- 用户特征比如：uid age gender city device_name device_type...
- 广告特征比如：task_id adv_id creative_type_cd...
- 用户和广告交互的时间：pd_d (训练集pt_d=1~7 测试集pt_d=8)

```
label 标签
uid 匿名化处理后的用户唯一标识
task_id 广告任务唯一标识
adv_id 广告任务对应的素材id
creat_type_cd 素材的创意类型id
adv_prim_id 广告任务对应的广告主id
dev_id 广告任务对应的开发者id
inter_typ_cd 广告任务对应的素材的交互类型
slot_id 广告位id
spread_app_id 投放广告任务对应的应用id
tags 广告任务对应的应用的标签
app_first_class 广告任务对应的应用的一级分类
app_second_class 广告任务对应的应用的二级分类
age 用户的年龄
city 用户的常驻城市
city_rank 用户常驻城市的等级
device_name 用户使用的手机机型
device_size 用户使用手机的尺寸
career 用户的职业
gender 用户的性别
net_type 行为发生的网络状态
residence 用户的常驻省份
his_app_size app存储尺寸
his_on_shelf_time 上架时间
app_score app得分（百分制噪声）
emui_dev emui版本号
list_time 上市时间
device_price 设备价格
up_life_duration 华为账号用户生命时长
up_membership_grade 服务会员级别
membership_life_duration 会员用户生命时长
consume_purchase 付费用户
communication_onlinerate手机在线时段
communication_avgonline_30d 手机日在线时长
indu_name 广告行业信息
pt_d 行为发生的时间
```

**For Details**

1. data_reprocess_step1.py 保存为两个pkl文件，其中一个为所有数据，另一个为展开cmr的所有数据
1. data_reprocess_step2.py 加入统计特征，进行target encoding，具体的是使用label的mean作为target encodeing
1. data_reprocess_step3.py 对cmr做embedding，对用户过去一天的序列做embedding