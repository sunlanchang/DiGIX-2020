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