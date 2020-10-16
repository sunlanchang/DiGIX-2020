# rank搜索

## 环境要求
1. xgboost
2. lightgbm
3. catboost


## lxf_rank
1. rank部分先在文件夹下建立data、ensemble、input三个文件夹
2. 运行preprocess得到合并并压缩的数据文件
3. 运行lgb得到lightgbm数据结果
4. 运行xgb_cat得到xgboost、catboost合并后的数据结果
5. 运行best_ensemble将lightgbm结果和catboost结果加权融合，得到lxf最终结果，线上0.443019

## zlh_rank
1. 运行catboost_baseline得到catboost结果，线上0.442826

## 最后融合
运行final_ensemble对lxf和zlh加权融合，得到线上最优结果0.444608