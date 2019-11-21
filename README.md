# 评分卡的半自动建模脚本
直接运行get_scorecard_model即可

## 函数说明：
* tree_split ：决策树分箱
* quantile_split ：等频分箱
* cal_woe ：计算woe
* monot_trim ： woe调成单调递减或单调递增
* judge_increasing ：判断一个list是否单调递增
* judge_decreasing ：判断一个list是否单调递减
* binning_var ：特征分箱，计算iv
* binning_trim ：调整单调后的分箱，计算IV
* forward_corr_delete ：相关性筛选
* vif_delete ：多重共线性筛选
* forward_pvalue_delete ：显著性筛选，前向逐步回归
* backward_pvalue_delete ：显著性筛选，后向逐步回归
* forward_delete_coef ：系数一致筛选
* get_map_df ：得到特征woe映射集合表
* var_mapping：特征映射
* plot_roc ：绘制roc
* plot_model_ks : 绘制ks
* cal_scale ：计算分数校准的A，B值，基础分
* get_score_map：得到特征score的映射集合表
* plot_score_hist ：绘制好坏用户得分分布图
* cal_psi ：计算psi
* get_scorecard_model ：评分卡建模
