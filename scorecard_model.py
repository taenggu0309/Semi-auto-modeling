#!/usr/bin/env python
# coding: utf-8


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import math 
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn import metrics 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier,_tree
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
warnings.filterwarnings('ignore')




def tree_split(df,col,target,max_bin,min_binpct,nan_value):
    """
    决策树分箱
    param:
        df -- 数据集 Dataframe
        col -- 分箱的字段名 string
        target -- 标签的字段名 string
        max_bin -- 最大分箱数 int
        min_binpct -- 箱体的最小占比 float
        nan_value -- 缺失的映射值 int/float
    return:
        split_list -- 分割点 list
    """
    miss_value_rate = df[df[col]==nan_value].shape[0]/df.shape[0]
    # 如果缺失占比小于5%，则直接对特征进行分箱
    if miss_value_rate<0.05:
        x = np.array(df[col]).reshape(-1,1)
        y = np.array(df[target])
        tree = DecisionTreeClassifier(max_leaf_nodes=max_bin,
                                  min_samples_leaf = min_binpct)
        tree.fit(x,y)
        thresholds = tree.tree_.threshold
        thresholds = thresholds[thresholds!=_tree.TREE_UNDEFINED]
        split_list = sorted(thresholds.tolist())
    # 如果缺失占比大于5%，则把缺失单独分为一箱，剩余部分再进行决策树分箱
    else:
        max_bin2 = max_bin-1
        x = np.array(df[~(df[col]==nan_value)][col]).reshape(-1,1)
        y = np.array(df[~(df[col]==nan_value)][target])
        tree = DecisionTreeClassifier(max_leaf_nodes=max_bin2,
                                  min_samples_leaf = min_binpct)
        tree.fit(x,y)
        thresholds = tree.tree_.threshold
        thresholds = thresholds[thresholds!=_tree.TREE_UNDEFINED]
        split_list = sorted(thresholds.tolist())
        split_list.insert(0,nan_value)
    
    return split_list




def quantile_split(df,col,target,max_bin,nan_value):
    """
    等频分箱
    param:
        df -- 数据集 Dataframe
        col -- 分箱的字段名 string
        target -- 标签的字段名 string
        max_bin -- 最大分箱数 int
        nan_value -- 缺失的映射值 int/float
    return:
        split_list -- 分割点 list
    """
    miss_value_rate = df[df[col]==nan_value].shape[0]/df.shape[0]
    
    # 如果缺失占比小于5%，则直接对特征进行分箱
    if miss_value_rate<0.05:
        bin_series,bin_cut = pd.qcut(df[col],q=max_bin,duplicates='drop',retbins=True)
        split_list = bin_cut.tolist()
        split_list.remove(split_list[0])
    # 如果缺失占比大于5%，则把缺失单独分为一箱，剩余部分再进行等频分箱
    else:
        df2 = df[~(df[col]==nan_value)]
        max_bin2 = max_bin-1
        bin_series,bin_cut = pd.qcut(df2[col],q=max_bin2,duplicates='drop',retbins=True)
        split_list = bin_cut.tolist()
        split_list[0] = nan_value
        
    split_list.remove(split_list[-1])
    
    # 当出现某个箱体只有好用户或只有坏用户时，进行前向合并箱体
    var_arr = np.array(df[col])
    target_arr = np.array(df[target])
    bin_trans = np.digitize(var_arr,split_list,right=True)
    var_tuple = [(x,y) for x,y in zip(bin_trans,target_arr)]
    
    delete_cut_list = []
    for i in set(bin_trans):
        target_list = [y for x,y in var_tuple if x==i]
        if target_list.count(1)==0 or target_list.count(0)==0:
            if i ==min(bin_trans):
                index=i
            else:
                index = i-1
            delete_cut_list.append(split_list[index])
    split_list = [x for x in split_list if x not in delete_cut_list]
    
    return split_list




def cal_woe(df,col,target,nan_value,cut=None):
    """
    计算woe
    param：
        df -- 数据集 Dataframe
        col -- 分箱的字段名 string
        target -- 标签的字段名 string
        nan_value -- 缺失的映射值 int/float
        cut -- 箱体分割点 list
    return:
        woe_list -- 每个箱体的woe list
    """
    total = df[target].count()
    bad = df[target].sum()
    good = total-bad
    
    bucket = pd.cut(df[col],cut)
    group = df.groupby(bucket)
        
    bin_df = pd.DataFrame()
    bin_df['total'] = group[target].count()
    bin_df['bad'] = group[target].sum()
    bin_df['good'] = bin_df['total'] - bin_df['bad']
    bin_df['badattr'] = bin_df['bad']/bad
    bin_df['goodattr'] = bin_df['good']/good
    bin_df['woe'] = np.log(bin_df['badattr']/bin_df['goodattr'])
    # 当cut里有缺失映射值时，说明是把缺失单独分为一箱的，后续在进行调成单调分箱时
    # 不考虑缺失的箱，故将缺失映射值剔除
    if nan_value in cut:
        woe_list = bin_df['woe'].tolist()[1:]
    else:
        woe_list = bin_df['woe'].tolist()
    return woe_list




def monot_trim(df,col,target,nan_value,cut=None):
    """
    woe调成单调递减或单调递增
    param:
        df -- 数据集 Dataframe
        col -- 分箱的字段名 string
        target -- 标签的字段名 string
        nan_value -- 缺失的映射值 int/float
        cut -- 箱体分割点 list
    return:
        new_cut -- 调整后的分割点 list
    """
    woe_lst = cal_woe(df,col,target,nan_value,cut = cut)
    # 若第一个箱体大于0，说明特征整体上服从单调递减
    if woe_lst[0]>0:
        while not judge_decreasing(woe_lst):
            # 找出哪几个箱不服从单调递减的趋势
            judge_list = [x>y for x, y in zip(woe_lst, woe_lst[1:])]
            # 用前向合并箱体的方式，找出需要剔除的分割点的索引，如果有缺失映射值，则索引+1
            if nan_value in cut:
                index_list = [i+2 for i,j in enumerate(judge_list) if j==False]
            else:
                index_list = [i+1 for i,j in enumerate(judge_list) if j==False]
            new_cut = [j for i,j in enumerate(cut) if i not in index_list]
            woe_lst = cal_woe(df,col,target,nan_value,cut = new_cut)
    # 若第一个箱体小于0，说明特征整体上服从单调递增
    elif woe_lst[0]<0:
        while not judge_increasing(woe_lst):
            # 找出哪几个箱不服从单调递增的趋势
            judge_list = [x<y for x, y in zip(woe_lst, woe_lst[1:])]
            # 用前向合并箱体的方式，找出需要剔除的分割点的索引，如果有缺失映射值，则索引+1
            if nan_value in cut:
                index_list = [i+2 for i,j in enumerate(judge_list) if j==False]
            else:
                index_list = [i+1 for i,j in enumerate(judge_list) if j==False]
            new_cut = [j for i,j in enumerate(cut) if i not in index_list]
            woe_lst = cal_woe(df,col,target,nan_value,cut = new_cut)
    
    return new_cut




def judge_increasing(L):
    """
    判断一个list是否单调递增
    """
    return all(x<y for x, y in zip(L, L[1:]))

def judge_decreasing(L):
    """
    判断一个list是否单调递减
    """
    return all(x>y for x, y in zip(L, L[1:]))




def binning_var(df,col,target,bin_type='dt',max_bin=5,min_binpct=0.05,nan_value=-999):
    """
    特征分箱，计算iv
    param:
        df -- 数据集 Dataframe
        col -- 分箱的字段名 string
        target -- 标签的字段名 string
        bin_type -- 分箱方式 默认是'dt',还有'quantile'(等频分箱)
        max_bin -- 最大分箱数 int
        min_binpct -- 箱体的最小占比 float
        nan_value -- 缺失映射值 int/float
    return:
        bin_df -- 特征的分箱明细表 Dataframe
        cut -- 分割点 list
    """
    total = df[target].count()
    bad = df[target].sum()
    good = total-bad
    
    # 离散型特征分箱,直接根据类别进行groupby
    if df[col].dtype == np.dtype('object') or df[col].dtype == np.dtype('bool') or df[col].nunique()<=max_bin:
        group = df.groupby([col],as_index=True)
        bin_df = pd.DataFrame()

        bin_df['total'] = group[target].count()
        bin_df['totalrate'] = bin_df['total']/total
        bin_df['bad'] = group[target].sum()
        bin_df['badrate'] = bin_df['bad']/bin_df['total']
        bin_df['good'] = bin_df['total'] - bin_df['bad']
        bin_df['goodrate'] = bin_df['good']/bin_df['total']
        bin_df['badattr'] = bin_df['bad']/bad
        bin_df['goodattr'] = (bin_df['total']-bin_df['bad'])/good
        bin_df['woe'] = np.log(bin_df['badattr']/bin_df['goodattr'])
        bin_df['bin_iv'] = (bin_df['badattr']-bin_df['goodattr'])*bin_df['woe']
        bin_df['IV'] = bin_df['bin_iv'].sum()
        cut = df[col].unique().tolist()
    # 连续型特征的分箱
    else:
        if bin_type=='dt':
            cut = tree_split(df,col,target,max_bin=max_bin,min_binpct=min_binpct,nan_value=nan_value)
        elif bin_type=='quantile':
            cut = quantile_split(df,col,target,max_bin=max_bin,nan_value=nan_value)
        cut.insert(0,float('-inf'))
        cut.append(float('inf'))
        
        bucket = pd.cut(df[col],cut)
        group = df.groupby(bucket)
        bin_df = pd.DataFrame()

        bin_df['total'] = group[target].count()
        bin_df['totalrate'] = bin_df['total']/total
        bin_df['bad'] = group[target].sum()
        bin_df['badrate'] = bin_df['bad']/bin_df['total']
        bin_df['good'] = bin_df['total'] - bin_df['bad']
        bin_df['goodrate'] = bin_df['good']/bin_df['total']
        bin_df['badattr'] = bin_df['bad']/bad
        bin_df['goodattr'] = (bin_df['total']-bin_df['bad'])/good
        bin_df['woe'] = np.log(bin_df['badattr']/bin_df['goodattr'])
        bin_df['bin_iv'] = (bin_df['badattr']-bin_df['goodattr'])*bin_df['woe']
        bin_df['IV'] = bin_df['bin_iv'].sum()
        
    return bin_df,cut




def binning_trim(df,col,target,cut=None,right_border=True):
    """
    调整单调后的分箱，计算IV
    param:
        df -- 数据集 Dataframe
        col -- 分箱的字段名 string
        target -- 标签的字段名 string
        cut -- 分割点 list
        right_border -- 箱体的右边界是否闭合 bool
    return:
        bin_df -- 特征的分箱明细表 Dataframe
    """
    total = df[target].count()
    bad = df[target].sum()
    good = total - bad
    bucket = pd.cut(df[col],cut,right=right_border)
    
    group = df.groupby(bucket)
    bin_df = pd.DataFrame()
    bin_df['total'] = group[target].count()
    bin_df['totalrate'] = bin_df['total']/total
    bin_df['bad'] = group[target].sum()
    bin_df['badrate'] = bin_df['bad']/bin_df['total']
    bin_df['good'] = bin_df['total'] - bin_df['bad']
    bin_df['goodrate'] = bin_df['good']/bin_df['total']
    bin_df['badattr'] = bin_df['bad']/bad
    bin_df['goodattr'] = (bin_df['total']-bin_df['bad'])/good
    bin_df['woe'] = np.log(bin_df['badattr']/bin_df['goodattr'])
    bin_df['bin_iv'] = (bin_df['badattr']-bin_df['goodattr'])*bin_df['woe']
    bin_df['IV'] = bin_df['bin_iv'].sum()
    
    return bin_df




def forward_corr_delete(df,col_list):
    """
    相关性筛选,设定的阈值为0.65
    param:
        df -- 数据集 Dataframe
        col_list -- 需要筛选的特征集合,需要提前按IV值从大到小排序好 list
    return:
        select_corr_col -- 筛选后的特征集合 list
    """
    corr_list=[]
    corr_list.append(col_list[0])
    delete_col = []
    # 根据IV值的大小进行遍历
    for col in col_list[1:]:
        corr_list.append(col)
        corr = df.loc[:,corr_list].corr()
        corr_tup = [(x,y) for x,y in zip(corr[col].index,corr[col].values)]
        corr_value = [y for x,y in corr_tup if x!=col]
        # 若出现相关系数大于0.65，则将该特征剔除
        if len([x for x in corr_value if abs(x)>=0.65])>0:
            delete_col.append(col)
    select_corr_col = [x for x in col_list if x not in delete_col]
    return select_corr_col




def vif_delete(df,list_corr):
    """
    多重共线性筛选
    param:
        df -- 数据集 Dataframe
        list_corr -- 相关性筛选后的特征集合，按IV值从大到小排序 list
    return:
        col_list -- 筛选后的特征集合 list
    """
    col_list = list_corr.copy()
    # 计算各个特征的方差膨胀因子
    vif_matrix=np.matrix(df[col_list])
    vifs_list=[variance_inflation_factor(vif_matrix,i) for i in range(vif_matrix.shape[1])]
    # 筛选出系数>10的特征
    vif_high = [x for x,y in zip(col_list,vifs_list) if y>10]
    
    # 根据IV从小到大的顺序进行遍历
    if len(vif_high)>0:
        for col in reversed(vif_high):
            col_list.remove(col)
            vif_matrix=np.matrix(df[col_list])
            vifs=[variance_inflation_factor(vif_matrix,i) for i in range(vif_matrix.shape[1])]
            # 当系数矩阵里没有>10的特征时，循环停止
            if len([x for x in vifs if x>10])==0:
                break
    return col_list




def forward_pvalue_delete(x,y):
    """
    显著性筛选，前向逐步回归
    param:
        x -- 特征数据集,woe转化后，且字段顺序按IV值从大到小排列 Dataframe
        y -- 标签列 Series
    return:
        pvalues_col -- 筛选后的特征集合 list
    """
    col_list = x.columns.tolist()
    pvalues_col=[]
    # 按IV值逐个引入模型
    for col in col_list:
        pvalues_col.append(col)
        # 每引入一个特征就做一次显著性检验
        x_const = sm.add_constant(x.loc[:,pvalues_col])
        sm_lr = sm.Logit(y,x_const)
        sm_lr = sm_lr.fit()
        pvalue = sm_lr.pvalues[col]
        # 当引入的特征P值>=0.05时，则剔除，原先满足显著性检验的则保留，不再剔除
        if pvalue>=0.05:
            pvalues_col.remove(col)
    return pvalues_col




def backward_pvalue_delete(x,y):
    """
    显著性筛选，后向逐步回归
    param:
        x -- 特征数据集,woe转化后，且字段顺序按IV值从大到小排列 Dataframe
        y -- 标签列 Series
    return:
        pvalues_col -- 筛选后的特征集合 list
    """
    x_c = x.copy()
    # 所有特征引入模型，做显著性检验
    x_const = sm.add_constant(x_c)
    sm_lr = sm.Logit(y,x_const).fit()
    pvalue_tup = [(i,j) for i,j in zip(sm_lr.pvalues.index,sm_lr.pvalues.values)][1:]
    delete_count = len([i for i,j in pvalue_tup if j>=0.05])
    # 当有P值>=0.05的特征时，执行循环
    while delete_count>0:
        # 按IV值从小到大的顺序依次逐个剔除
        remove_col = [i for i,j in pvalue_tup if j>=0.05][-1]
        del x_c[remove_col]
        # 每次剔除特征后都要重新做显著性检验，直到入模的特征P值都小于0.05
        x2_const = sm.add_constant(x_c)
        sm_lr2 = sm.Logit(y,x2_const).fit()
        pvalue_tup2 = [(i,j) for i,j in zip(sm_lr2.pvalues.index,sm_lr2.pvalues.values)][1:]
        delete_count = len([i for i,j in pvalue_tup2 if j>=0.05])
        
    pvalues_col = x_c.columns.tolist()
    
    return pvalues_col




def forward_delete_coef(x,y):
    """
    系数一致筛选
    param:
        x -- 特征数据集,woe转化后，且字段顺序按IV值从大到小排列 Dataframe
        y -- 标签列 Series
    return:
        coef_col -- 筛选后的特征集合 list
    """
    col_list = list(x.columns)
    coef_col = []
    # 按IV值逐个引入模型，输出系数
    for col in col_list:
        coef_col.append(col)
        x2 = x.loc[:,coef_col]
        sk_lr = LogisticRegression(random_state=0).fit(x2,y)
        coef_dict = {k:v for k,v in zip(coef_col,sk_lr.coef_[0])}
        # 当引入特征的系数为负，则将其剔除
        if coef_dict[col]<0:
            coef_col.remove(col)

    return coef_col




def get_map_df(bin_df_list):
    """
    得到特征woe映射集合表
    param:
        bin_df_list -- 每个特征的woe映射表 list
    return:
        map_merge_df -- 特征woe映射集合表 Dataframe
    """
    map_df_list=[]
    for dd in bin_df_list:
        # 添加特征名列
        map_df = dd.reset_index().assign(col=dd.index.name).rename(columns={dd.index.name:'bin'})
        # 将特征名列移到第一列，便于查看
        temp1 = map_df['col']
        temp2 = map_df.iloc[:,:-1]
        map_df2 = pd.concat([temp1,temp2],axis=1)
        map_df_list.append(map_df2)
        
    map_merge_df = pd.concat(map_df_list,axis=0)
    
    return map_merge_df




def var_mapping(df,map_df,var_map,target):
    """
    特征映射
    param:
        df -- 原始数据集 Dataframe
        map_df -- 特征映射集合表 Dataframe
        var_map -- map_df里映射的字段名，如"woe","score" string
        target -- 标签字段名 string
    return:
        df2 -- 映射后的数据集 Dataframe
    """
    df2 = df.copy()
    # 去掉标签字段，遍历特征
    for col in df2.drop([target],axis=1).columns:
        x = df2[col]
        # 找到特征的映射表
        bin_map = map_df[map_df.col==col]
        # 新建一个映射array，填充0
        bin_res = np.array([0]*x.shape[0],dtype=float)
        for i in bin_map.index:
            # 每个箱的最小值和最大值
            lower = bin_map['min_bin'][i]
            upper = bin_map['max_bin'][i]
            # 对于类别型特征，每个箱的lower和upper时一样的
            if lower == upper:
                x1 = x[np.where(x == lower)[0]]
            # 连续型特征，左开右闭
            else:
                x1 = x[np.where((x>lower)&(x<=upper))[0]]
            mask = np.in1d(x,x1)
            # 映射array里填充对应的映射值
            bin_res[mask] = bin_map[var_map][i]
        bin_res = pd.Series(bin_res,index=x.index)
        bin_res.name = x.name
        # 将原始值替换为映射值
        df2[col] = bin_res
    return df2




def plot_roc(y_label,y_pred):
    """
    绘制roc曲线
    param:
        y_label -- 真实的y值 list/array
        y_pred -- 预测的y值 list/array
    return:
        roc曲线
    """
    tpr,fpr,threshold = metrics.roc_curve(y_label,y_pred) 
    AUC = metrics.roc_auc_score(y_label,y_pred) 
    fig = plt.figure(figsize=(6,4))
    ax = fig.add_subplot(1,1,1)
    ax.plot(tpr,fpr,color='blue',label='AUC=%.3f'%AUC) 
    ax.plot([0,1],[0,1],'r--')
    ax.set_ylim(0,1)
    ax.set_xlim(0,1)
    ax.set_title('ROC')
    ax.legend(loc='best')
    return plt.show(ax)




def plot_model_ks(y_label,y_pred):
    """
    绘制ks曲线
    param:
        y_label -- 真实的y值 list/array
        y_pred -- 预测的y值 list/array
    return:
        ks曲线
    """
    pred_list = list(y_pred) 
    label_list = list(y_label)
    total_bad = sum(label_list)
    total_good = len(label_list)-total_bad 
    items = sorted(zip(pred_list,label_list),key=lambda x:x[0]) 
    step = (max(pred_list)-min(pred_list))/200 
    
    pred_bin=[]
    good_rate=[] 
    bad_rate=[] 
    ks_list = [] 
    for i in range(1,201): 
        idx = min(pred_list)+i*step 
        pred_bin.append(idx) 
        label_bin = [x[1] for x in items if x[0]<idx] 
        bad_num = sum(label_bin)
        good_num = len(label_bin)-bad_num  
        goodrate = good_num/total_good 
        badrate = bad_num/total_bad
        ks = abs(goodrate-badrate) 
        good_rate.append(goodrate)
        bad_rate.append(badrate)
        ks_list.append(ks)
    
    fig = plt.figure(figsize=(6,4))
    ax = fig.add_subplot(1,1,1)
    ax.plot(pred_bin,good_rate,color='green',label='good_rate')
    ax.plot(pred_bin,bad_rate,color='red',label='bad_rate')
    ax.plot(pred_bin,ks_list,color='blue',label='good-bad')
    ax.set_title('KS:{:.3f}'.format(max(ks_list)))
    ax.legend(loc='best')
    return plt.show(ax)




def cal_scale(score,odds,PDO,model):
    """
    计算分数校准的A，B值，基础分
    param:
        odds：设定的坏好比 float
        score: 在这个odds下的分数 int
        PDO: 好坏翻倍比 int
        model:模型
    return:
        A,B,base_score(基础分)
    """
    B = 20/(np.log(odds)-np.log(2*odds))
    A = score-B*np.log(odds)
    base_score = A+B*model.intercept_[0]
    return A,B,base_score




def get_score_map(woe_df,coe_dict,B):
    """
    得到特征score的映射集合表
    param:
        woe_df -- woe映射集合表 Dataframe
        coe_dict -- 系数对应的字典
    return:
        score_df -- score的映射集合表 Dataframe
    """
    scores=[]
    for cc in woe_df.col.unique():
        woe_list = woe_df[woe_df.col==cc]['woe'].tolist()
        coe = coe_dict[cc]
        score = [round(coe*B*w,0) for w in woe_list]
        scores.extend(score)
    woe_df['score'] = scores
    score_df = woe_df.copy()
    return score_df




def plot_score_hist(df,target,score_col,title,plt_size=None):
    """
    绘制好坏用户得分分布图
    param:
        df -- 数据集 Dataframe
        target -- 标签字段名 string
        score_col -- 模型分的字段名 string
        plt_size -- 绘图尺寸 tuple
        title -- 图表标题 string
    return:
        好坏用户得分分布图
    """    
    plt.figure(figsize=plt_size)
    plt.title(title)
    x1 = df[df[target]==1][score_col]
    x2 = df[df[target]==0][score_col]
    sns.kdeplot(x1,shade=True,label='bad',color='hotpink')
    sns.kdeplot(x2,shade=True,label='good',color ='seagreen')
    plt.legend()
    return plt.show()




def cal_psi(df1,df2,col,bin_num=5):
    """
    计算psi
    param:
        df1 -- 数据集A Dataframe
        df2 -- 数据集B Dataframe
        col -- 字段名 string
        bin_num -- 连续型特征的分箱数 默认为5
    return:
        psi float
        bin_df -- psi明细表 Dataframe
    """
    # 对于离散型特征直接根据类别进行分箱，分箱逻辑以数据集A为准
    if df1[col].dtype == np.dtype('object') or df1[col].dtype == np.dtype('bool') or df1[col].nunique()<=bin_num:
        bin_df1 = df1[col].value_counts().to_frame().reset_index().rename(columns={'index':col,col:'total_A'})
        bin_df1['totalrate_A'] = bin_df1['total_A']/df1.shape[0]
        bin_df2 = df2[col].value_counts().to_frame().reset_index().rename(columns={'index':col,col:'total_B'})
        bin_df2['totalrate_B'] = bin_df2['total_B']/df2.shape[0]
    else:
        # 这里采用的是等频分箱
        bin_series,bin_cut = pd.qcut(df1[col],q=bin_num,duplicates='drop',retbins=True)
        bin_cut[0] = float('-inf')
        bin_cut[-1] = float('inf')
        bucket1 = pd.cut(df1[col],bins=bin_cut)
        group1 = df1.groupby(bucket1)
        bin_df1=pd.DataFrame()
        bin_df1['total_A'] = group1[col].count()
        bin_df1['totalrate_A'] = bin_df1['total_A']/df1.shape[0]
        bin_df1 = bin_df1.reset_index()

        bucket2 = pd.cut(df2[col],bins=bin_cut)
        group2 = df2.groupby(bucket2)
        bin_df2=pd.DataFrame()
        bin_df2['total_B'] = group2[col].count()
        bin_df2['totalrate_B'] = bin_df2['total_B']/df2.shape[0]
        bin_df2 = bin_df2.reset_index()
    # 计算psi
    bin_df = pd.merge(bin_df1,bin_df2,on=col)
    bin_df['a'] = bin_df['totalrate_B'] - bin_df['totalrate_A']
    bin_df['b'] = np.log(bin_df['totalrate_B']/bin_df['totalrate_A'])
    bin_df['Index'] = bin_df['a']*bin_df['b']
    bin_df['PSI'] = bin_df['Index'].sum()
    bin_df = bin_df.drop(['a','b'],axis=1)
    
    psi =bin_df.PSI.iloc[0]
    
    return psi,bin_df


def get_scorecard_model(train_data,test_data,target,nan_value=-999,score=400,odds=999/1,pdo=20):
    """
    评分卡建模
    param:
        train_data -- 训练数据集，预处理好的 Dataframe
        test_data -- 测试数据集，预处理好的 Dataframe
        target -- 标签字段名 string
        nan_value -- 缺失的映射值 int 默认-999
        odds -- 设定的坏好比 float 默认999/1
        score -- 在这个odds下的分数 int 默认400
        PDO -- 好坏翻倍比 int 默认20
    return:
        lr_model -- lr模型
        score_map_df -- woe,score映射集合表 Dataframe
        valid_score -- 验证集模型分表 Dataframe
        test_score -- 测试集模型分表 Dataframe
    """
    # psi筛选，剔除psi大于0.25以上的特征
    all_col = [x for x in train_data.columns if x!=target]
    psi_tup = []
    for col in all_col:
        psi,psi_bin_df = cal_psi(train_data,test_data,col)
        psi_tup.append((col,psi))
    psi_delete = [x for x,y in psi_tup if y>=0.25]
    train = train_data.drop(psi_delete,axis=1)
    print('psi筛选特征完成')
    print('-------------')

    # 特征分箱,默认用的是决策树分箱
    train_col = [x for x in train.columns if x!=target]
    bin_df_list=[]
    cut_list=[]
    for col in train_col:
        try:
            bin_df,cut = binning_var(train,col,target)
            bin_df_list.append(bin_df)
            cut_list.append(cut)
        except:
            pass
    print('特征分箱完成')
    print('-------------')
    
    # 剔除iv无限大的特征
    bin_df_list = [x for x in bin_df_list if x.IV.iloc[0]!=float('inf')]
    # 保存每个特征的分割点list
    cut_dict={}
    for dd,cc in zip(bin_df_list,cut_list):
        col = dd.index.name
        cut_dict[col] = cc
    # 将IV从大到小进行排序
    iv_col = [x.index.name for x in bin_df_list]
    iv_value = [x.IV.iloc[0] for x in bin_df_list]
    iv_sort = sorted(zip(iv_col,iv_value),key=lambda x:x[1],reverse=True)

    # iv筛选，筛选iv大于0.02的特征
    iv_select_col = [x for x,y in iv_sort if y>=0.02]
    print('iv筛选特征完成')
    print('-------------')
    # 特征分类
    cate_col = []
    num_col = []
    for col in iv_select_col:
        if train[col].dtype==np.dtype('object') or train[col].dtype==np.dtype('bool') or train[col].nunique()<=5:
            cate_col.append(col)
        else:
            num_col.append(col)

    #相关性筛选，相关系数阈值0.65
    corr_select_col = forward_corr_delete(train,num_col)
    print('相关性筛选完成')
    print('-------------')

    # 多重共线性筛选，系数阈值10
    vif_select_col = vif_delete(train,corr_select_col)
    print('多重共线性筛选完成')
    print('-------------')

    # 自动调整单调分箱
    trim_var_dict = {k:v for k,v in cut_dict.items() if k in vif_select_col}
    trim_bin_list=[]
    for col in trim_var_dict.keys():
        bin_cut = trim_var_dict[col]
        df_bin = [x for x in bin_df_list if x.index.name==col][0]
        woe_lst = df_bin['woe'].tolist()
        if not judge_decreasing(woe_lst) and not judge_increasing(woe_lst):
            monot_cut = monot_trim(train, col, target, nan_value=nan_value, cut=bin_cut)
            monot_bin_df = binning_trim(train, col, target, cut=monot_cut, right_border=True)
            trim_bin_list.append(monot_bin_df)
        else:
            trim_bin_list.append(df_bin)
    # 调整后的分箱再根据iv筛选一遍
    select_num_df = []
    for dd in trim_bin_list:
        if dd.IV.iloc[0]>=0.02:
            select_num_df.append(dd)
    print('自动调整单调分箱完成')
    print('-------------')
    
    # 连续型特征的woe映射集合表
    woe_map_num = get_map_df(select_num_df)
    woe_map_num['bin'] = woe_map_num['bin'].map(lambda x:str(x))
    woe_map_num['min_bin'] = woe_map_num['bin'].map(lambda x:x.split(',')[0][1:])
    woe_map_num['max_bin'] = woe_map_num['bin'].map(lambda x:x.split(',')[1][:-1])
    woe_map_num['min_bin'] = woe_map_num['min_bin'].map(lambda x:float(x))
    woe_map_num['max_bin'] = woe_map_num['max_bin'].map(lambda x:float(x))
    
    if len(cate_col)>0:
        bin_cate_list = [x for x in bin_df_list if x.index.name in cate_col]
        # 剔除woe不单调的离散形特征
        select_cate_df=[]
        for i,dd in enumerate(bin_cate_list):
            woe_lst = dd['woe'].tolist()
            if judge_decreasing(woe_lst) or judge_increasing(woe_lst):
                select_cate_df.append(dd)
        # 离散型特征的woe映射集合表
        if len(select_cate_df)>0:
            woe_map_cate = get_map_df(select_cate_df)
            woe_map_cate['min_bin'] = list(woe_map_cate['bin'])
            woe_map_cate['max_bin'] = list(woe_map_cate['bin'])
            woe_map_df = pd.concat([woe_map_cate,woe_map_num],axis=0).reset_index(drop=True)
    else:
        woe_map_df = woe_map_num.reset_index(drop=True)

    # 显著性筛选，前向逐步回归
    select_all_col = woe_map_df['col'].unique().tolist()
    select_sort_col = [x for x,y in iv_sort if x in select_all_col]
    
    train2 = train.loc[:,select_sort_col+[target]].reset_index(drop=True)
    # woe映射
    train_woe = var_mapping(train2,woe_map_df,'woe',target)
    X = train_woe.loc[:,select_sort_col]
    y = train_woe[target]

    pvalue_select_col = forward_pvalue_delete(X,y)
    print('显著性筛选完成')
    print('-------------')

    # 剔除系数为负数的特征
    X2 = X.loc[:,pvalue_select_col]
    coef_select_col = forward_delete_coef(X2,y)

    # LR建模
    X3 = X2.loc[:,coef_select_col]
    x_train,x_valid,y_train,y_valid = train_test_split(X3,y,test_size=0.2,random_state=0)
    # 保存验证集的index
    valid_index = x_valid.index.tolist()
    
    lr_model = LogisticRegression(C=1.0).fit(x_train,y_train)
    print('建模完成')
    print('-------------')
    
    # 绘制验证集的auc，ks
    valid_pre = lr_model.predict_proba(x_valid)[:,1]
    print('验证集的AUC，KS:')
    plot_roc(y_valid,valid_pre)
    plot_model_ks(y_valid,valid_pre)
    
    woe_map_df2 = woe_map_df[woe_map_df.col.isin(coef_select_col)].reset_index(drop=True)
    # 绘制测试集的auc，ks
    test = test_data.loc[:,coef_select_col+[target]].reset_index(drop=True)
    test_woe = var_mapping(test,woe_map_df2,'woe',target)
    x_test = test_woe.drop([target],axis=1)
    y_test = test_woe[target]
    test_pre = lr_model.predict_proba(x_test)[:,1]
    print('测试集的AUC，KS:')
    plot_roc(y_test,test_pre)
    plot_model_ks(y_test,test_pre)
    
    # 评分转换
    A,B,base_score = cal_scale(score,odds,pdo,lr_model)
    score_map_df  = get_score_map(woe_map_df2,lr_model,B)
    # 分数映射
    valid_data = train2.iloc[valid_index,:].loc[:,coef_select_col+[target]].reset_index(drop=True)
    valid_score = var_mapping(valid_data,score_map_df,'score',target)
    valid_score['final_score'] = base_score
    for col in coef_select_col:
        valid_score['final_score']+=valid_score[col]
    valid_score['final_score'] = valid_score['final_score'].map(lambda x:int(x))
    
    test_score = var_mapping(test,score_map_df,'score',target)
    test_score['final_score'] = base_score
    for col in coef_select_col:
        test_score['final_score']+=test_score[col]
    test_score['final_score'] = test_score['final_score'].map(lambda x:int(x))
    print('评分转换完成')
    print('-------------')
    # 验证集的评分分布
    plot_score_hist(valid_score, target, 'final_score','valid_score',plt_size=(6,4))
    # 测试集的评分分布
    plot_score_hist(test_score, target, 'final_score','test_score',plt_size=(6,4))

    return lr_model,score_map_df,valid_score,test_score


