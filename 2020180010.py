# -*- coding: utf-8 -*-
################################
# 此段代码为Windows 命令行输出编码格式设置，若在Mac运行无需此代码
################################
import io
import sys
# import urllib.request
import platform
if(platform.system()=='Windows'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf8') #改变标准输出的默认编码

##############################
#请使用最大信息增益算法为课件73页的数据构建决策树，要求代码运行能够直接打印出决策树。代码只能包含一个文件，文件名为学号_姓名.py。编程环境要求如下：
#	Python 3.6.10
#	python standard library
#	numpy == 1.16.2
#	scipy == 1.2.1
#	pandas == 0.24.2
##############################
from math import log
# 创建数据集
def DB():
    # 属性集
    attr_data = []
    # 训练集
    train_data = []

    attr_data.append(["男","运动员","70后","光头","80后","离婚","选秀","篮球","内地","演员"])

    train_data.append(["姚明","是","是","否","否","是","否","否","是","是","否"])
    train_data.append(["刘翔","是","是","否","否","是","是","否","否","是","否"])
    train_data.append(["科比","是","是","是","是","否","否","否","是","否","否"])
    train_data.append(["c罗","是","是","否","否","是","否","否","否","否","否"])
    train_data.append(["刘德华","是","否","否","否","否","否","否","否","否","是"])
    train_data.append(["毛不易","是","否","否","否","否","否","是","否","是","否"])
    train_data.append(["周杰伦","是","否","是","否","否","否","否","否","否","是"])
    train_data.append(["黄渤","是","否","是","否","否","否","否","否","是","是"])
    train_data.append(["徐峥","是","否","是","是","否","否","否","否","是","是"])
    train_data.append(["张怡宁","否","是","否","否","是","否","否","否","是","否"])
    train_data.append(["郎平","否","是","否","否","否","是","否","否","是","否"])
    train_data.append(["朱婷","否","是","否","否","否","否","否","否","是","否"])
    train_data.append(["杨超越","否","否","否","否","否","否","是", "否","是","是"])
    train_data.append(["杨幂","否","否","否","否","是","是","否","否","是","是"])
    train_data.append(["邓紫棋","否","否","否","否","否","否","否","否","否","否"])
    train_data.append(["徐佳莹","否","否","否","否","是","否","是","否","否","否"])
    train_data.append(["赵丽颖","否","否","否","否","是","否","否","否","是","是"])

    print("属性集：")
    print(attr_data)
    print("数据集：")
    print(train_data)

    return attr_data, train_data

# 如果数据集中的axis列，值为value，那么取出这一行，且去掉这一列，加入子数据集中
def split_data_set(data_set, axis, value):
    sub_data_set = []
    for line in data_set:
        if line[axis] == value:
            newline = line[:]
            del newline[axis]
            sub_data_set.append(newline)
    return sub_data_set

# 计算数组中重复的数
def count_data_rep(data_arr):
    b = set(data_arr)
    dic={}
    for each_b in b:
        count = 0
        for each_a in data_arr:
            if each_b == each_a:
                count += 1
        dic[each_b]=count
    return dic


# 计算随机变量熵的值 -∑PilogPi
def math_entropy(cur_num,all_num):
    # 概率
    P=float(cur_num) / all_num
    # -PilogPi
    entropy=P * log(P,2)
    return entropy

# 计算信息熵
def info_entropy(data_set):
    # 数据集样本条数
    num = len(data_set)
    # 标签计数字典
    count = {}
    for i in data_set:
        # 获取样本的标签
        current_label = i[0]
        # 如果当前标签不在计数字典里，则初始化
        if current_label not in count.keys():
            count[current_label] = 0
        count[current_label] += 1

    # 信息熵初始化
    entropy = 0.0
    # 计算信息熵:-∑PilogPi
    for key in count.keys():
        entropy+=math_entropy(count[key],num)
        print("人名", key, "概率", float(count[key]) / num, "信息熵",math_entropy(count[key],num))

    print("Ent(D)=", entropy)
    return entropy

# 计算最大信息增益
def max_entropy(data_set):
    # 属性个数
    feature_num = len(data_set[0]) - 1  # 为啥-1
    # 根节点的信息熵
    print("\n根节点信息熵计算：")
    root_node = info_entropy(data_set)
    # 最大化信息增益
    maxinfo_gain = 0.0
    # 最大信息增益对应的索引
    max_index = -1
    # 包含的选项
    option_list=["是","否"]
    # axis为列号
    for axis in range(feature_num):
        new_ent = 0.0
        # 将 是与否的个数提取出来
        option_dic=count_data_rep(data_set[axis])
        # value为列可能的取值，在20问读心游戏里为：是/否
        for value in option_list:
            sub_data_set = split_data_set(data_set, axis, value)
            # 计算条件概率
            P = float(option_dic[value]) / len(data_set)
            # 计算条件熵
            temp = P * info_entropy(sub_data_set) # Ent(a)
            new_ent += temp
            print("属性值", value, "条件概率", P, "条件熵", temp)  
        print("条件熵总和为：", new_ent)
        # 计算信息增益
        info_gain = root_node - new_ent # Gain(D,A)
        print("信息增益为：", info_gain)
        if info_gain > maxinfo_gain:
            maxinfo_gain = info_gain
            max_index = axis
    return max_index

# 递归方式创建决策树
# data_set为当前数据集，attr为剩余的还未用过的属性集
def create_tree(data_set, attr):
    # 获取标签，data_set的最后一列
    # note: 因为递归，data_set会改变，每次要从新的data_set中获取
    true_labels = []
    for line in data_set:
        true_labels.append(line[-1])
    # 递归终止条件：标签全部相同（例如全是‘姚明’），或者只有一个标签，没必要再进行下去，直接返回这个标签
    if true_labels.count(true_labels[0]) == len(true_labels):
        return true_labels[0]
    # 递归终止条件：数据集中只有一个属性（只有一列，标签列，实际没有意义），没必要继续下去，直接返回列中相同值最多的标签
    if len(data_set[0]) == 1:
        return max(true_labels, key=true_labels.count)
    # 从当前数据集和剩余的属性集attr中获取最优属性(信息增益最大)的索引和属性名
    best_index = max_entropy(data_set)
    best_attr = attr[best_index]
    print("最好的属性为：", attr[best_index])
    print("*********************************************************")
    # 开始创建决策树
    # 初始化字典，创建根节点，第一个属性对应的也是一个字典
    root = {best_attr: {}}
    # 获取最优属性对应的列并去重
    best_row = []
    for line in data_set:
        best_row.append(line[best_index])
    unique_row = set(best_row)
    # value为列可能的取值，在20问读心游戏里为：是/否
    for value in unique_row:
        # 新建子属性集合，并且将用完的属性从属性集中删除
        # note: 最好是不要改变attr的内容，新建一个sub_attr拷贝attr，对sub_attr做删除操作
        sub_attr = attr[:]
        del sub_attr[best_index]
        # 递归构造决策树
        # note: root[best_attr]是一个字典
        #  根据当前best_attr属性的所有可能的取值value进行构造，在20问读心游戏里为：是/否
        #  也就是说构造出的决策树是二叉树
        root[best_attr][value] = create_tree(split_data_set(data_set, best_index, value), sub_attr)
    return root

attr_data,train_data=DB()
attr_copy = train_data.copy()
# info_entropy(train_data)
decision_tree = create_tree(attr_data, attr_copy)