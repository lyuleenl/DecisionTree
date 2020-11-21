# -*- coding: utf-8 -*-
################################
# 此段代码为Windows 命令行输出编码格式设置，若在Mac运行无需此代码
################################
import io
import sys
import platform
if(platform.system()=='Windows'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf8') #改变标准输出的默认编码

##############################
#请使用最大信息增益算法为课件73页的数据构建决策树，要求代码运行能够直接打印出决策树。代码只能包含一个文件，文件名为学号_姓名.py。编程环境要求如下：
#	Python 3.6.10 当前环境3.6.8
#	python standard library
#	numpy == 1.16.2
#	scipy == 1.2.1
#	pandas == 0.24.2
#	networkx == 2.4
#	graphviz==0.13.2
#	matplotlib==3.2.1
##############################
from math import log
import matplotlib.pyplot as pyplot
import numpy as np

# 显示中文标签
pyplot.rcParams['font.sans-serif'] = 'SimHei'
# 设置决策树样式
# boxstyle为文本框的类型，sawtooth是锯齿形,round4是圆一点的四边形，fc是边框线粗细
# 属性节点
decision_node = dict(boxstyle="sawtooth", fc="0.8")
# 叶子节点
leaf_node = dict(boxstyle="round4", fc="0.8")
# arrowstyle是树的线为箭头样式
arrow = dict(arrowstyle="<-")
# 创建数据集
def DB():
    # 属性集
    attr_data=np.array(["人物","男","运动员","70后","光头","80后","离婚","选秀","篮球","内地","演员"])
    # 训练集
    train_data=np.array([["姚明","是","是","否","否","是","否","否","是","是","否"],
                        ["刘翔","是","是","否","否","是","是","否","否","是","否"],
                        ["科比","是","是","是","是","否","否","否","是","否","否"],
                        ["c罗","是","是","否","否","是","否","否","否","否","否"],
                        ["刘德华","是","否","否","否","否","否","否","否","否","是"],
                        ["毛不易","是","否","否","否","否","否","是","否","是","否"],
                        ["周杰伦","是","否","是","否","否","否","否","否","否","是"],
                        ["黄渤","是","否","是","否","否","否","否","否","是","是"],
                        ["徐峥","是","否","是","是","否","否","否","否","是","是"],
                        ["张怡宁","否","是","否","否","是","否","否","否","是","否"],
                        ["郎平","否","是","否","否","否","是","否","否","是","否"],
                        ["朱婷","否","是","否","否","否","否","否","否","是","否"],
                        ["杨超越","否","否","否","否","否","否","是", "否","是","是"],
                        ["杨幂","否","否","否","否","是","是","否","否","是","是"],
                        ["邓紫棋","否","否","否","否","否","否","否","否","否","否"],
                        ["徐佳莹","否","否","否","否","是","否","是","否","否","否"],
                        ["赵丽颖","否","否","否","否","是","否","否","否","是","是"]])
    print("属性集：")
    print(attr_data)
    print("数据集：")
    print(train_data)

    return attr_data, train_data

# 删除数组的列
def get_data_clumn(data_set, clumn, value):
    sub_data_set=[]
    for line in data_set:
        if line[clumn] == value:
            newline = line[:]
            #del newline[clumn]
            newline=np.delete(newline,clumn)
            sub_data_set.append(newline)
    return np.array(sub_data_set)

# 计算数组中重复的元素
def count_data_rep(data_set,clumn):
    sub_data_set = data_set[:,clumn]
    dic={}
    b=set(sub_data_set)
    for each_b in b:
        count = 0
        for each_a in sub_data_set:
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
    for i in data_set[:,0]:
        # 获取样本的标签
        count[i] = 1

    # 信息熵初始化
    entropy = 0.0
    # 计算信息熵:-∑PilogPi
    for key in count.keys():
        entropy-=math_entropy(count[key],num)
        print("人名", key, "概率", float(count[key]) / num, "信息熵",math_entropy(count[key],num))

    print("Ent(D)=", entropy)
    return entropy

# 计算最大信息增益
def max_entropy(data_set):
    # 属性索引
    feature_index = len(data_set[0])
    # 根节点的信息熵
    print("\n根节点信息熵计算：")
    root_node = info_entropy(data_set)
    # 最大化信息增益
    maxinfo_gain = 0.0
    # 最大信息增益对应的索引
    max_index = -1
    # 包含的选项
    option_list=["是","否"]
    # clumn为列号
    for clumn in range(1,feature_index):
        new_ent = 0.0
        # 将 是与否的个数提取出来
        option_dic=count_data_rep(data_set,clumn)
        # value为列可能的取值，在20问读心游戏里为：是/否
        for value in option_list:
            if value not in option_dic:
                continue
            data_clumn = get_data_clumn(data_set, clumn, value)
            # 计算条件概率
            P = float(option_dic[value]) / len(data_set)
            # 计算条件熵  Ent(a)
            temp = P * info_entropy(data_clumn)
            new_ent += temp
            print("属性值", value, "条件概率", P, "条件熵", temp)
        print("条件熵总和为：", new_ent)
        # 计算信息增益  Gain(D,A)
        info_gain = root_node - new_ent
        print("信息增益为：", info_gain)
        if info_gain > maxinfo_gain:
            maxinfo_gain = info_gain
            max_index = clumn
    return max_index

# 创建决策树
def create_tree(data_set, attr):
    # 获取标签，data_set[0]
    true_labels = []
    for line in data_set:
        true_labels.append(line[0])
    #  到人名 递归终止
    if true_labels.count(true_labels[0]) == len(true_labels):
        return true_labels[0]
    # 遍历完终止
    if len(data_set[0]) == 1:
        return max(true_labels, key=true_labels.count)
    # 最大信息增益
    best_index = max_entropy(data_set)
    best_attr = attr[best_index]

    print("--------------------------------------")
    print("当前最大增益属性为：", attr[best_index])
    # 开始创建决策树
    root = {best_attr: {}}
    # 获取最优属性对应的列并去重
    best_row = np.array(data_set[:,best_index])
    # for line in data_set:
    #     best_row.append(line[best_index])
    unique_row = set(best_row)
    for value in unique_row:
        # 子属性集合 不改变原数组
        sub_attr = attr[:]
        # del sub_attr[best_index]
        sub_attr=np.delete(sub_attr,best_index)
        # 子节点递归
        root[best_attr][value] = create_tree(get_data_clumn(data_set, best_index, value), sub_attr)
    return root

# 获取树的层数
def get_depth(decision_tree):
    max_depth = 0
    # 将决策树dict的key转化为list并获取根结点属性名称
    root_attr = list(decision_tree.keys())[0]
    # 根据根节点属性获取子树
    sub_tree = decision_tree[root_attr]
    # 对子树字典所有的key，也就是root_attr所有的取值（在20问读心游戏里为：是/否）遍历
    for key in sub_tree.keys():
        # 如果是字典对象，说明还未到叶子，继续递归
        if isinstance(sub_tree[key], dict):
            depth = get_depth(sub_tree[key]) + 1
        # 如果不是字典对象，说明已经到达叶子，停止递归
        else:
            depth = 1
        # 判断深度是否大于最大深度
        if depth > max_depth:
            max_depth = depth
    return max_depth


# 获取树的叶子节点个数，也就是标签数
def get_leaf_num(decision_tree):
    num = 0
    # 将决策树dict的key转化为list并获取根结点属性名称
    root_attr = list(decision_tree.keys())[0]
    # 根据根节点属性获取子树
    sub_tree = decision_tree[root_attr]
    # 对子树字典所有的key，也就是root_attr所有的取值（在20问读心游戏里为：是/否）遍历
    for key in sub_tree.keys():
        # 如果是字典对象，说明还未到叶子，继续递归
        if isinstance(sub_tree[key], dict):
            num += get_leaf_num(sub_tree[key])
        # 如果不是字典对象，说明已经到达叶子，停止递归
        else:
            num += 1
    return num


# 创建图对象，初始化，画图
def create_plot(decision_tree):
    # 定义一个背景为白色的画布，并把画布清空
    fig = pyplot.figure(1, facecolor='white')
    fig.clf()
    # ax_prop为图形的样式，没有坐标轴标签
    ax_prop = dict(xticks=[], yticks=[])
    # 使用subplot为定义了一个图，一行一列一个图，
    # frameon=False代表没有矩形边框
    # note: 在python里，[函数名称].[变量名]相当于是全局变量
    # ax1相当于是图对象，在其它函数中使用
    create_plot.ax1 = pyplot.subplot(111, frameon=False, **ax_prop)
    # total_width和total_depth分别代表初始决策树的叶子节点数目和深度，不改变
    plot_tree.total_width = float(get_leaf_num(decision_tree))
    plot_tree.total_depth = float(get_depth(decision_tree))
    # 图的大小是长0~1，宽0~1
    # note: x_offset实质是每个叶子的x坐标的位置
    #  第一片叶子的x为0.5/叶子数目，因此初始的x_offset设为-0.5/叶子数目
    #  每次将x_offset+(1/x_offset)，也就是第一个叶子不紧贴边框，有0.5/叶子数目的内边距
    #  例如绘制3个叶子，坐标应为1/3、2/3、3/3
    #  但这样整个图形会偏右，因此将初始的x_offset设为-0.5/3
    #  这样的话，每个叶子向左移了0.5/3，坐标变成了0.5/3、1.5/3、2.5/3，就刚好让图形在正中间了
    #  初始的y_offset显然为1，也就是最高点，每下降一层将y_offset-(1/深度)即可
    plot_tree.x_offset = -0.5 / plot_tree.total_width
    plot_tree.y_offset = 1.0
    # 初始根节点位置为图形的正中间最上方，即(0.5, 1.0)
    # 初始节点文本为空，等待获取
    plot_tree(decision_tree, (0.5, 1.0), '')
    pyplot.show()


# 递归画出决策树
# parent_pos为父节点位置，也就是当前决策树根节点的父节点位置
# arrow_text为父节点指来的箭头上的内容（在20问读心游戏里为：是/否）
def plot_tree(decision_tree, parent_pos, arrow_text):
    # 获取当前决策树叶子数目
    # note: leaf_num与plot_tree.total_depth不同，前者针对的是当前的决策树，后者是针对的原来的整个决策树
    leaf_num = get_leaf_num(decision_tree)
    # 将决策树dict的key转化为list并获取根结点属性名称
    root_attr = list(decision_tree.keys())[0]
    # root_pos为当前决策树的根节点的位置
    # note: 计算方法为
    #  拆分为三部分：
    #  1. plot_tree.x_offset：初始的x偏移，基准值
    #  2. float(numLeafs) / 2.0 / plotTree.totalW：
    #       float(numLeafs) * (1 / plotTree.totalW)为该决策树所包含的所有叶子所占的横坐标宽度
    #       / 2.0就是这个宽度的中间点
    #  3. 0.5 / plotTree.totalW：因为x_offset初始有-0.5/plotTree.totalW的偏移
    #       导致该节点并不是在区域中点，而是向左有个0.5/plotTree.totalW偏移
    #       因此+0.5 / plotTree.totalW，使其位于区域正中
    #  最终的公式经过合并就是下式：
    root_pos = (plot_tree.x_offset + (1.0 + float(leaf_num)) / 2.0 / plot_tree.total_width, plot_tree.y_offset)
    # 画出由当前子决策树父节点指来的箭头和箭头上的文本（在20问读心游戏里为：是/否）以及箭头指向的当前决策树的根节点
    plot_arrow_text(root_pos, parent_pos, arrow_text)
    # 节点类型为决策类型decision_node，不是叶子
    plot_node(root_pos, parent_pos, decision_node, root_attr)
    # 根据根节点属性获取子树
    sub_tree = decision_tree[root_attr]
    # note: 每下降一层，将y_offset减1.0 / plot_tree.total_depth
    plot_tree.y_offset = plot_tree.y_offset - 1.0 / plot_tree.total_depth
    # 对子树字典所有的key，也就是root_attr所有的取值（在20问读心游戏里为：是/否）遍历
    for key in sub_tree.keys():
        # 如果是字典对象，说明还未到叶子，继续递归
        if isinstance(sub_tree[key], dict):
            # note: 子决策树为sub_tree[key]
            #  子决策树的父节点为当前决策树的根节点
            #  当前决策树指向子决策树的箭头上的文本为key，因为key不是字符串，要进行类型转换
            plot_tree(sub_tree[key], root_pos, str(key))
        # 如果不是字典对象，说明已经到达叶子，停止递归
        else:
            # 每到一个叶子，就把x_offset加1.0 / plot_tree.total_width
            plot_tree.x_offset = plot_tree.x_offset + 1.0 / plot_tree.total_width
            # 画出叶子、箭头
            # (plot_tree.x_offset, plot_tree.y_offset)刚好是叶子的坐标
            # root_pos为当前决策树的根节点坐标
            # 节点类型为叶子类型leaf_node
            # 因为是叶子，sub_tree[key]为字符串类型，也就是标签
            plot_node((plot_tree.x_offset, plot_tree.y_offset), root_pos, leaf_node, sub_tree[key])
            # 画出箭头上的文本
            # 当前决策树指向叶子的箭头上的文本为key，因为key不是字符串，要进行类型转换
            plot_arrow_text((plot_tree.x_offset, plot_tree.y_offset), root_pos, str(key))
    # note: 易错点，每次递归结束需要将y_offset加1.0 / plot_tree.total_depth，回到上一层
    plot_tree.y_offset = plot_tree.y_offset + 1.0 / plot_tree.total_depth

# 画节点和指向节点的箭头的函数
# root_pos为子节点的位置，也就是箭头指向的位置
# parent_pos为父节点的位置，也就是箭头尾部所在的位置
# node_type为节点类型，两种：决策节点（decision_node）和叶节点（leaf_node）
# node_text为要显示的文本，也就是节点的内容，即属性的名称（例如：男、运动员……）
def plot_node(root_pos, parent_pos, node_type, node_text):
    # note: annotate用于在图形上给数据添加文本注解，支持带箭头的划线工具
    #  参数如下：
    #  s：注释文本的内容
    #  xy：被注释的坐标点，二维元组形如(x,y)
    #  xytext：注释文本的坐标点，也就是文本写的地方，也是二维元组，默认与xy相同
    #  xycoords：被注释点的坐标系属性，axes fraction是以子绘图区左下角为参考，单位是百分比
    #  textcoords：注释文本的坐标系属性，默认与xycoords属性值相同
    #  va="center"，ha="center"表示注释的坐标以注释框的正中心为准，而不是注释框的左下角(v代表垂直方向，h代表水平方向)
    #  bbox是注释框的风格和颜色深度，fc越小，注释框的颜色越深，支持输入一个字典
    #  arrowprops：箭头的样式，字典型数据，在画图的开头定义了
    create_plot.ax1.annotate(node_text, xy=parent_pos, xycoords='axes fraction',
                             xytext=root_pos, textcoords='axes fraction',
                             va="center", ha="center", bbox=node_type, arrowprops=arrow)


# 画箭头上文本的函数
# root_pos为子节点的位置，也就是箭头指向的位置
# parent_pos为父节点的位置，也就是箭头尾部所在的位置
# text为要显示的文本，也就是箭头上写的内容，即属性的取值（在20问读心游戏里为：是/否）
def plot_arrow_text(root_pos, parent_pos, arrow_text):
    # 文本的位置应该处于箭头中间，也就是文本坐标=箭头头坐标+（箭头尾坐标-箭头头坐标）/2，因为箭头是向下指的
    x_mid = root_pos[0] + (parent_pos[0] - root_pos[0]) / 2.0
    y_mid = root_pos[1] + (parent_pos[1] - root_pos[1]) / 2.0
    create_plot.ax1.text(x_mid, y_mid, arrow_text, va="center", ha="center", rotation=30)


attr_data,train_data=DB()
attr_copy = attr_data.copy()
# info_entropy(train_data)
decision_tree = create_tree(train_data, attr_copy)
print("决策树结构为：")
print(decision_tree)
# 绘制决策树
print("绘制决策树……")
create_plot(decision_tree)
print("决策树的深度为：%d" % get_depth(decision_tree))
print("决策树叶子数目为：%d" % get_leaf_num(decision_tree))