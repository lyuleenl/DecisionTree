
'''
姓名：于雷 学号：2020180010
作业题目：采用 RNN 为小 Baby 起个英文名字吧
 神经网络语言模型，即通过神经网络，计算一项自然语言（例如一条句子）
的出现概率，或者根据上文中的词推断句子中某个词的出现概率。例如，下图采
用了一个具有一个输入层、一个隐藏层和一个输出层的 MLP 网络，建模三元文法
模型：
本作业提供了 8000 多个英文名字，试训练一个环神经网络语言模型，进而
给定若干个开始字母，由语言模型自动生成后续的字母，直到生成一个名字的结
束符。从模型生成的名字中，挑选你最喜欢的一个，并采用一种可视化技术，绘
制出模型为每个时刻预测的前 5 个最可能的候选字母。
Tips：事实上，你也可以给定结尾的若干个字母，或者随意给出中间的若干
个字母，让 RNN 补全其它字母，从而得到一个完整的名字。因此，你也可以尝试
设计并实现一个这样的 RNN 模型，从模型生成的名字中，挑选你最喜欢的一个，
并采用可视化技术，绘制出模型为每个时刻预测的前 5 个最可能的候选字母。
参考：https://666wxy666.github.io/2020/07/03/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0-%E5%AE%9E%E9%AA%8C-RNN-%E4%B8%BAbaby%E8%B5%B7%E5%90%8D%E5%AD%97/
'''
from __future__ import unicode_literals, print_function, division

import glob
import math
import os
import random
import string
import time
from io import open

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import unicodedata
import numpy as np

all_letters = string.ascii_letters + " .,;'-"
img_letters=string.ascii_letters + "-' $"
n_letters = len(all_letters) + 1
criterion = nn.NLLLoss()
learning_rate = 0.0005  # 学习率
max_length = 20
path = "Data/*.txt"


# 将Unicode转为ASCII
# https://stackoverflow.com/a/518232/2809427
def Unicode_to_ASCII(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )


# 将行分割成数组, 并把 Unicode 转换成 ASCII 编码, 最后放进一个字典里 {category: [names ...]}
category_lines = {}
all_categories = []

for file_name in glob.glob(path):
    category = os.path.splitext(os.path.basename(file_name))[0]
    all_categories.append(category)
    line_list = open(file_name, encoding='utf-8').read().strip().split('\n')
    lines = [Unicode_to_ASCII(line) for line in line_list]
    category_lines[category] = lines

category_num = len(all_categories)

if category_num == 0:
    raise RuntimeError("未找到数据集！")

print("数据集类别：", category_num, all_categories)


'''
---构建RNN网络---
'''
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size

        self.i2h = nn.Linear(category_num + input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(category_num + input_size + hidden_size, output_size)
        self.o2o = nn.Linear(hidden_size + output_size, output_size)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, category, input, hidden):
        input_combined = torch.cat((category, input, hidden), 1)
        hidden = self.i2h(input_combined)
        output = self.i2o(input_combined)
        output_combined = torch.cat((hidden, output), 1)
        output = self.o2o(output_combined)
        output = self.dropout(output)
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)


# 生成用于输入的tensor
def generate_input_tensor(letters):
    tensor = torch.zeros(len(letters), 1, n_letters)
    for i in range(len(letters)):
        letter = letters[i]
        tensor[i][0][all_letters.find(letter)] = 1
    return tensor


# 生成目标tensor
def generate_target_Tensor(letters):
    letter_indexes = [all_letters.find(letters[i]) for i in range(1, len(letters))]
    letter_indexes.append(n_letters - 1)  # EOS
    return torch.LongTensor(letter_indexes)


# 类别的One-hot向量
def generate_category_tensor(category):
    i = all_categories.index(category)
    tensor = torch.zeros(1, category_num)
    tensor[0][i] = 1
    return tensor


# 利用辅助函数从数据集中获取随机的category和line
def random_pair():
    category = all_categories[random.randint(0, len(all_categories) - 1)]
    line = category_lines[category][random.randint(0, len(category_lines[category]) - 1)]
    return category, line


# 从随机的（category, line）对中生成 category, input, 和 target Tensor
def randomTrainingExample():
    category, line = random_pair()
    category_tensor = generate_category_tensor(category)
    input_tensor = generate_input_tensor(line)
    target_tensor = generate_target_Tensor(line)
    return category_tensor, input_tensor, target_tensor



'''
---训练RNN网络---
'''
def train(category_tensor, input_tensor, target_tensor):
    global output
    target_tensor.unsqueeze_(-1)
    hidden = rnn.init_hidden()

    rnn.zero_grad()

    loss = 0

    for i in range(input_tensor.size(0)):
        output, hidden = rnn(category_tensor, input_tensor[i], hidden)
        l = criterion(output, target_tensor[i])
        loss += l

    loss.backward()

    for p in rnn.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate)

    return output, loss.item() / input_tensor.size(0)


# 时间戳 /s
def time_format(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


# 从一个类中获取一个以start_letters开头的名字
def sample(category, start_letters='Al'):
    indices=[]
    values=[]
    with torch.no_grad():
        category_tensor = generate_category_tensor(category)
        input = generate_input_tensor(start_letters)
        hidden = rnn.init_hidden()

        output_name = start_letters

        for i in range(max_length):
            output, hidden = rnn(category_tensor, input[0], hidden)
            topv, topi_temp = output.topk(5)
            topi = topi_temp[0][0].item()
            indices.append(topi_temp)
            values.append(topv)
            if topi == n_letters - 1:
                break
            else:
                letter = all_letters[topi]
                output_name += letter
            input = generate_input_tensor(letter)
        print(output_name)
        return output_name, indices, values


# 可视化
def visualization(start_letters, output_name, indices, values):
    area = np.pi ** 2
    tmp_y = [v - v[0][4] + 0.5 for v in values]
    y = [v.numpy()[0] for v in tmp_y]
    x = [i for i in range(1, len(y) + 1)]
    for i in range(len(y[0])):
        tmp_y = [j[i] for j in y]
        plt.scatter(x, tmp_y, s=area * (5 - i), alpha=(5 - i) / 5)
        print(x, tmp_y)
        tmp_l = [k[0][i] for k in indices]
        for j in range(len(tmp_l)):
            letter = img_letters[tmp_l[j]]
            print(x[j], tmp_y[j], letter)
            plt.text(x[j], tmp_y[j], letter, ha='center', va='center', fontsize=10)
    plt.title("Input: " + start_letters + "   Output: " + output_name)
    plt.grid(linestyle='-.')
    plt.show()

'''
-------开始运行-------
'''
rnn = RNN(n_letters, 128, n_letters)

all_losses = []
total_loss = 0

# 超参数参考值
# n_its = 100000
# print_every = 5000
# plot_every = 500

print("请分别输入总迭代次数,打印精度,绘图精度(按空格分隔): ", end="")
n_its, print_every, plot_every = map(int, input().split())
print("开始训练......")
start = time.time()
print("[时间戳]\t\t百分比\t\t已迭代数\t\t损失")
for it in range(1, n_its + 1):
    output, loss = train(*randomTrainingExample())
    total_loss += loss

    if it % print_every == 0:
        print("[%10s]  <%3d%%>  %10d  %.4f" % (time_format(start), it / n_its * 100, it, loss))

    if it % plot_every == 0:
        all_losses.append(total_loss / plot_every)
        total_loss = 0
print("训练完成！")

# 画出损失图像
print("绘制损失图像......")
plt.figure()
plt.plot(all_losses)
plt.show()

# 生成测试
print("请输入是否要进行网络采样测试(1,是;0,否): ", end="")
flag = int(input())
while flag:
    try:
        print("请输入类别和姓名首字母(按空格分割,例: female Al): ", end="")
        cat, letters = input().split()
        name, indices, values = sample(cat, letters)
        # 可视化
        # visualization(letters, name, indices, values)
    except ellipsis:
        print(ellipsis)
    else:
        print("要继续吗(1,继续;0,停止)？", end="")
        flag = int(input())

