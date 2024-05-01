import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def draw_three_linegraph(x,y1,y2,y1_new,y2_new,x_begin,x_end,y_begin,y_end,smooth=False):

    # deluxe #DB9370  #F7E697
    # basic #545454  #38B0DE

    alpha = 1.0
    label1 = 'basic'
    label2 = 'deluxe'

    if smooth:
        plt.plot(x, y1_new, label=label1, color='#38B0DE') #  #F7E697 marker='^'
        plt.plot(x, y2_new, label=label2, color='#F7E697') # marker='s'
        alpha = 0.2
        label1 = None
        label2 = None

    # 创建折线图
    plt.plot(x, y1, label=label1, color='#38B0DE', alpha=alpha) #  #F7E697 marker='^'
    plt.plot(x, y2, label=label2, color='#F7E697', alpha=alpha) # marker='s'


    # plt.plot(x, y3, label='Zhengzhou', color='blue', marker='o')
    # 设置图表标题和轴标签
    plt.title('vanilla diffaug D')
    plt.xlabel('iter')
    plt.ylabel('loss')

    # 设置图例
    plt.legend(loc='upper right')

    plt.xlim(x_begin, x_end)
    plt.ylim(y_begin, y_end)

    # 显示网格线
    plt.grid(True)

    plt.savefig('loss_plot1.png')


def read_file(filename):
    with open(filename,'r') as file:
        data_ls = file.readline().split(',')

    loss_ls = [float(x) for x in data_ls]
    return loss_ls


def ls2df(ls):
    dic = {"loss_num": ls}
    df = pd.DataFrame(dic)

    return df

def smooth_ls(ls, rolling_intv=5):
    df = ls2df(ls)
    d = list(np.hstack(df.rolling(rolling_intv, min_periods=1).mean().values))

    return d




if __name__=='__main__':

    filename1 = "/home/zhaoyiwen/learning/All-in-generative-models/course_assignment/hw3/vanilla_grumpifyBprocessed_basic_diffaug__total_D_loss.txt"
    filename2 = "/home/zhaoyiwen/learning/All-in-generative-models/course_assignment/hw3/vanilla_grumpifyBprocessed_deluxe_diffaug__total_D_loss.txt"
    smooth = True
   
    x = [10 * i for i in range(650)]
    loss1 = read_file(filename1)
    loss2 = read_file(filename2)

    if smooth:
        rolling_intv = 5 #5 neighbour average
        loss1_new = smooth_ls(loss1,8)
        loss2_new = smooth_ls(loss2,8)

    value_range = (min(min(loss1), min(loss2)), max(max(loss1), max(loss2)))
    

    # 设置x和y的坐标轴范围
    x_begin, x_end = 0, 6500
    y_begin, y_end = 0, value_range[1]+(value_range[1]-value_range[0])/10.0
    draw_three_linegraph(x,loss1,loss2,loss1_new,loss2_new,x_begin,x_end,y_begin,y_end,smooth)
