import math
import numpy as np

def findAllIndexInList(aim,List):
    pos = 0
    index=[]
    for each in List:
        if each == aim:
            index.append(pos)
        pos += 1
    return index

def CreateNewListByIndex(Index, List):
    newList = []
    List = list(List)
    Index = list(Index)
    for each in Index:
        newList.append(List[each])
    return newList

def Pi(aim, List):
    length = len(list(List))
    aimcount = (list(List)).count(aim)
    pi = (float)(aimcount/length)
    return pi


def calcEntropy(data: list):  # 输入的data 是 X 所有取值（重复值不去除）的列表

    data1 = np.unique(data)
    # 找到列表里所有值（去除重复值）组成一个新的列表data1
    # 这里没有numpy库的 可以用set函数把data变成一个集合，也可以去除重复值~

    resultEn = 0  # 单个元素的熵H(X)保存在resultEn

    for each in data1:  # data1里保存的值不重复
        pi = Pi(each, data)  # 求出data（data里的值可能重复）中每个 xi出现的概率
        resultEn -= pi * math.log(pi, 2)  # 对不同xi的信息熵求和过程

    return resultEn