'''
Author: Guosy_wxy 1579528809@qq.com
Date: 2024-07-04 06:45:05
LastEditors: Guosy_wxy 1579528809@qq.com
LastEditTime: 2024-07-04 06:49:06
FilePath: /Prompt/mvlpt-master/scripts/mvlpt/test.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import torch
dic = {"a" : [1, 2, 3, 4], "b" : [5, 6, 7, 8], "c" : [9, 10, 11, 12, 13]}

for name, data_l in dic.items():
    print("name", name)
    data = iter(data_l)
    print("data",next(data))