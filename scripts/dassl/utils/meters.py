'''
Author: Guosy_wxy 1579528809@qq.com
Date: 2024-06-28 12:39:25
LastEditors: Guosy_wxy 1579528809@qq.com
LastEditTime: 2024-07-08 06:58:57
FilePath: /Prompt/mvlpt-master/scripts/dassl/utils/meters.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from collections import defaultdict
import torch

__all__ = ["AverageMeter", "MetricMeter", "Multi_MetricMeter"]


class AverageMeter:
    """Compute and store the average and current value.

    Examples::
        >>> # 1. Initialize a meter to record loss
        >>> losses = AverageMeter()
        >>> # 2. Update meter after every mini-batch update
        >>> losses.update(loss_value, batch_size)
    """

    def __init__(self, ema=False):
        """
        Args:
            ema (bool, optional): apply exponential moving average.
        """
        self.ema = ema
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if isinstance(val, torch.Tensor):
            val = val.item()

        self.val = val
        self.sum += val * n
        self.count += n

        if self.ema:
            self.avg = self.avg * 0.9 + self.val * 0.1
        else:
            self.avg = self.sum / self.count


class MetricMeter:
    """Store the average and current value for a set of metrics.

    Examples::
        >>> # 1. Create an instance of MetricMeter
        >>> metric = MetricMeter()
        >>> # 2. Update using a dictionary as input
        >>> input_dict = {'loss_1': value_1, 'loss_2': value_2}
        >>> metric.update(input_dict)
        >>> # 3. Convert to string and print
        >>> print(str(metric))
    """

    def __init__(self, delimiter=" "):
        self.meters = defaultdict(AverageMeter) 
        self.delimiter = delimiter

    def update(self, input_dict):
        if input_dict is None:
            return

        if not isinstance(input_dict, dict):
            raise TypeError(
                "Input to MetricMeter.update() must be a dictionary"
            )
        for k, v in input_dict.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            self.meters[k].update(v)

    def __str__(self):
        output_str = []
        for name, meter in self.meters.items():
            output_str.append(f"{name} {meter.val:.4f} ({meter.avg:.4f})")
        return self.delimiter.join(output_str)


class Multi_MetricMeter:
    """Store the average and current value for a set of metrics.

    Examples::
        >>> # 1. Create an instance of MetricMeter
        >>> metric = MetricMeter()
        >>> # 2. Update using a dictionary as input
        >>> input_dict = {'loss_1': value_1, 'loss_2': value_2}
        >>> metric.update(input_dict)
        >>> # 3. Convert to string and print
        >>> print(str(metric))
    """

    def __init__(self, task_num, delimiter=" "):
        self.task_num = task_num
        self.meters = [defaultdict(AverageMeter) for _ in range (task_num)]
        self.delimiter = delimiter

    def update(self, input_dict):
        if input_dict is None:
            return

        if not isinstance(input_dict, dict):
            raise TypeError(
                "Input to MetricMeter.update() must be a dictionary"
            )
        print("input_dict", input_dict)
        for task_name in input_dict.keys():
            for k, v in input_dict[task_name].items():
                if isinstance(v, torch.Tensor):
                    v = v.item()
                self.meters[task_name][k].update(v)

    def __str__(self):
        for task in range(self.task_num):
            output_str = []
            for name, meter in self.meters[task].items():
                output_str.append(f"{task} {name} {meter.val:.4f} ({meter.avg:.4f})======")
        return self.delimiter.join(output_str)
