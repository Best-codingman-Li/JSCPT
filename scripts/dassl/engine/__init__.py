'''
Author: Guosy_wxy 1579528809@qq.com
Date: 2024-06-28 12:39:25
LastEditors: Guosy_wxy 1579528809@qq.com
LastEditTime: 2024-07-10 09:26:11
FilePath: /Prompt/mvlpt-master/scripts/dassl/engine/__init__.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from .build import TRAINER_REGISTRY, build_trainer  # isort:skip
from .trainer import TrainerX, TrainerXX, Trainer_Bli_level, TrainerX1, TrainerXU, TrainerBase, SimpleTrainer, SimpleNet  # isort:skip

from .da import *
from .dg import *
from .ssl import *
