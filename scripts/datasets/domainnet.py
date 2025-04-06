'''
Author: Guosy_wxy 1579528809@qq.com
Date: 2024-07-20 07:09:05
LastEditors: Guosy_wxy 1579528809@qq.com
LastEditTime: 2024-08-06 11:18:46
FilePath: /Prompt/mvlpt-master/scripts/datasets/domainnet.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import os.path as osp
import os
import pickle
import math
import random
from collections import defaultdict

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import listdir_nohidden, mkdir_if_missing

from .oxford_pets import OxfordPets
from .dtd import DescribableTextures as DTD

@DATASET_REGISTRY.register()
class DomainNet(DatasetBase):
    """DomainNet.

    Statistics:
        - 6 distinct domains: Clipart, Infograph, Painting, Quickdraw,
        Real, Sketch.
        - Around 0.6M images.
        - 345 categories.
        - URL: http://ai.bu.edu/M3SDA/.

    Special note: the t-shirt class (327) is missing in painting_train.txt.

    Reference:
        - Peng et al. Moment Matching for Multi-Source Domain
        Adaptation. ICCV 2019.
    """

    dataset_dir = "domainnet"
    domains = [
        "clipart", "infograph", "painting", "quickdraw", "real", "sketch"
    ]

    def __init__(self, cfg):
        root = osp.abspath(osp.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.image_dir = osp.join(self.dataset_dir, "clipart")
        self.split_dir = osp.join(self.dataset_dir, "splits")
        self.split_fewshot_dir = osp.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)

        train = self._read_data(["clipart"], split="train")
        val = self._read_data(["clipart"], split="test")
        test = self._read_data(["clipart"], split="test")

        num_shots = cfg.DATASET.NUM_SHOTS
        if num_shots >= 1:
            seed = cfg.SEED
            preprocessed = osp.join(self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl")
            
            if osp.exists(preprocessed):
                print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    train, val = data["train"], data["val"]
            else:
                train = self.generate_fewshot_dataset(train, num_shots=num_shots)
                val = self.generate_fewshot_dataset(val, num_shots=min(num_shots, 4))
                data = {"train": train, "val": val}
                print(f"Saving preprocessed few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

        subsample = cfg.DATASET.SUBSAMPLE_CLASSES
        train, val, test = OxfordPets.subsample_classes(train, val, test, subsample=subsample)

        super().__init__(train_x=train, val=val, test=test)


    def _read_data(self, input_domains, split="train"):
        items = []
        if split == "train":
            lines_num = 5107#8556
        else:
            lines_num = 2233#3737

        for domain, dname in enumerate(input_domains):
            filename = dname + "_" + split + ".txt"
            split_file = osp.join(self.dataset_dir, filename)

            with open(split_file, "r") as f:
                lines = f.readlines()
                for line_index, line in enumerate(lines):
                    if line_index < lines_num:
                        line = line.strip()
                        impath, label = line.split(" ")
                        classname = impath.split("/")[1]
                        impath = osp.join(self.dataset_dir, impath)
                        label = int(label)
                        item = Datum(
                            impath=impath,
                            label=label,
                            domain=domain,
                            classname=classname
                        )
                        items.append(item)
                    else:
                        break

        return items
