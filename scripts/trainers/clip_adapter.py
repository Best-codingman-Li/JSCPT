import os.path as osp
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerXX, Trainer_Bli_level
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.data import DataManager
from dassl.data.data_manager import build_data_loader
from dassl.data.datasets import build_dataset
from dassl.data.samplers import build_sampler
from dassl.data.transforms import INTERPOLATION_MODES, build_transform
from tabulate import tabulate

from trainers.vision_benchmark.evaluation import construct_dataloader, construct_multitask_dataset

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()


CUSTOM_TEMPLATES = {
    'OxfordPets': 'a photo of a {}, a type of pet.',
    'OxfordFlowers': 'a photo of a {}, a type of flower.',
    'FGVCAircraft': 'a photo of a {}, a type of aircraft.',
    'DescribableTextures': '{} texture.',
    'EuroSAT': 'a centered satellite photo of {}.',
    'StanfordCars': 'a photo of a {}.',
    'Food101': 'a photo of {}, a type of food.',
    'SUN397': 'a photo of a {}.',
    'Caltech101': 'a photo of a {}.',
    'UCF101': 'a photo of a person doing {}.',
    'ImageNet': 'a photo of a {}.',
    'ImageNetSketch': 'a photo of a {}.',
    'ImageNetV2': 'a photo of a {}.',
    'ImageNetA': 'a photo of a {}.',
    'ImageNetR': 'a photo of a {}.',
    'OfficeHome': 'a photo of a {}.',
    'Digit5': 'a photo of a {}.',
    'VisDA17': 'a photo of a {}.',
    'Office31': 'a photo of a {}.',
    'DomainNet': 'a photo of a {}.',
    'miniDomainNet': 'a photo of a {}.',
    'PACS': 'a photo of a {}.',
    'VLCS': 'a photo of a {}.'
}

def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)


    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


class Adapter(nn.Module):
    def __init__(self, c_in, reduction=4):
        super(Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.fc(x)
        return x
    
    
class TextEncoder(nn.Module):

    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.cfg = cfg
        self.classnames = classnames
        self.clip_model = clip_model
        self.dtype = clip_model.dtype
    
    def forward(self):
        temp = CUSTOM_TEMPLATES[self.cfg.DATASET.NAME]
        prompts = [temp.format(c.replace('_', ' ')) for c in self.classnames]
        prompts = torch.cat([clip.tokenize(p) for p in prompts])
        prompts = prompts.to('cuda')
        text_features = self.clip_model.encode_text(prompts)
        x = text_features
        return x



class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model, dm=None):
        super().__init__()

        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(cfg, classnames, clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.adapter = Adapter(512, 4).to(clip_model.dtype)

        self.multi_task_label_pertask = cfg.DATASET.MULTITASK_LABEL_PERTASK
        if self.multi_task_label_pertask:
            self.class_index_pertask_start = torch.arange(dm._num_classes)
            self.class_index_pertask_end = torch.arange(dm._num_classes)
            start_index = 0

            for class_index, task in enumerate(dm._task_names):
                class_num = len(dm._labelmap[task])
                self.class_index_pertask_start[class_index] = start_index
                start_index += class_num
                self.class_index_pertask_end[class_index] = start_index
            self.index = torch.arange(dm._num_classes).unsqueeze(0)

    def forward(self, image, task=None):

        image_features = self.image_encoder(image.type(self.dtype))
        x = self.adapter(image_features)

        ratio = 0.2
        image_features = ratio * x + (1 - ratio) * image_features

        text_features = self.text_encoder()

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()



        if self.multi_task_label_pertask:
            # Here we perform prompt selection
            domain_start_indexs = self.class_index_pertask_start[task].unsqueeze(-1)
            domain_end_indexs = self.class_index_pertask_end[task].unsqueeze(-1)
            # print(domain_start_indexs.shape, domain_end_indexs.shape, logits.shape)
            select_index = self.index.repeat(logits.shape[0], 1)
            select_index = (select_index >= domain_start_indexs).float() * (select_index < domain_end_indexs).float()
            # exit()
            logits = logits * select_index.to(logits.device)

        return logits


'''
class MVLPTCOOPDataManager(DataManager):

    def __init__(self, cfg, custom_tfm_train=None, custom_tfm_test=None, dataset_wrapper=None):
        # Load dataset
        label_offset = 0
        self.num_classes_list = []
        self.classnames_list = []
        self.lab2cname_list = {}
        self.dataset = None
        self._task_names = cfg.DATASET.DATASET.split(',')
        
        self._id2task = {}
        self._task_class_idx = {}

        #cfg.DATALOADER.TRAIN_X.SAMPLER = "SequentialSampler"
        #cfg.DATALOADER.TRAIN_U.SAMPLER = "SequentialSampler"

        for domain, dataset_name in enumerate(self._task_names):
            
            cfg.defrost()
            cfg.DATASET.NAME = dataset_name
            cfg.freeze()
            self._id2task[domain] = dataset_name
            dataset = build_dataset(cfg)

            self.num_classes_list.append(dataset._num_classes)
            self.classnames_list += dataset._classnames
            new_lab2cname_dict = {}
            for key, value in dataset._lab2cname.items():
                new_lab2cname_dict[key+label_offset] = value
            self.lab2cname_list.update(new_lab2cname_dict)
            for i in range(len(dataset._train_x)):
                dataset._train_x[i]._label += label_offset
                dataset._train_x[i]._domain = domain
            
            if dataset._train_u:
                for i in range(len(dataset._train_u)):
                    dataset._train_u[i]._label += label_offset
                    dataset._train_u[i]._domain = domain
                if self.dataset is not None:
                    self.dataset._train_u = self.dataset._train_u + dataset._train_u
            if dataset._val:
                for i in range(len(dataset._val)):
                    dataset._val[i]._label += label_offset
                    dataset._val[i]._domain = domain

            for i in range(len(dataset._test)):
                dataset._test[i]._label += label_offset
                dataset._test[i]._domain = domain
            
            if self.dataset is not None:
                self.dataset._train_x = self.dataset._train_x + dataset._train_x
                self.dataset._val = self.dataset.val + dataset.val
                self.dataset._test = self.dataset.test + dataset.test

            print(dataset._train_u is None, dataset._val is None)
            if self.dataset is None:
                self.dataset = dataset

            self._task_class_idx[dataset_name] = ( label_offset, label_offset + dataset._num_classes )
            label_offset += dataset._num_classes
            
        dataset = self.dataset
        dataset._classnames = self.classnames_list
        dataset._lab2cname = self.lab2cname_list
        dataset._num_classes = sum(self.num_classes_list)
        #print(self.num_classes_list, len(dataset._classnames), dataset._lab2cname, dataset._num_classes)
           
        # Build transform
        if custom_tfm_train is None:
            tfm_train = build_transform(cfg, is_train=True)
        else:
            print("* Using custom transform for training")
            tfm_train = custom_tfm_train

        if custom_tfm_test is None:
            tfm_test = build_transform(cfg, is_train=False)
        else:
            print("* Using custom transform for testing")
            tfm_test = custom_tfm_test

        # Build train_loader_x, cfg.DATALOADER.TRAIN_X.SAMPLER
        train_loader_x = build_data_loader(
            cfg,
            sampler_type=cfg.DATALOADER.TEST.SAMPLER,
            data_source=dataset.train_x,
            batch_size=cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
            n_domain=cfg.DATALOADER.TRAIN_X.N_DOMAIN,
            n_ins=cfg.DATALOADER.TRAIN_X.N_INS,
            tfm=tfm_train,
            is_train=True,
            dataset_wrapper=dataset_wrapper
        )

        # Build train_loader_u
        train_loader_u = None
        if dataset.train_u:
            sampler_type_ = cfg.DATALOADER.TEST.SAMPLER #cfg.DATALOADER.TRAIN_U.SAMPLER
            batch_size_ = cfg.DATALOADER.TRAIN_U.BATCH_SIZE
            n_domain_ = cfg.DATALOADER.TRAIN_U.N_DOMAIN
            n_ins_ = cfg.DATALOADER.TRAIN_U.N_INS

            if cfg.DATALOADER.TRAIN_U.SAME_AS_X:
                sampler_type_ = cfg.DATALOADER.TEST.SAMPLER #cfg.DATALOADER.TRAIN_X.SAMPLER
                batch_size_ = cfg.DATALOADER.TRAIN_X.BATCH_SIZE
                n_domain_ = cfg.DATALOADER.TRAIN_X.N_DOMAIN
                n_ins_ = cfg.DATALOADER.TRAIN_X.N_INS

            train_loader_u = build_data_loader(
                cfg,
                sampler_type=sampler_type_,
                data_source=dataset.train_u,
                batch_size=batch_size_,
                n_domain=n_domain_,
                n_ins=n_ins_,
                tfm=tfm_train,
                is_train=True,
                dataset_wrapper=dataset_wrapper
            )
        # Build val_loader
        val_loader = None
        if dataset.val:
            val_loader = build_data_loader(
                cfg,
                sampler_type=cfg.DATALOADER.TEST.SAMPLER,
                data_source=dataset.val,
                batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
                tfm=tfm_test,
                is_train=False,
                dataset_wrapper=dataset_wrapper
            )

        # Build test_loader
        test_loader = build_data_loader(
            cfg,
            sampler_type=cfg.DATALOADER.TEST.SAMPLER,
            data_source=dataset.test,
            batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
            tfm=tfm_test,
            is_train=False,
            dataset_wrapper=dataset_wrapper
        )

        # Attributes
        self._num_classes = dataset.num_classes
        self._num_source_domains = len(cfg.DATASET.SOURCE_DOMAINS)
        self._lab2cname = dataset.lab2cname

        # Dataset and data-loaders
        self.dataset = dataset
        self.train_loader_x = train_loader_x
        self.train_loader_u = train_loader_u
        self.val_loader = val_loader
        self.test_loader = test_loader

        if cfg.VERBOSE:
            self.show_dataset_summary(cfg)
'''

class MVLPTCOOPDataManager(DataManager):

    def __init__(self, cfg, custom_tfm_train=None, custom_tfm_test=None, dataset_wrapper=None):
        # Load dataset
        label_offset = 0
        self.num_classes_list = []
        self.classnames_list = []
        self.lab2cname_list = {}
        self.dataset = None
        self._task_names = cfg.DATASET.DATASET.split(',')
        self._id2task = {}
        self._task_class_idx = {}
        
        train_loader_x, train_loader_u, val_loader, test_loader = {}, {}, {}, {}

        for domain, dataset_name in enumerate(self._task_names):
            print("dataset_name", dataset_name)
            cfg.defrost()
            cfg.DATASET.NAME = dataset_name
            cfg.freeze()
            self._id2task[domain] = dataset_name
            dataset = build_dataset(cfg)
            print("dataset._num_classes", dataset._num_classes)
            print("dataset._classnames", dataset._classnames)

            self.num_classes_list.append(dataset._num_classes)
            self.classnames_list += dataset._classnames
            new_lab2cname_dict = {}

            print("dataset._lab2cname", dataset._lab2cname)
            
            if "OfficeHome" not in self._task_names and "MiniDomainNet" not in self._task_names:
                for key, value in dataset._lab2cname.items():
                    new_lab2cname_dict[key+label_offset] = value
                self.lab2cname_list.update(new_lab2cname_dict)

                #print("self.lab2cname_list", self.lab2cname_list)
                #print("label_offset", label_offset)

            for i in range(len(dataset._train_x)):
                dataset._train_x[i]._label += label_offset
                dataset._train_x[i]._domain = domain
            
            if dataset._train_u:
                for i in range(len(dataset._train_u)):
                    dataset._train_u[i]._label += label_offset
                    dataset._train_u[i]._domain = domain

                if self.dataset is not None:
                    self.dataset._train_u = self.dataset._train_u + dataset._train_u
                    
            if dataset._val:
                for i in range(len(dataset._val)):
                    dataset._val[i]._label += label_offset
                    dataset._val[i]._domain = domain

            for i in range(len(dataset._test)):
                dataset._test[i]._label += label_offset
                dataset._test[i]._domain = domain
            
            if self.dataset is not None:
                self.dataset._train_x = self.dataset._train_x + dataset._train_x
                self.dataset._val = self.dataset.val + dataset.val
                self.dataset._test = self.dataset.test + dataset.test

            #print(dataset._train_u is None, dataset._val is None)
            if self.dataset is None:
                self.dataset = dataset

            self._task_class_idx[dataset_name] = ( label_offset, label_offset + dataset._num_classes )

            if "OfficeHome" not in self._task_names and "DomainNet" not in self._task_names:
                label_offset += dataset._num_classes

            #the last lines in for loop are added
            # Build transform
            if custom_tfm_train is None:
                tfm_train = build_transform(cfg, is_train=True)
            else:
                print("* Using custom transform for training")
                tfm_train = custom_tfm_train

            if custom_tfm_test is None:
                tfm_test = build_transform(cfg, is_train=False)
            else:
                print("* Using custom transform for testing")
                tfm_test = custom_tfm_test

            # Build train_loader_x
            #cfg.DATALOADER.TRAIN_X.SAMPLER,
            train_loader_x[domain] = build_data_loader(
                cfg,
                sampler_type=cfg.DATALOADER.TRAIN_X.SAMPLER,
                data_source=dataset.train_x,
                batch_size=cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
                n_domain=cfg.DATALOADER.TRAIN_X.N_DOMAIN,
                n_ins=cfg.DATALOADER.TRAIN_X.N_INS,
                tfm=tfm_train,
                is_train=True,
                dataset_wrapper=dataset_wrapper
            )

            # Build train_loader_u
            #cfg.DATALOADER.TRAIN_X.SAMPLER,
            train_loader_u[domain] = None
            if dataset.train_u:
                sampler_type_ = cfg.DATALOADER.TRAIN_X.SAMPLER #### replaced
                batch_size_ = cfg.DATALOADER.TRAIN_U.BATCH_SIZE
                n_domain_ = cfg.DATALOADER.TRAIN_U.N_DOMAIN
                n_ins_ = cfg.DATALOADER.TRAIN_U.N_INS

                if cfg.DATALOADER.TRAIN_U.SAME_AS_X:
                    sampler_type_ = cfg.DATALOADER.TRAIN_X.SAMPLER ##### replaced
                    batch_size_ = cfg.DATALOADER.TRAIN_X.BATCH_SIZE
                    n_domain_ = cfg.DATALOADER.TRAIN_X.N_DOMAIN
                    n_ins_ = cfg.DATALOADER.TRAIN_X.N_INS

                train_loader_u[domain] = build_data_loader(
                    cfg,
                    sampler_type=sampler_type_,
                    data_source=dataset.train_u,
                    batch_size=batch_size_,
                    n_domain=n_domain_,
                    n_ins=n_ins_,
                    tfm=tfm_train,
                    is_train=True,
                    dataset_wrapper=dataset_wrapper
                )
            # Build val_loader
            val_loader[domain] = None
            if dataset.val:
                val_loader[domain] = build_data_loader(
                    cfg,
                    sampler_type=cfg.DATALOADER.TEST.SAMPLER,
                    data_source=dataset.val,
                    batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
                    tfm=tfm_test,
                    is_train=False,
                    dataset_wrapper=dataset_wrapper
                )

            # Build test_loader
            test_loader[domain] = build_data_loader(
                cfg,
                sampler_type=cfg.DATALOADER.TEST.SAMPLER,
                data_source=dataset.test,
                batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
                tfm=tfm_test,
                is_train=False,
                dataset_wrapper=dataset_wrapper
            )

        dataset = self.dataset
        if "OfficeHome" not in self._task_names and "DomainNet" not in self._task_names:
            dataset._classnames = self.classnames_list
            dataset._lab2cname = self.lab2cname_list
            dataset._num_classes = sum(self.num_classes_list)
        #print(len(dataset._classnames), dataset._lab2cname, dataset._num_classes)

        # Attributes
        self._num_classes = dataset.num_classes
        self._num_source_domains = len(cfg.DATASET.SOURCE_DOMAINS)
        self._lab2cname = dataset.lab2cname

        # Dataset and data-loaders
        self.dataset = dataset
        self.train_loader_x = train_loader_x
        self.train_loader_u = train_loader_u
        self.val_loader = val_loader
        self.test_loader = test_loader

        if cfg.VERBOSE:
            self.show_dataset_summary(cfg)


from trainers.vision_benchmark.datasets import class_map_metric, get_metric
import random
class MVLPTDataManager(DataManager):

    def __init__(self, cfg):
        # Load dataset
        train_loader_x, val_loader, test_loader, class_map, train_dataset = construct_dataloader(cfg)

        self._metric = get_metric(class_map_metric[cfg.DATASET.DATASET])
        self._metric_name = class_map_metric[cfg.DATASET.DATASET]

        # Attributes
        self._num_classes = len(class_map)
        self._num_source_domains = len(cfg.DATASET.SOURCE_DOMAINS)
        self._lab2cname = {}
        # random.seed(cfg.DATASET.RANDOM_SEED_SAMPLING)
        for key, value in enumerate(class_map):
            if isinstance(value, list):
                # value = random.choice(value)
                value = value[0]
            self._lab2cname[key] = value

        # Dataset and data-loaders
        # self.dataset.train_x = train_dataset
        self.train_loader_x = train_loader_x
        # self.train_loader_u = train_loader_u
        self.train_loader_u = None
        self.val_loader = val_loader
        self.test_loader = test_loader

        if cfg.VERBOSE:
            pass
            # self.show_dataset_summary(cfg)

class MVLPTMTDataManager(DataManager):

    def __init__(self, cfg):
        # Load dataset
        train_loader_x, val_loader, test_loader, train_dataset, test_dataloader_by_task = construct_multitask_dataset(cfg)

        self._labelmap = train_dataset.labelmap
        self._task_names = train_dataset._task_names
        self._task2id = { v: k for k, v in enumerate(self._task_names) }
        self._id2task = { k: v for k, v in enumerate(self._task_names) }
        self._metric = { task: get_metric(class_map_metric[task]) for task in self._task_names }
        self._metric_name = { task: class_map_metric[task] for task in self._task_names }

        class_idx = 0
        self._task_class_idx = {}
        for task in self._task_names:
            class_num = len(self._labelmap[task])
            self._task_class_idx[task] = ( class_idx, class_idx + class_num )
            class_idx += class_num
        
        from trainers.vision_benchmark.datasets import class_map

        print(self._task_names)
        print(self._labelmap)
        print(class_map.keys())

        mt_class_map = dict()
        for task in self._labelmap:
            for label_idx, label in enumerate(class_map[task]):
                cnt = train_dataset._get_cid( label_idx, task)
                mt_class_map[cnt] = label
        
        print(mt_class_map)
        # Attributes
        self._num_classes = len(mt_class_map)
        self._num_source_domains = len(cfg.DATASET.SOURCE_DOMAINS)
        self._lab2cname = {}
        
        # random.seed(cfg.DATASET.RANDOM_SEED_SAMPLING)
        for key, value in mt_class_map.items():
            if isinstance(value, list):
                value = value[0] #random.choice(value)
            self._lab2cname[key] = value

        # Dataset and data-loaders
        # self.dataset.train_x = train_dataset
        self.train_loader_x = train_loader_x
        # self.train_loader_u = train_loader_u
        self.train_loader_u = None
        self.val_loader = val_loader
        self.test_loader = test_loader

        if cfg.VERBOSE:
            pass


@TRAINER_REGISTRY.register()
class CLIP_Adapter(Trainer_Bli_level):
    """Context Optimization (Trainer_Bli_level).

    Learning to Prompt for Vision-Language Models
    https://arxiv.org/abs/2109.01134
    """

    def check_cfg(self, cfg):
        assert cfg.TRAINER.MVLPT.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        if self.cfg.DATASET.COOP:
            classnames = self.dm.dataset.classnames
        else:
            classnames = self.dm.lab2cname.values()

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        
        if cfg.TRAINER.MVLPT.PREC == "fp32" or cfg.TRAINER.MVLPT.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model, dm=self.dm)
        
        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if 'adapter' not in name:
                param.requires_grad_(False)

        print(f"Clip-Adapter Tunable Param: {sum([p.numel() for p in self.model.parameters() if p.requires_grad])/10**6}M, Original CLIP {sum([p.numel() for p in self.model.parameters() if not p.requires_grad])/10**6}M")
        
        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.adapter, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model.adapter, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("CLIP-Adapter", self.model.adapter, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.MVLPT.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    
    def build_data_loader(self):
        """Create essential data-related attributes.

        A re-implementation of this method must create the
        same attributes (self.dm is optional).
        """
        self.multi_task = self.cfg.DATASET.MULTITASK
        self.multi_task_label_pertask = self.cfg.DATASET.MULTITASK_LABEL_PERTASK

        if self.cfg.DATASET.COOP:
            print("********cfg.DATASET.COOP********")
            dm = MVLPTCOOPDataManager(self.cfg)
        elif self.cfg.DATASET.MULTITASK:
            print("********cfg.DATASET.MULTITASK********")
            dm = MVLPTMTDataManager(self.cfg)
        else:
            print("********MVLPTDataManager********")
            dm = MVLPTDataManager(self.cfg)

        self.train_loader_x = dm.train_loader_x
        self.train_loader_u = dm.train_loader_u  # optional, can be None
        self.val_loader = dm.val_loader  # optional, can be None
        self.test_loader = dm.test_loader

        self.num_classes = dm.num_classes
        self.num_source_domains = dm.num_source_domains
        self.lab2cname = dm.lab2cname  # dict {label: classname}

        self.dm = dm
        self.task_nums = len(self.dm._task_names)

    def forward_backward(self, batch_list, val_batch_list, aux_train_batch_list):

        loss_summary = {}
        task_loss_list = [0] * self.task_nums

        for task_index, batch in enumerate(batch_list):
            image, label = self.parse_batch_train(batch)

            # HACK: for multi-label classification, either works
            if len(label.shape) > 1 and label.shape[-1] > 1:
                label = label.float()
                label /= label.sum(dim=-1, keepdim=True)
            
            prec = self.cfg.TRAINER.MVLPT.PREC
            if prec == "amp":
                with autocast():
                    output = self.model(image, task=task_index)
                    loss = F.cross_entropy(output, label)
                self.optim.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optim)
                self.scaler.update()
            else:
                output = self.model(image, task=task_index)
                task_loss_list[task_index] = F.cross_entropy(output, label)

            # HACK: During training, we hack the eval of multi-label by selecting only one class
            if len(label.shape) > 1 and label.shape[-1] > 1:
                label = torch.argmax(label, dim=1)
            
            # result = self.dm._metric(label.squeeze().cpu().detach().numpy(), output.cpu().detach().numpy())
            loss_summary[task_index] = {
                "loss": task_loss_list[task_index].item(),
                "acc": compute_accuracy(output, label)[0].item(),
            }

        final_loss = sum(task_loss_list)

        self.optim.zero_grad()
        final_loss.backward()
        self.optim.step()

        
        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary
    

    def _prepare_dataloaders(self, dataloaders, task_name):

        loader = {}
        batch_num = []
        for task in task_name:
            loader[task] = [dataloaders[task], iter(dataloaders[task])]
            batch_num.append(len(dataloaders[task]))
        return loader, batch_num

    def _process_data(self, loader):
        try:
            batch = next(loader[1])
        except:
            loader[1] = iter(loader[0])
            batch = next(loader[1])
        return batch

    def parse_batch_train(self, batch):
        if self.cfg.DATASET.COOP:
            inp_key, lab_key, task_key = 'img', 'label', 'domain'
        else:
            inp_key, lab_key, task_key = 0, 1, 3
        input = batch[inp_key]
        label = batch[lab_key]
        # print(label.shape, 'label', input.shape, 'input')
        tasks = None
        if self.multi_task:
            tasks = batch[task_key]

        input = input.to(self.device)
        label = label.to(self.device)
        
        return input, label

    def parse_batch_test(self, batch):
        if self.cfg.DATASET.COOP:
            inp_key, lab_key, task_key = 'img', 'label', 'domain'
        else:
            inp_key, lab_key, task_key = 0, 1, 3
        input = batch[inp_key]
        label = batch[lab_key]
        tasks = None
        if self.multi_task:
            tasks = batch[task_key]

        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def model_inference(self, image, task=None):
        return self.model(image=image, task=task)

    @torch.no_grad()
    def test(self, split=None):
        from tqdm import tqdm
        import copy 
        import numpy as np
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"  # in case val_loader is None
            data_loader = self.test_loader

        print(f"Evaluate on the *{split}* set")

        self.evaluator_task = dict()
        self.elevator_evaluator = { 'y_pred': [], 'y_true': [] }
        
        if self.multi_task:
            if self.cfg.DATASET.COOP:
                
                self.evaluator_task = { task: copy.deepcopy( self.evaluator ) for task in self.dm._task_names }
            else:
                self.evaluator_task = { task: copy.deepcopy( self.elevator_evaluator ) for task in self.dm._task_names }

        
        for task_ in data_loader.keys():
            
            for batch_idx, batch in enumerate(tqdm(data_loader[task_])):
                image, label = self.parse_batch_train(batch)

                output = self.model_inference(image, task=task_)

                # HACK: make everything one-hot vector label!
                if self.cfg.DATASET.COOP:
                    self.evaluator.process(output, label)
                
                else:
                    self.elevator_evaluator['y_pred'].append( output.cpu().detach().numpy() )
                    self.elevator_evaluator['y_true'].append( label.cpu().detach().numpy() )

                tasks = [task_] * len(label)
                if tasks is not None:
                    for out, lab, task in zip(output, label, tasks):
   
                        #task = self.dm._id2task[task.item()]
                        task = self.dm._id2task[task]
                        
                        if self.cfg.DATASET.COOP:
                            class_start, class_end = self.dm._task_class_idx[task]#[0, 101], [101, 138], [138, 535]
                            
                            # Evaluate on the task-specific class
                            out = out[class_start:class_end]
                            lab -= class_start
                            self.evaluator_task[task].process(out.unsqueeze(0), lab.unsqueeze(0))
                        else:
                            self.evaluator_task[task]['y_pred'].append( [out.cpu().detach().numpy()] )
                            self.evaluator_task[task]['y_true'].append( [lab.cpu().detach().numpy()] )
        

        results_overall = {}
        for task in self.evaluator_task:
            print(f"evaluate on the *{task}* !")
            if self.cfg.DATASET.COOP:
                results = self.evaluator_task[task].evaluate()
                results_overall[task] = results['accuracy']
            else:
                y_true = np.concatenate( self.evaluator_task[task]['y_true'] , axis=0)
                y_pred = np.concatenate( self.evaluator_task[task]['y_pred'] , axis=0)
                class_start, class_end = self.dm._task_class_idx[task]
                y_true = y_true[:, class_start:class_end]
                y_pred = y_pred[:, class_start:class_end]
                
                if self.dm._metric_name[task] == 'accuracy':
                    y_true = np.argmax(y_true, axis=-1)
                metric_result = self.dm._metric[task]( y_true, y_pred )
                results = { self.dm._metric_name[task]: metric_result }
                results_overall[ task ] = metric_result
            print( 'results', results )
            for k, v in results.items():
                tag = f"{split}/{task}/{k}"
                self.write_scalar(tag, v, self.epoch)
        
        print(f"Overall evaluation !")
        if self.multi_task:
            multi_task_evalkey = self.cfg.DATASET.MULTITASK_EVALKEY
            if multi_task_evalkey == 'average':
                results = {'average' : sum([v for k, v in results_overall.items()]) / len(results_overall)}
            else:
                assert multi_task_evalkey in results_overall
                results = {multi_task_evalkey : results_overall[multi_task_evalkey]}
                print(f"select {multi_task_evalkey} as the evaluation key")
        else:
            if not self.cfg.DATASET.COOP:
                y_true = np.concatenate( self.elevator_evaluator['y_true'] , axis=0)
                y_pred = np.concatenate( self.elevator_evaluator['y_pred'] , axis=0)
                results = { self.dm._metric_name: self.dm._metric( y_true, y_pred ) }
            else:
                results = self.evaluator.evaluate()
        print( 'results', results )
        for k, v in results.items():
            tag = f"/{split}/{k}"
            self.write_scalar(tag, v, self.epoch)
        
        return list(results.values())[0]


    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            # issue 1 bug fix for UPT key name mismatch
            state_dict = {k.replace("upt_proj", "mvlpt_proj"):v for k, v in state_dict.items()}
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors

            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
