"""Shared utilities for all main scripts."""

import os
import pickle
import random
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, default_collate
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange
from datasets.calvin_dataset import transfer
from LLCB_test_pt import Add_LoRA, Model_init, preprocess, input_processing_real_batch, parse_args
from model.llava.mm_utils import tokenizer_image_token
from peft import LoraConfig, get_peft_model
import time
import math
from pathlib import Path
import transformers

class linearlayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(linearlayer, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

class BaseTrainTester:
    """Basic train/test class to be inherited."""

    def __init__(self, args):
        """Initialize."""
        if dist.get_rank() == 0:
            args.save(str(args.log_dir / "hparams.json"))

        self.args = args

        if dist.get_rank() == 0:
            path = args.log_dir
            redirect_path = Path(str(path).replace('/storage/dingpengxiang','/dingpengxiang/Pengxiang/eccv2024'))
            print('redirect path:', redirect_path )
            redirect_path.mkdir(exist_ok=True, parents=True)
            self.writer = SummaryWriter(log_dir=redirect_path)
    
    @staticmethod
    def get_datasets():
        """Initialize datasets."""
        train_dataset = None
        test_dataset = None
        return train_dataset, test_dataset

    def get_loaders(self, collate_fn=default_collate):
        """Initialize data loaders."""
        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)
            np.random.seed(np.random.get_state()[1][0] + worker_id)
        # Datasets
        train_dataset, test_dataset = self.get_datasets()
        # Samplers and loaders
        g = torch.Generator()
        g.manual_seed(0)
        train_sampler = DistributedSampler(train_dataset)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            worker_init_fn=seed_worker,
            collate_fn=collate_fn,
            pin_memory=True,
            sampler=train_sampler,
            drop_last=True,
            generator=g
        )
        
        #单独写一个train_evaluate_loader是为了防止evaluate的时候out of memory，他与train_loader的区别在于batch_size更小
        # train_eva_batchsize = math.ceil(self.args.batch_size / 5)
        train_eva_batchsize = 1
        train_evaluate_loader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size_val,
            shuffle=False,
            num_workers=0,
            worker_init_fn=seed_worker,
            collate_fn=collate_fn,
            pin_memory=True,
            sampler=train_sampler,
            drop_last=True,
            generator=g
        )

        test_sampler = DistributedSampler(test_dataset, shuffle=True)
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.args.batch_size_val,
            shuffle=False,
            num_workers=0,
            worker_init_fn=seed_worker,
            collate_fn=collate_fn,
            pin_memory=True,
            sampler=test_sampler,
            drop_last=False,
            generator=g
        )
        return train_loader, test_loader, train_evaluate_loader

    @staticmethod
    def get_model():
        """Initialize the model."""
        return None

    @staticmethod
    def get_criterion():
        """Get loss criterion for training."""
        # criterion is a class, must have compute_loss and compute_metrics
        return None

    def get_optimizer(self, model):
        """Initialize optimizer."""
        optimizer_grouped_parameters = [
            {"params": [], "weight_decay": 0.0, "lr": self.args.lr},
            {"params": [], "weight_decay": 5e-4, "lr": self.args.lr}
        ]
        no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
        for name, param in model.named_parameters():
            if any(nd in name for nd in no_decay):
                optimizer_grouped_parameters[0]["params"].append(param)
            else:
                optimizer_grouped_parameters[1]["params"].append(param)
        optimizer = optim.AdamW(optimizer_grouped_parameters)
        return optimizer
    
    #clip_loss所需的函数
    # 停止梯度计算的函数
    def stop_gradient(self, x):
        return x.detach()

    # 定义辅助损失函数
    def clip_auxiliary_loss(self, z_act, g_txt, processor, device, model, linear_layer):
        # 计算文本输入的 CLIP 嵌入
        with torch.no_grad():
            inputs = processor(text=[g_txt], return_tensors="pt", padding=True, truncation=True)
            inputs = {key: value.to(device) for key, value in inputs.items()}
            g_txt_embed = model.get_text_features(**inputs)
        
        # 通过线性层调整 CLIP 嵌入的维度
        g_txt_embed = linear_layer(g_txt_embed)
        
        # 确保 g_txt_embed 和 z_act 在相同的设备上
        g_txt_embed = g_txt_embed.to(z_act.device)
        
        # 计算嵌入之间的余弦相似度
        loss = F.cosine_similarity(self.stop_gradient(g_txt_embed), z_act, dim=-1).mean()
        return loss

    def caculate_clip_loss(self, z_act, g_txt, processor, device, model, linear_layer):#g_txt是一个列表，z_act是一个tensor
        total_aux_loss = 0
        for i in range(len(g_txt)):
            # 计算辅助损失
            aux_loss = self.clip_auxiliary_loss(z_act[i], g_txt[i], processor, device, model, linear_layer)
            total_aux_loss += aux_loss.item()
        return total_aux_loss
    
        # 定义辅助损失函数
    def batch_clip_loss(self, z_act, g_txt_batch, processor, device, model, linear_layer):
        # 计算文本输入的 CLIP 嵌入
        with torch.no_grad():
            # 处理批量的文本输入
            inputs = processor(text=g_txt_batch, return_tensors="pt", padding=True, truncation=True)
            inputs = {key: value.to(device) for key, value in inputs.items()}
            g_txt_embed = model.module.get_text_features(**inputs)

        # 通过线性层调整z_act的维度与CLIP嵌入的维度相同
        g_txt_embed = linear_layer(g_txt_embed)

        # 确保 g_txt_embed和z_act在相同的设备上
        g_txt_embed = g_txt_embed.to(z_act.device)

        # 计算嵌入之间的余弦相似度
        loss = F.cosine_similarity(self.stop_gradient(g_txt_embed), z_act, dim=-1).mean()
        return loss
    
    def caculate_batch_clip_loss(self, z_act_batch, g_txt_batch, processor, device, model, linear_layer):
        # 计算整个批量的辅助损失
        aux_loss = self.batch_clip_loss(z_act_batch, g_txt_batch, processor, device, model, linear_layer)
        
        # 返回整个批量的损失值
        return aux_loss.item()

    def main(self, collate_fn=default_collate):
        """Run main training/testing pipeline."""
        # Get loaders
        train_loader, test_loader, train_evaluate_loader = self.get_loaders(collate_fn)

        # Get model
        model = self.get_model()

        # Get criterion
        criterion = self.get_criterion()

        # Get optimizer
        optimizer = self.get_optimizer(model)

        # Move model to devices
        if torch.cuda.is_available():
            model = model.cuda()
        model = DistributedDataParallel(
            model, device_ids=[self.args.local_rank],
            broadcast_buffers=False, find_unused_parameters=True
        )

        # Check for a checkpoint
        start_iter, best_loss = 0, None
        if self.args.training_checkpoint:
            assert os.path.isfile(self.args.training_checkpoint)
            start_iter, best_loss = self.load_checkpoint(model, optimizer)

        # Eval only
        if bool(self.args.eval_only):
            print("Test evaluation.......")
            model.eval()
            new_loss = self.evaluate_nsteps(
                model, criterion, test_loader, step_id=-1,
                val_iters=max(
                    5,
                    int(4 * len(self.args.tasks)/self.args.batch_size_val)
                )
            )
            return model
        
        #===============LLM初始化（CLIP/tokenizer）==============
        llava_dir = self.args.llava_dir
        vision_tower = self.args.vision_tower
        sample_rate = self.args.sample_rate
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.bfloat16
        clip_image_processor, tokenizer, LCB_model = Model_init(vision_tower, llava_dir, torch_dtype)
        LCB_model.resize_token_embeddings(len(tokenizer))
        for name, param in LCB_model.named_parameters():
            if param.requires_grad:
                print(f"Module: {name}, Shape: {param.shape}")
        # import pdb; pdb.set_trace()
        # Get LLM optimizer
        import torch.optim as optim
        # 加载 AdamW 优化器
        LLM_optimizer = optim.AdamW(
            LCB_model.parameters(),  # 模型的可训练参数
            lr=0.00005,          # 学习率
            weight_decay=0.00001,    # 权重衰减
            betas=(0.9, 0.95)  # beta 参数
        )
        if torch.cuda.is_available():
            LCB_model = LCB_model.cuda()
        # LCB_model = LCB_model.to(device)
        LCB_model = DistributedDataParallel(
            LCB_model, device_ids=[self.args.local_rank],
            broadcast_buffers=False, find_unused_parameters=True
        )
        # 加载预训练的 CLIP 模型和处理器
        # model_id = "/storage/dingpengxiang/clip-vit-large-patch14"
        # clip_model = CLIPModel.from_pretrained(model_id).to(device)
        # clip_model = DistributedDataParallel(
        #     clip_model, device_ids=[self.args.local_rank],
        #     broadcast_buffers=False, find_unused_parameters=True
        # )
        # processor = CLIPProcessor.from_pretrained(model_id)
        # # 创建线性层实例，输入维度需要与 CLIP 模型的嵌入维度相同
        # clip_embedding_dim = clip_model.module.config.projection_dim  # 获取 CLIP 嵌入维度
        # linear_layer = linearlayer(input_dim=512, output_dim=192).to(device)

        # Training loop
        iter_loader = iter(train_loader)
        model.train()
        LCB_model.train()
        for step_id in trange(start_iter, self.args.stage2_train_iters):
            try:
                sample = next(iter_loader)
            except StopIteration:
                iter_loader = iter(train_loader)
                sample = next(iter_loader)
            # 准备数据
            conversations, questions = transfer(sample['instr_text'])
            #第一阶段开始================LLM===================
            start_idx = []
            initial_id=0
            for i in range(len(conversations)):
                if sample['curr_gripper_history'][i][0][0]==sample['curr_gripper_history'][i][2][0]:
                    initial_id = i
                if (i-initial_id) % sample_rate == 0:
                    start_idx.append(i)
            image_rgb = sample['rgbs'][:, 0]
            image_select_rows = image_rgb[start_idx]
            conv_select = [conversations[i] for i in start_idx]
            image_clip, image, input_ids, attention_masks, targets = input_processing_real_batch(image_tensor=image_select_rows, conv_list=conv_select, clip_image_processor=clip_image_processor, tokenizer=tokenizer)
            
            pred_actions_embedding, ce_loss = LCB_model.module.model_forward(
                images=image,  
                images_clip=image_clip,  
                input_ids=input_ids,
                labels=targets,
                attention_masks=attention_masks,
                tokenizer=tokenizer,
            )
            # ce_loss = ce_loss / self.args.accumulate_grad_batches
            # print("ce_loss:", ce_loss)
            # import pdb;pdb.set_trace()
            
            # 创建一个与原始张量形状相同的全零张量
            total_action_embedding = torch.zeros(len(conversations), pred_actions_embedding.shape[1])#batch, 512
            # 遍历 batch 的维度并进行赋值
            for i in range(len(conversations)):
                if i in start_idx:
                    # 如果当前 i 在 start_idx 中，则将 pred_actions_embedding 的相应行赋值到 total_action_embedding
                    total_action_embedding[i] = pred_actions_embedding[start_idx.index(i)]
                else:
                    # 如果当前 i 不在 start_idx 中，则赋值为上一个 start_idx 的值
                    previous_idx = max([idx for idx in start_idx if idx < i])
                    total_action_embedding[i] = total_action_embedding[previous_idx]
            #先把clip loss注销了
            # clip_loss = (1 - self.caculate_batch_clip_loss(total_action_embedding, sample['instr_text'], processor, device, clip_model, linear_layer))
            #得到embedding之后直接embedding=embedding.unsqueeze(1),把embedding的维度从（8，512）改为（8，1，512），这样就可以直接当instruction输入了
            # total_action_embedding = total_action_embedding.unsqueeze(1).to(torch.float32)
            total_action_embedding = total_action_embedding.unsqueeze(1)
            # print("action_embedding的形状变成了:", action_embedding.shape)
            # print("第一阶段结束")
            #第二阶段开始================3dda===================
            # ce_loss = ce_loss.to(torch.float32)
            # lmcliploss = ce_loss
            #下面这行是policy训练的主要代码
            self.train_one_step(model, criterion, optimizer, step_id, sample, total_action_embedding)
            # 执行LLM优化器的step，更新参数
            if (step_id + 1) % self.args.accumulate_grad_batches == 0:
                LLM_optimizer.step()
                LLM_optimizer.zero_grad()
            #希望记录的数据点多一些，所以每隔0.01*val_freq记录一次
            if dist.get_rank() == 0 and (step_id + 1) % (0.01 * self.args.val_freq) == 0:
                self.writer.add_scalar("ce_loss", ce_loss, step_id)
                # self.writer.add_scalar("clip_loss", clip_loss, step_id)
            if (step_id + 1) % self.args.val_freq == 0:
                # lcb的范式先不用在训练中进行evaluate，以后有必要再开启
                # print("Train evaluation.......")
                # model.eval()
                # new_loss = self.evaluate_nsteps_lcb(
                #     model, criterion, train_evaluate_loader, step_id, LCB_model, clip_image_processor, tokenizer,
                #     val_iters=max(
                #         5,
                #         int(4 * len(self.args.tasks)/self.args.batch_size_val),
                #     ),
                #     split='train'
                # )
                # print("Test evaluation.......")
                # model.eval()
                # new_loss = self.evaluate_nsteps_lcb(
                #     model, criterion, test_loader, step_id, LCB_model, clip_image_processor, tokenizer,
                #     val_iters=max(
                #         5,
                #         int(4 * len(self.args.tasks)/self.args.batch_size_val)
                #     )
                # )
                if dist.get_rank() == 0:  # save model
                    best_loss = self.save_checkpoint(
                        model, optimizer, step_id, best_loss
                    )
                    save_path = self.args.log_dir / '{:07d}'.format(step_id)
                    os.makedirs(save_path, exist_ok=True)
                    LCB_model.module.save_pretrained(save_path)
                    print(f"Model saved at {save_path}")
                    torch.save(LCB_model.module.state_dict(), os.path.join(save_path, 'pytorch_model.bin'))
                    model.train()

        return model

    def train_one_step(self, model, criterion, optimizer, step_id, sample):
        """Run a single training step."""
        pass

    @torch.no_grad()
    def evaluate_nsteps(self, model, criterion, loader, step_id, val_iters,
                        split='val'):
        """Run a given number of evaluation steps."""
        return None

    def load_checkpoint(self, model, optimizer):
        """Load from checkpoint."""
        print("=> loading checkpoint '{}'".format(self.args.training_checkpoint))

        model_dict = torch.load(self.args.training_checkpoint, map_location="cpu")
        model.load_state_dict(model_dict["weight"])
        if 'optimizer' in model_dict:
            optimizer.load_state_dict(model_dict["optimizer"])
            for p in range(len(optimizer.param_groups)):
                optimizer.param_groups[p]['lr'] = self.args.lr
        start_iter = model_dict.get("iter", 0)
        best_loss = model_dict.get("best_loss", None)

        print("=> loaded successfully '{}' (step {})".format(
            self.args.training_checkpoint, model_dict.get("iter", 0)
        ))
        del model_dict
        torch.cuda.empty_cache()
        return start_iter, best_loss

    #真正调用的save_checkpoint函数是main_trajectory_calvin.py中的save_checkpoint函数
    def save_checkpoint(self, model, optimizer, step_id, new_loss, best_loss):
        """Save checkpoint if requested."""
        if new_loss is None or best_loss is None or new_loss <= best_loss:
            best_loss = new_loss
            torch.save({
                "weight": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "iter": step_id + 1,
                "best_loss": best_loss
            }, self.args.log_dir / "best.pth")
        torch.save({
            "weight": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iter": step_id + 1,
            "best_loss": best_loss
        }, self.args.log_dir / "last.pth")
        return best_loss

    def synchronize_between_processes(self, a_dict):
        all_dicts = all_gather(a_dict)

        if not is_dist_avail_and_initialized() or dist.get_rank() == 0:
            merged = {}
            for key in all_dicts[0].keys():
                device = all_dicts[0][key].device
                merged[key] = torch.cat([
                    p[key].to(device) for p in all_dicts
                    if key in p
                ])
            a_dict = merged
        return a_dict


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)

    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # obtain Tensor size of each rank
    local_size = torch.tensor([tensor.numel()], device="cuda")
    size_list = [torch.tensor([0], device="cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.empty(
            (max_size,), dtype=torch.uint8, device="cuda"
        ))
    if local_size != max_size:
        padding = torch.empty(
            size=(max_size - local_size,),
            dtype=torch.uint8, device="cuda"
        )
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()
