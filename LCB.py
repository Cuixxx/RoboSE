from typing import List
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BitsAndBytesConfig, CLIPVisionModel


from peft import LoraConfig, get_peft_model
from model.llava.model.language_model.llava_llama import (LlavaLlamaForCausalLM, LlavaLlamaModel)

mse_loss = nn.MSELoss()




class LisaMetaModel:
    def __init__(self, config, **kwargs):
        super(LisaMetaModel, self).__init__(config)

        self.config = config
        self.config.out_dim = kwargs["out_dim"]
        self.vision_pretrained = kwargs.get("vision_pretrained", None)
        self.initialize_lisa_modules(self.config)

    def initialize_lisa_modules(self, config):
        # Projection layer
        in_dim = config.hidden_size
        out_dim = config.out_dim
        text_fc = [nn.Linear(in_dim, in_dim), nn.ReLU(inplace=True), nn.Linear(in_dim, out_dim), nn.Dropout(0.0)]

        output_fc = [nn.Linear(out_dim, out_dim), nn.ReLU(inplace=True), nn.Linear(out_dim, 8), nn.Dropout(0.0),]
        self.text_hidden_fcs = nn.ModuleList([nn.Sequential(*text_fc)])
        self.out_fcs = nn.ModuleList([nn.Sequential(*output_fc)])
        self.text_hidden_fcs.train()
        self.out_fcs.train()
        for param in self.text_hidden_fcs.parameters():
            param.requires_grad = True

        for param in self.out_fcs.parameters():
            param.requires_grad = True


class LisaModel(LisaMetaModel, LlavaLlamaModel):
    def __init__(self, config, **kwargs):
        super(LisaModel, self).__init__(config, **kwargs)

        self.config.use_cache = False
        self.config.vision_tower = self.config.mm_vision_tower
        self.config.mm_vision_select_feature = "patch"
        self.config.image_aspect_ratio = "square"
        self.config.image_grid_pinpoints = None
        self.config.tune_mm_mlp_adapter = False
        self.config.freeze_mm_mlp_adapter = True
        self.config.pretrain_mm_mlp_adapter = None
        self.config.mm_use_im_patch_token = False


class LISAForCausalLM(LlavaLlamaForCausalLM):
    def __init__(self, config, **kwargs):
        config.mm_use_im_start_end = kwargs.pop("use_mm_start_end", True)
        config.mm_vision_tower = kwargs.get("vision_tower", "openai/clip-vit-large-patch14")
        self.seg_token_idx = kwargs.pop("seg_token_idx")

        super().__init__(config)

        self.model = LisaModel(config, **kwargs)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(self, **kwargs):
        if "past_key_values" in kwargs:
            return super().forward(**kwargs)
        return self.model_forward(**kwargs)

    def model_forward(self,
        images: torch.FloatTensor,
        images_clip: torch.FloatTensor,
        input_ids: torch.LongTensor,
        labels: torch.LongTensor,
        attention_masks: torch.LongTensor,
        #offset: torch.LongTensor,
        **kwargs,
    ):
        batch_size = images.shape[0]
        #assert batch_size == len(offset) - 1

        seg_token_mask = input_ids[:, 1:] == self.seg_token_idx #建一个bool mask，找[ACT] token的位置
        seg_token_mask = torch.cat(
            [seg_token_mask,
             torch.zeros((seg_token_mask.shape[0], 1)).bool().cuda()],dim=1,)  #在 seg_token_mask 的末尾添加一个全零列。为了确保掩码的长度与原始输入序列的长度一致。[bs, sequence_length]
        # hack for IMAGE_TOKEN_INDEX (we suppose that there is only one image, and it is in the front)
        seg_token_mask = torch.cat([torch.zeros((seg_token_mask.shape[0], 255)).bool().cuda(), seg_token_mask], dim=1,) #[bs, 255+sequence_length] 

        output = super().forward(
            images=images_clip,
            attention_mask=attention_masks,
            input_ids=input_ids,
            labels=labels,
            output_hidden_states=True,
        )
        output_hidden_states = output.hidden_states  

        hidden_states = []

        assert len(self.model.text_hidden_fcs) == 1
        hidden_states.append(self.model.text_hidden_fcs[0](output_hidden_states[-1]))

        last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)
        pred_embeddings = last_hidden_state[seg_token_mask]
        seg_token_counts = seg_token_mask.int().sum(-1)  # [bs, ]

        seg_token_offset = seg_token_counts.cumsum(-1)
        seg_token_offset = torch.cat([torch.zeros(1).long().cuda(), seg_token_offset], dim=0)

        seg_token_offset = seg_token_offset

        pred_embeddings_ = []
        for i in range(len(seg_token_offset) - 1):
            start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
            pred_embeddings_.append(pred_embeddings[start_i:end_i])
        pred_embeddings = pred_embeddings_

        pred_actions = []
        for i in pred_embeddings:
            pred_action = self.model.out_fcs[0](i)
            pred_actions.append(pred_action)

        model_output = output
        output = model_output.logits
        ce_loss = model_output.loss

        return pred_actions, ce_loss

