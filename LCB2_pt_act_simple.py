from typing import List
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BitsAndBytesConfig, CLIPVisionModel


from peft import LoraConfig, get_peft_model
from model.llava.model.language_model.llava_llama import (LlavaLlamaForCausalLM, LlavaLlamaModel)
from datasets.utils_llcb import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX)

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
        out_dim = self.config.out_dim
        # text_fc = [nn.Linear(in_dim, in_dim), nn.ReLU(inplace=True), nn.Linear(in_dim, out_dim), nn.Dropout(0.0)]
        text_fc = [nn.Linear(in_dim, out_dim)]
        self.text_hidden_fcs = nn.ModuleList([nn.Sequential(*text_fc)])
        self.text_hidden_fcs.train()

        # Action prediction
        self.pred_act_mlps = nn.Linear(in_dim, in_dim//2)
        self.pred_pos_act = nn.Linear(in_dim//2, 3) # arm action
        self.pred_rot_act = nn.Linear(in_dim//2, 6) # arm action
        self.pred_gripper_act = nn.Linear(in_dim//2, 1) # gripper action (binary)

        for param in self.text_hidden_fcs.parameters():
            param.requires_grad = True

        # if config.pooling == 'max':
        #     self.global_1d_pool = nn.AdaptiveMaxPool1d(1)
        # else:
        self.global_1d_pool = nn.AdaptiveAvgPool1d(1)


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
        self.seg_token_idx = 32003

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
        images_clip: torch.FloatTensor,
        input_ids: torch.LongTensor,
        labels: torch.LongTensor,
        attention_masks: torch.LongTensor,
        tokenizer,
        #offset: torch.LongTensor,
        **kwargs,
    ):

        # import pdb;pdb.set_trace()
        seg_token_mask = input_ids[:, 1:] == self.seg_token_idx #建一个bool mask，找[ACT] token的位置
        # seg_token_mask = torch.cat(
        #     [seg_token_mask,
        #      torch.zeros((seg_token_mask.shape[0], 1)).bool().cuda()],dim=1,)  #在 seg_token_mask 的末尾添加一个全零列。为了确保掩码的长度与原始输入序列的长度一致。[bs, sequence_length]
        # # hack for IMAGE_TOKEN_INDEX (we suppose that there is only one image, and it is in the front)
        seg_token_mask = torch.cat([torch.zeros((seg_token_mask.shape[0], 256)).bool().cuda(), seg_token_mask], dim=1,) #[bs, 255+sequence_length] 255+82=337

        output = super().forward(
            images=images_clip,
            attention_mask=attention_masks,
            input_ids=input_ids,
            # labels=labels, #不输入label就不会计算loss
            output_hidden_states=True,
        )
        # print(output)
        # import pdb;pdb.set_trace()
        output_hidden_states = output.hidden_states 
        # print("output_hidden_states:", output_hidden_states.shape)
        # import pdb;pdb.set_trace()

        hidden_states = []

        assert len(self.model.text_hidden_fcs) == 1
        hidden_states.append(self.model.text_hidden_fcs[0](output_hidden_states[-1].float()))
        # import pdb; pdb.set_trace()
        action_latents = self.model.pred_act_mlps(output_hidden_states[-1][seg_token_mask].float())
        pos_pred = self.model.pred_pos_act(action_latents)
        rot_pred = self.model.pred_rot_act(action_latents)
        gripper_pred = self.model.pred_gripper_act(action_latents)
        act_pred = torch.cat([pos_pred,rot_pred,gripper_pred],dim=-1)
        # hidden_states.append(self.model.text_hidden_fcs[0](output_hidden_states[-1]))

        last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)
        # pred_embeddings = self.model.global_1d_pool(last_hidden_state.permute(0,2,1)).squeeze(-1)
        pred_embeddings = last_hidden_state[seg_token_mask]
        # print("这里pred_embeddings内容为", pred_embeddings)#这里已经没了
        # seg_token_counts = seg_token_mask.int().sum(-1)  # [bs, ]
        # print("seg_token_counts:", seg_token_counts)
        # seg_token_offset = seg_token_counts.cumsum(-1)
        # seg_token_offset = torch.cat([torch.zeros(1).long().cuda(), seg_token_offset], dim=0)

        # pred_embeddings_ = []
        # for i in range(len(seg_token_offset) - 1):
        #     start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
        #     pred_embeddings_.append(pred_embeddings[start_i:end_i])
        # pred_embeddings = torch.cat(pred_embeddings_, dim=0)

        # model_output = output
        # 注释掉的几行代码用于打印训练时的输出
        # output = model_output.logits
        # output_ids = torch.argmax(output, dim=-1)
        # output_ids = output_ids[0][output_ids[0] != IMAGE_TOKEN_INDEX]
        # text_output = tokenizer.decode(output_ids, skip_special_tokens=False)
        # text_output = text_output.replace("\n", "").replace("  ", " ")
        # print("text_output: ", text_output)

        # ce_loss = model_output.loss
        ce_loss = 0
        return pred_embeddings, ce_loss, act_pred
    
    def evaluate(
        self,
        images_clip,
        input_ids,
        attention_masks,
        tokenizer=None
    ):
        with torch.no_grad():
            seg_token_mask = input_ids[:, 1:] == self.seg_token_idx #建一个bool mask，找[ACT] token的位置
            # seg_token_mask = torch.cat(
            # [seg_token_mask,
            #  torch.zeros((seg_token_mask.shape[0], 1)).bool().cuda()],dim=1,)  #在 seg_token_mask 的末尾添加一个全零列。为了确保掩码的长度与原始输入序列的长度一致。[bs, sequence_length]
            # # hack for IMAGE_TOKEN_INDEX (we suppose that there is only one image, and it is in the front)
            seg_token_mask = torch.cat([torch.zeros((seg_token_mask.shape[0], 256)).bool().cuda(), seg_token_mask], dim=1,) #[bs, 255+sequence_length] 255+82=337

            
            output = super().forward(
            images=images_clip,
            attention_mask=attention_masks,
            input_ids=input_ids,
            # labels=labels, #不输入label就不会计算loss
            output_hidden_states=True)
            output_hidden_states = output.hidden_states
            hidden_states = []
            
            assert len(self.model.text_hidden_fcs) == 1
            # import pdb; pdb.set_trace()
            hidden_states.append(self.model.text_hidden_fcs[0](output_hidden_states))
            last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)
            pred_embeddings = last_hidden_state[seg_token_mask]
            # if tokenizer is not None:
            #     action_idx = tokenizer(['left right up down forward back static']).input_ids
            #     # import pdb; pdb.set_trace()
            #     seg_emb = output_hidden_states[seg_token_mask]
            #     output_ids = self.lm_head(seg_emb)
            #     action_prob = F.softmax(output_ids[0][action_idx])
                # print('left:',action_prob[0],'right:',action_prob[1],'up:',action_prob[2],'down:',action_prob[3],'forward:',action_prob[4],'back:',action_prob[5],'static:', action_prob[6])
                # _, topk_ids = torch.topk(output_ids, 10, largest=True)
                # text = tokenizer.decode(topk_ids[0], skip_special_tokens=False)
                # print(text)


            # pred_embeddings = self.model.global_1d_pool(last_hidden_state.permute(1,0)).permute(1,0)
        
        return None, pred_embeddings
    
    # def evaluate(
    #     self,
    #     images_clip,
    #     input_ids,
    #     max_new_tokens=32,
    #     tokenizer=None,
    # ):
    #     with torch.no_grad():
    #         outputs = self.generate(
    #             images=images_clip,
    #             input_ids=input_ids,
    #             max_new_tokens=max_new_tokens,
    #             num_beams=1,
    #             output_hidden_states=True,
    #             return_dict_in_generate=True,
    #         )
    #         # import pdb;pdb.set_trace()
    #         output_hidden_states = outputs.hidden_states[-1]
    #         output_ids = outputs.sequences

    #         seg_token_mask = output_ids[:, 1:] == self.seg_token_idx
    #         # hack for IMAGE_TOKEN_INDEX (we suppose that there is only one image, and it is in the front)
    #         seg_token_mask = torch.cat(
    #             [
    #                 torch.zeros((seg_token_mask.shape[0], 255)).bool().cuda(),
    #                 seg_token_mask,
    #             ],
    #             dim=1,
    #         )

    #         hidden_states = []

    #         assert len(self.model.text_hidden_fcs) == 1
    #         hidden_states.append(self.model.text_hidden_fcs[0](output_hidden_states))

    #         last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)
    #         pred_embeddings = last_hidden_state[seg_token_mask]

    #         seg_token_counts = seg_token_mask.int().sum(-1)  # [bs, ]
    #         seg_token_offset = seg_token_counts.cumsum(-1)
    #         seg_token_offset = torch.cat(
    #             [torch.zeros(1).long().cuda(), seg_token_offset], dim=0
    #         )

    #         pred_embeddings_ = []
    #         for i in range(len(seg_token_offset) - 1):
    #             start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
    #             pred_embeddings_.append(pred_embeddings[start_i:end_i])
    #         pred_embeddings = pred_embeddings_[0]#这里可能没这个[0]


    #     return output_ids, pred_embeddings


