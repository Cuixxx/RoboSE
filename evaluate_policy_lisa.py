import os
import argparse
import transformers
from transformers import CLIPImageProcessor
import cv2
import torch
import torch.nn.functional as F
from model.llava.mm_utils import tokenizer_image_token
from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_PATCH_TOKEN)
from peft import PeftModel
from model.llava.constants import (DEFAULT_IMAGE_TOKEN, IGNORE_INDEX)

from peft import LoraConfig, get_peft_model
from model.llava import conversation as conversation_lib
from model.llava.model.language_model.llava_llama import (LlavaLlamaForCausalLM, LlavaLlamaModel)
from LCB2 import LISAForCausalLM

def LLM_parse_args():
    parser = argparse.ArgumentParser(description="LISA Model Training")
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument("--lora_alpha", default=16, type=int)
    parser.add_argument("--lora_dropout", default=0.05, type=float)
    parser.add_argument("--lora_target_modules", default="q_proj,v_proj", type=str)
    parser.add_argument("--llava_dir", default="/zhaohan/Wenxuan/LLaVA-7B-Lightening-v1-1/LLaVA-7B-Lightening-v1-1/LLaVA-7B-Lightening-v1-1", type=str)
    parser.add_argument("--vision_tower", default="/zhaohan/Wenxuan/clip-vit-large-patch14", type=str)
    return parser.parse_args()

def Add_LoRA(model, tokenizer):
        # ===============Add Lora===============
    lora_r = 8
    if lora_r > 0:

        def find_linear_layers(model, lora_target_modules):
            cls = torch.nn.Linear
            lora_module_names = set()
            for name, module in model.named_modules():
                if (
                    isinstance(module, cls)
                    and all(
                        [
                            x not in name
                            for x in [
                                "visual_model",
                                "vision_tower",
                                "mm_projector",
                                "text_hidden_fcs",
                            ]
                        ]
                    )
                    and any([x in name for x in lora_target_modules])
                ):
                    lora_module_names.add(name)
            return sorted(list(lora_module_names))

        lora_alpha = LLM_args.lora_alpha
        lora_dropout = LLM_args.lora_dropout
        lora_target_modules = find_linear_layers(
            model, LLM_args.lora_target_modules.split(",")
        )
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

        model.print_trainable_parameters()

    model.resize_token_embeddings(len(tokenizer))
    
    for n, p in model.named_parameters():
        if any(
            [
                x in n
                for x in ["lm_head", "embed_tokens", "mask_decoder", "text_hidden_fcs"]
            ]
        ):
            print("n: ", n, "p.shape: ", p.shape)
            p.requires_grad = True
    return model

def Model_init(LLM_args):
    
    clip_image_processor = CLIPImageProcessor.from_pretrained(LLM_args.vision_tower)
    # 加载 tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(LLM_args.llava_dir, cache_dir=None, model_max_length=512, padding_side="right", use_fast=False)
    # 设置填充token
    tokenizer.pad_token = tokenizer.unk_token
    # 添加新的token [SEG]
    num_added_tokens = tokenizer.add_tokens("<ACT>")
    seg_token_idx = tokenizer("<ACT>", add_special_tokens=False).input_ids[0]

    tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
    # 添加新的token [REG]
    num_added_tokens = tokenizer.add_tokens("<REJ>")
    rej_token_idx = tokenizer("<REJ>", add_special_tokens=False).input_ids[0]
    
    model_args = {
        "seg_token_idx": seg_token_idx,
        "out_dim": 512,
        "vision_tower": LLM_args.vision_tower,
        "use_mm_start_end": True,
    }
    
    model = LISAForCausalLM.from_pretrained(LLM_args.llava_dir, torch_dtype=torch_dtype, low_cpu_mem_usage=True, **model_args).cuda()
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype, device=0)
    #不add lora, 因为 PeftModel.from_pretrained 会自动将 LoRA 层应用到模型中。如果您在加载模型前手动添加了 LoRA 层，可能会导致模型结构重复或不匹配。
    # model = Add_LoRA(model, tokenizer)
    
    return clip_image_processor, tokenizer, model

def preprocess(x: torch.Tensor) -> torch.Tensor:
    """处理输入图像."""
    # Normalize colors
    img_size = 224
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    x = (x - pixel_mean) / pixel_std

    # Pad
    h, w = x.shape[-2:]
    padh = img_size - h
    padw = img_size - w
    x = F.pad(x, (0, padw, 0, padh))
    return x

def input_processing_batch(image_dir_list, conv_list):
    '''
    preprocess input (image/text)
    '''
    image_clip_list = []
    image_list = []
    for image_dir in image_dir_list:
        image = cv2.imread(image_dir)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
        image_clip = clip_image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0].to(torch.bfloat16).cuda() # [3, 224, 224]
        image_clip_list.append(image_clip.unsqueeze(0))
        image = preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous()).to(torch.bfloat16).cuda()                   # [3, 224, 224]
        image_list.append(image.unsqueeze(0))
        
    image = torch.cat(image_list, dim=0)
    image_clip = torch.cat(image_clip_list, dim=0)
    
    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors="pt") for prompt in conv_list]
    
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id).cuda()  #相同batch中进行补齐
    attention_masks = input_ids.ne(tokenizer.pad_token_id)
    
    targets = input_ids.clone()
    
    return image_clip, image, input_ids, attention_masks, targets

if __name__ == "__main__":
    LLM_args = LLM_parse_args()
    torch_dtype = torch.bfloat16
    import time
    
    #===============模型初始化（CLIP/tokenizer）==============
    
    clip_image_processor, tokenizer, LCB_model = Model_init(LLM_args)
    LCB_model.resize_token_embeddings(len(tokenizer))
    # 加载权重文件的状态字典
    state_dict = torch.load('/zhaohan/Wenxuan/3d_diffuser_actor/train_logs/Planner_Calvin/10041015lcbABC_D-gpu8-step167500-step275000-C192-B30-lr3e-4-DI1-20-H1-DT25-backboneclip-S256,256-R1-wd5e-3/0074999/pytorch_model.bin', map_location='cpu')
    # 创建一个新的状态字典，修改参数名称
    new_state_dict = {}
    for key, value in state_dict.items():
        # 检查参数名称是否以 'base_model.model.' 开头
        if key.startswith('base_model.model.'):
            # 移除前缀
            new_key = key.replace('base_model.model.', '')
        else:
            new_key = key
        new_state_dict[new_key] = value
    # import pdb; pdb.set_trace()
    # 将权重加载到模型中
    LCB_model.load_state_dict(new_state_dict, strict=False)
    # LCB_model.load_state_dict(new_state_dict, strict=True)
    #加载lora部分权重
    peft_model = PeftModel.from_pretrained(LCB_model, '/zhaohan/Wenxuan/3d_diffuser_actor/train_logs/Planner_Calvin/10041015lcbABC_D-gpu8-step167500-step275000-C192-B30-lr3e-4-DI1-20-H1-DT25-backboneclip-S256,256-R1-wd5e-3/0074999')   

    for name, param in peft_model.named_parameters():
        if "lm_head" in name:
            print(f"Parameter name: {name}, parameters: {param}")

    LLM_model = peft_model.merge_and_unload()
    LLM_model.cuda()  
    # LCB_model = peft_model.model 
    #检查了一下，加载不同的checkpoint模型的参数确实不同，因此参数应该是保存下来并且加载进来了
    #================输入输出===================
    image_dir_list = ['/zhaohan/Wenxuan/3d_diffuser_actor/output.jpg']
    conv = conversation_lib.conv_templates['llava_v1'].copy()
    conv.messages = []
    conversation_list = ["A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <im_start><image><im_end>\nCan you control the robot to push the red block? ASSISTANT:"]
    
    #image_clip, image, input_ids, attention_masks, targets = input_processing(image_dir=image_dir, conv_list=conversation_list)
    image_clip, image, input_ids, attention_masks, targets = input_processing_batch(image_dir_list=image_dir_list, conv_list=conversation_list)
 
    # LCB_model.eval()
    LLM_model.eval()
    start_time = time.time()
    # output_ids, pred_embeddings = LCB_model.evaluate(image_clip, input_ids, max_new_tokens=512, tokenizer=tokenizer,)
    # import pdb; pdb.set_trace()
    output_ids, pred_embeddings = LLM_model.evaluate(image_clip, input_ids, max_new_tokens=512, tokenizer=tokenizer,)#input_ids.size()=torch.Size([1, 53])
    output_ids = output_ids[0][output_ids[0] != -200]

    text_output = tokenizer.decode(output_ids, skip_special_tokens=False)
    text_output = text_output.replace("\n", "").replace("  ", " ")
    over_time2 = time.time()
    
    print(over_time2 - start_time)
    print("text_output: ", text_output)
    print(pred_embeddings.shape)
    
    
    
    
    