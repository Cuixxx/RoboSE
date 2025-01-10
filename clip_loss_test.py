# import torch
# import torch.nn.functional as F
# from transformers import CLIPModel, CLIPProcessor

# # 加载 CLIP 模型和处理器
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model_id = "/zhaohan/Wenxuan/clip-vit-large-patch14"
# model = CLIPModel.from_pretrained(model_id).to(device)
# processor = CLIPProcessor.from_pretrained(model_id)

# # 定义要比较的两个句子
# text1 = ""
# text2 = "he is very tall"

# # 将文本转换为 CLIP 嵌入
# with torch.no_grad():
#     inputs = processor(text=[text1, text2], return_tensors="pt", padding=True, truncation=True)
#     inputs = {key: value.to(device) for key, value in inputs.items()}
#     text_embeds = model.get_text_features(**inputs)

# # 计算余弦相似度
# cosine_sim = F.cosine_similarity(text_embeds[0], text_embeds[1], dim=0)

# print(f"Cosine similarity between '{text1}' and '{text2}': {cosine_sim.item()}")


import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor

# 加载 CLIP 模型和处理器
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "/zhaohan/Wenxuan/clip-vit-large-patch14"
model = CLIPModel.from_pretrained(model_id).to(device)
processor = CLIPProcessor.from_pretrained(model_id)

# 定义要比较的两个句子
text1 = ""  # 可以保留空字符串，但需确保生成的是全零向量
text2 = "he is very tall"

# 将文本转换为 CLIP 嵌入
with torch.no_grad():
    inputs = processor(text=[text1, text2], return_tensors="pt", padding=True, truncation=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    text_embeds = model.get_text_features(**inputs)

# 如果text1不是全零，手动生成一个全零向量
if not torch.equal(text_embeds[0], torch.zeros_like(text_embeds[0])):
    text_embeds[0] = torch.zeros_like(text_embeds[0])

# 计算余弦相似度
cosine_sim = F.cosine_similarity(text_embeds[0], text_embeds[1], dim=0)

print(f"Cosine similarity between zero vector and '{text2}': {cosine_sim.item()}")
