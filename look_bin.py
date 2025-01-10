import torch

def print_model_parameters(model_path):
    # 加载模型的状态字典
    state_dict = torch.load(model_path, map_location='cpu')

    # 遍历状态字典中的参数
    for name, param in state_dict.items():
        print(f"Parameter name: {name}, shape: {param.shape}")

if __name__ == "__main__":
    model_path = "/zhaohan/Wenxuan/3d_diffuser_actor/train_logs/Planner_Calvin/10021938lmpart-gpu4-accumulation-val_frequency5-step165700-step2105000-C192-B90-lr3e-4-DI1-20-H1-DT25-backboneclip-S256,256-R1-wd5e-3/0065099/pytorch_model.bin"
    print_model_parameters(model_path)
