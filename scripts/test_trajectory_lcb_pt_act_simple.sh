main_dir=Planner_Calvin

dataset=/wangdonglin/calvin/packaged_ABC_D/training
valset=/wangdonglin/calvin/packaged_ABC_D/validation

lr=3e-4
wd=5e-3
dense_interpolation=1
interpolation_length=20
num_history=1
diffusion_timesteps=25
B=45
C=192
ngpus=8
backbone=clip
image_size="256,256"
relative_action=1
fps_subsampling_factor=3
lang_enhanced=1
gripper_loc_bounds=tasks/calvin_rel_traj_location_bounds_task_ABC_D.json
gripper_buffer=0.01
val_freq=5000
quaternion_format=wxyz

export PYTHONPATH=`pwd`:$PYTHONPATH

# torchrun --nproc_per_node $ngpus --master_port $RANDOM \
#     online_evaluation_calvin/evaluate_policy_lcb_pt_enrich.py \
#     --calvin_dataset_path /wangdonglin/calvin/task_ABC_D \
#     --calvin_model_path /dingpengxiang/Pengxiang/eccv2024/calvin/calvin_models \
#     --text_encoder clip \
#     --text_max_length 16 \
#     --tasks A B C D\
#     --backbone $backbone \
#     --gripper_loc_bounds $gripper_loc_bounds \
#     --gripper_loc_bounds_buffer $gripper_buffer \
#     --calvin_gripper_loc_bounds /wangdonglin/calvin/task_ABC_D/validation/statistics.yaml \
#     --embedding_dim $C \
#     --action_dim 7 \
#     --use_instruction 1 \
#     --rotation_parametrization 6D \
#     --diffusion_timesteps $diffusion_timesteps \
#     --interpolation_length $interpolation_length \
#     --num_history $num_history \
#     --relative_action $relative_action \
#     --fps_subsampling_factor $fps_subsampling_factor \
#     --lang_enhanced $lang_enhanced \
#     --save_video 0 \
#     --base_log_dir train_logs/${main_dir}/${run_log_dir}/eval_logs_pt_100_sy_enrich/ \
#     --quaternion_format $quaternion_format \
#     --checkpoint /storage/dingpengxiang/3d_diffuser_actor/train_logs/Planner_Calvin/10201547prompttuninglcbABC_D-gpu8-step167000-step2100000-C192-B15-lr3e-4-DI1-20-H1-DT25-backboneclip-S256,256-R1-wd5e-3/0090499.pth \
#     --llm_ckpt /storage/dingpengxiang/3d_diffuser_actor/train_logs/Planner_Calvin/10201547prompttuninglcbABC_D-gpu8-step167000-step2100000-C192-B15-lr3e-4-DI1-20-H1-DT25-backboneclip-S256,256-R1-wd5e-3/0090499
    # --checkpoint /storage/dingpengxiang/3d_diffuser_actor/train_logs/Planner_Calvin/10151511lcbfromscratchABC_D-gpu8-step1130000-step2130000-C192-B15-lr3e-4-DI1-20-H1-DT25-backboneclip-S256,256-R1-wd5e-3/0062499.pth \
    # --llm_ckpt /storage/dingpengxiang/3d_diffuser_actor/train_logs/Planner_Calvin/10151511lcbfromscratchABC_D-gpu8-step1130000-step2130000-C192-B15-lr3e-4-DI1-20-H1-DT25-backboneclip-S256,256-R1-wd5e-3/0062499
    # --checkpoint /storage/dingpengxiang/3d_diffuser_actor/train_logs/Planner_Calvin/10121724lcbABC_D-gpu8-step165000-step268000-C192-B15-lr3e-4-DI1-20-H1-DT25-backboneclip-S256,256-R1-wd5e-3/0067999.pth \
    # --llm_ckpt /storage/dingpengxiang/3d_diffuser_actor/train_logs/Planner_Calvin/10121724lcbABC_D-gpu8-step165000-step268000-C192-B15-lr3e-4-DI1-20-H1-DT25-backboneclip-S256,256-R1-wd5e-3/0067999
    # --checkpoint /storage/dingpengxiang/3d_diffuser_actor/train_logs/Planner_Calvin/10171918lcbABC_D-gpu8-step165000-step2100000-C192-B15-lr3e-4-DI1-20-H1-DT25-backboneclip-S256,256-R1-wd5e-3/0068999.pth \
    # --llm_ckpt /storage/dingpengxiang/3d_diffuser_actor/train_logs/Planner_Calvin/10171918lcbABC_D-gpu8-step165000-step2100000-C192-B15-lr3e-4-DI1-20-H1-DT25-backboneclip-S256,256-R1-wd5e-3/0068999

# torchrun --nproc_per_node $ngpus --master_port $RANDOM \
#     online_evaluation_calvin/evaluate_policy_lcb_pt_asy10.py \
#     --calvin_dataset_path /wangdonglin/calvin/task_ABC_D \
#     --calvin_model_path /dingpengxiang/Pengxiang/eccv2024/calvin/calvin_models \
#     --text_encoder clip \
#     --text_max_length 16 \
#     --tasks A B C D\
#     --backbone $backbone \
#     --gripper_loc_bounds $gripper_loc_bounds \
#     --gripper_loc_bounds_buffer $gripper_buffer \
#     --calvin_gripper_loc_bounds /wangdonglin/calvin/task_ABC_D/validation/statistics.yaml \
#     --embedding_dim $C \
#     --action_dim 7 \
#     --use_instruction 1 \
#     --rotation_parametrization 6D \
#     --diffusion_timesteps $diffusion_timesteps \
#     --interpolation_length $interpolation_length \
#     --num_history $num_history \
#     --relative_action $relative_action \
#     --fps_subsampling_factor $fps_subsampling_factor \
#     --lang_enhanced $lang_enhanced \
#     --save_video 0 \
#     --base_log_dir train_logs/${main_dir}/${run_log_dir}/eval_logs_pt_100_asy10_2/ \
#     --quaternion_format $quaternion_format \
#     --checkpoint /storage/dingpengxiang/3d_diffuser_actor/train_logs/Planner_Calvin/10201547prompttuninglcbABC_D-gpu8-step167000-step2100000-C192-B15-lr3e-4-DI1-20-H1-DT25-backboneclip-S256,256-R1-wd5e-3/0090499.pth \
#     --llm_ckpt /storage/dingpengxiang/3d_diffuser_actor/train_logs/Planner_Calvin/10201547prompttuninglcbABC_D-gpu8-step167000-step2100000-C192-B15-lr3e-4-DI1-20-H1-DT25-backboneclip-S256,256-R1-wd5e-3/0090499

# torchrun --nproc_per_node $ngpus --master_port $RANDOM \
#     online_evaluation_calvin/evaluate_policy_lcb_pt_asy20.py \
#     --calvin_dataset_path /wangdonglin/calvin/task_ABC_D \
#     --calvin_model_path /dingpengxiang/Pengxiang/eccv2024/calvin/calvin_models \
#     --text_encoder clip \
#     --text_max_length 16 \
#     --tasks A B C D\
#     --backbone $backbone \
#     --gripper_loc_bounds $gripper_loc_bounds \
#     --gripper_loc_bounds_buffer $gripper_buffer \
#     --calvin_gripper_loc_bounds /wangdonglin/calvin/task_ABC_D/validation/statistics.yaml \
#     --embedding_dim $C \
#     --action_dim 7 \
#     --use_instruction 1 \
#     --rotation_parametrization 6D \
#     --diffusion_timesteps $diffusion_timesteps \
#     --interpolation_length $interpolation_length \
#     --num_history $num_history \
#     --relative_action $relative_action \
#     --fps_subsampling_factor $fps_subsampling_factor \
#     --lang_enhanced $lang_enhanced \
#     --save_video 0 \
#     --base_log_dir train_logs/${main_dir}/${run_log_dir}/eval_logs_pt_100_asy20_2/ \
#     --quaternion_format $quaternion_format \
#     --checkpoint /storage/dingpengxiang/3d_diffuser_actor/train_logs/Planner_Calvin/10201547prompttuninglcbABC_D-gpu8-step167000-step2100000-C192-B15-lr3e-4-DI1-20-H1-DT25-backboneclip-S256,256-R1-wd5e-3/0090499.pth \
#     --llm_ckpt /storage/dingpengxiang/3d_diffuser_actor/train_logs/Planner_Calvin/10201547prompttuninglcbABC_D-gpu8-step167000-step2100000-C192-B15-lr3e-4-DI1-20-H1-DT25-backboneclip-S256,256-R1-wd5e-3/0090499

torchrun --nproc_per_node $ngpus --master_port $RANDOM \
    online_evaluation_calvin/evaluate_policy_lcb_pt_act_simple.py \
    --calvin_dataset_path /wangdonglin/calvin/task_ABC_D \
    --calvin_model_path /dingpengxiang/Pengxiang/eccv2024/calvin/calvin_models \
    --text_encoder clip \
    --text_max_length 16 \
    --tasks A B C D\
    --backbone $backbone \
    --gripper_loc_bounds $gripper_loc_bounds \
    --gripper_loc_bounds_buffer $gripper_buffer \
    --calvin_gripper_loc_bounds /wangdonglin/calvin/task_ABC_D/validation/statistics.yaml \
    --embedding_dim $C \
    --action_dim 7 \
    --use_instruction 1 \
    --rotation_parametrization 6D \
    --diffusion_timesteps $diffusion_timesteps \
    --interpolation_length $interpolation_length \
    --num_history $num_history \
    --relative_action $relative_action \
    --fps_subsampling_factor $fps_subsampling_factor \
    --lang_enhanced $lang_enhanced \
    --save_video 0 \
    --base_log_dir train_logs/${main_dir}/${run_log_dir}/eval_logs_pt_100_act_simple_2/ \
    --quaternion_format $quaternion_format \
    --checkpoint /storage/dingpengxiang/3d_diffuser_actor/train_logs/Planner_Calvin/1113simpleactprompttuninglcbABC_D-gpu8-step167000-step2100000-C192-B15-lr3e-4-DI1-20-H1-DT25-backboneclip-S256,256-R1-wd5e-3/0091499.pth \
    --llm_ckpt /storage/dingpengxiang/3d_diffuser_actor/train_logs/Planner_Calvin/1113simpleactprompttuninglcbABC_D-gpu8-step167000-step2100000-C192-B15-lr3e-4-DI1-20-H1-DT25-backboneclip-S256,256-R1-wd5e-3/0091499


torchrun --nproc_per_node $ngpus --master_port $RANDOM \
    online_evaluation_calvin/evaluate_policy_lcb_pt_act_simple_enrich.py \
    --calvin_dataset_path /wangdonglin/calvin/task_ABC_D \
    --calvin_model_path /dingpengxiang/Pengxiang/eccv2024/calvin/calvin_models \
    --text_encoder clip \
    --text_max_length 16 \
    --tasks A B C D\
    --backbone $backbone \
    --gripper_loc_bounds $gripper_loc_bounds \
    --gripper_loc_bounds_buffer $gripper_buffer \
    --calvin_gripper_loc_bounds /wangdonglin/calvin/task_ABC_D/validation/statistics.yaml \
    --embedding_dim $C \
    --action_dim 7 \
    --use_instruction 1 \
    --rotation_parametrization 6D \
    --diffusion_timesteps $diffusion_timesteps \
    --interpolation_length $interpolation_length \
    --num_history $num_history \
    --relative_action $relative_action \
    --fps_subsampling_factor $fps_subsampling_factor \
    --lang_enhanced $lang_enhanced \
    --save_video 0 \
    --base_log_dir train_logs/${main_dir}/${run_log_dir}/eval_logs_pt_100_act_simple_enrich_3/ \
    --quaternion_format $quaternion_format \
    --checkpoint /storage/dingpengxiang/3d_diffuser_actor/train_logs/Planner_Calvin/1113simpleactprompttuninglcbABC_D-gpu8-step167000-step2100000-C192-B15-lr3e-4-DI1-20-H1-DT25-backboneclip-S256,256-R1-wd5e-3/0091499.pth \
    --llm_ckpt /storage/dingpengxiang/3d_diffuser_actor/train_logs/Planner_Calvin/1113simpleactprompttuninglcbABC_D-gpu8-step167000-step2100000-C192-B15-lr3e-4-DI1-20-H1-DT25-backboneclip-S256,256-R1-wd5e-3/0091499
    

# torchrun --nproc_per_node $ngpus --master_port $RANDOM \
#     online_evaluation_calvin/evaluate_policy_lcb_pt_act_simple_enrich.py \
#     --calvin_dataset_path /wangdonglin/calvin/task_ABC_D \
#     --calvin_model_path /dingpengxiang/Pengxiang/eccv2024/calvin/calvin_models \
#     --text_encoder clip \
#     --text_max_length 16 \
#     --tasks A B C D\
#     --backbone $backbone \
#     --gripper_loc_bounds $gripper_loc_bounds \
#     --gripper_loc_bounds_buffer $gripper_buffer \
#     --calvin_gripper_loc_bounds /wangdonglin/calvin/task_ABC_D/validation/statistics.yaml \
#     --embedding_dim $C \
#     --action_dim 7 \
#     --use_instruction 1 \
#     --rotation_parametrization 6D \
#     --diffusion_timesteps $diffusion_timesteps \
#     --interpolation_length $interpolation_length \
#     --num_history $num_history \
#     --relative_action $relative_action \
#     --fps_subsampling_factor $fps_subsampling_factor \
#     --lang_enhanced $lang_enhanced \
#     --save_video 1 \
#     --base_log_dir train_logs/${main_dir}/${run_log_dir}/eval_logs_pt_100_act_simple_enrich/ \
#     --quaternion_format $quaternion_format \
#     --checkpoint /storage/dingpengxiang/3d_diffuser_actor/train_logs/Planner_Calvin/1113simpleactprompttuninglcbABC_D-gpu8-step167000-step2100000-C192-B15-lr3e-4-DI1-20-H1-DT25-backboneclip-S256,256-R1-wd5e-3/0091499.pth \
#     --llm_ckpt /storage/dingpengxiang/3d_diffuser_actor/train_logs/Planner_Calvin/1113simpleactprompttuninglcbABC_D-gpu8-step167000-step2100000-C192-B15-lr3e-4-DI1-20-H1-DT25-backboneclip-S256,256-R1-wd5e-3/0091499
    

# torchrun --nproc_per_node $ngpus --master_port $RANDOM \
#     online_evaluation_calvin/evaluate_policy_lcb_pt_asy40.py \
#     --calvin_dataset_path /wangdonglin/calvin/task_ABC_D \
#     --calvin_model_path /dingpengxiang/Pengxiang/eccv2024/calvin/calvin_models \
#     --text_encoder clip \
#     --text_max_length 16 \
#     --tasks A B C D\
#     --backbone $backbone \
#     --gripper_loc_bounds $gripper_loc_bounds \
#     --gripper_loc_bounds_buffer $gripper_buffer \
#     --calvin_gripper_loc_bounds /wangdonglin/calvin/task_ABC_D/validation/statistics.yaml \
#     --embedding_dim $C \
#     --action_dim 7 \
#     --use_instruction 1 \
#     --rotation_parametrization 6D \
#     --diffusion_timesteps $diffusion_timesteps \
#     --interpolation_length $interpolation_length \
#     --num_history $num_history \
#     --relative_action $relative_action \
#     --fps_subsampling_factor $fps_subsampling_factor \
#     --lang_enhanced $lang_enhanced \
#     --save_video 0 \
#     --base_log_dir train_logs/${main_dir}/${run_log_dir}/eval_logs_pt_100_asy40_2/ \
#     --quaternion_format $quaternion_format \
#     --checkpoint /storage/dingpengxiang/3d_diffuser_actor/train_logs/Planner_Calvin/10201547prompttuninglcbABC_D-gpu8-step167000-step2100000-C192-B15-lr3e-4-DI1-20-H1-DT25-backboneclip-S256,256-R1-wd5e-3/0090499.pth \
#     --llm_ckpt /storage/dingpengxiang/3d_diffuser_actor/train_logs/Planner_Calvin/10201547prompttuninglcbABC_D-gpu8-step167000-step2100000-C192-B15-lr3e-4-DI1-20-H1-DT25-backboneclip-S256,256-R1-wd5e-3/0090499


# torchrun --nproc_per_node $ngpus --master_port $RANDOM \
#     online_evaluation_calvin/evaluate_policy_lcb_pt_asy50.py \
#     --calvin_dataset_path /wangdonglin/calvin/task_ABC_D \
#     --calvin_model_path /dingpengxiang/Pengxiang/eccv2024/calvin/calvin_models \
#     --text_encoder clip \
#     --text_max_length 16 \
#     --tasks A B C D\
#     --backbone $backbone \
#     --gripper_loc_bounds $gripper_loc_bounds \
#     --gripper_loc_bounds_buffer $gripper_buffer \
#     --calvin_gripper_loc_bounds /wangdonglin/calvin/task_ABC_D/validation/statistics.yaml \
#     --embedding_dim $C \
#     --action_dim 7 \
#     --use_instruction 1 \
#     --rotation_parametrization 6D \
#     --diffusion_timesteps $diffusion_timesteps \
#     --interpolation_length $interpolation_length \
#     --num_history $num_history \
#     --relative_action $relative_action \
#     --fps_subsampling_factor $fps_subsampling_factor \
#     --lang_enhanced $lang_enhanced \
#     --save_video 0 \
#     --base_log_dir train_logs/${main_dir}/${run_log_dir}/eval_logs_pt_100_asy50_2/ \
#     --quaternion_format $quaternion_format \
#     --checkpoint /storage/dingpengxiang/3d_diffuser_actor/train_logs/Planner_Calvin/10201547prompttuninglcbABC_D-gpu8-step167000-step2100000-C192-B15-lr3e-4-DI1-20-H1-DT25-backboneclip-S256,256-R1-wd5e-3/0090499.pth \
#     --llm_ckpt /storage/dingpengxiang/3d_diffuser_actor/train_logs/Planner_Calvin/10201547prompttuninglcbABC_D-gpu8-step167000-step2100000-C192-B15-lr3e-4-DI1-20-H1-DT25-backboneclip-S256,256-R1-wd5e-3/0090499

# torchrun --nproc_per_node $ngpus --master_port $RANDOM \
#     online_evaluation_calvin/evaluate_policy_lcb_pt.py \
#     --calvin_dataset_path /wangdonglin/calvin/task_ABC_D \
#     --calvin_model_path /dingpengxiang/Pengxiang/eccv2024/calvin/calvin_models \
#     --text_encoder clip \
#     --text_max_length 16 \
#     --tasks A B C D\
#     --backbone $backbone \
#     --gripper_loc_bounds $gripper_loc_bounds \
#     --gripper_loc_bounds_buffer $gripper_buffer \
#     --calvin_gripper_loc_bounds /wangdonglin/calvin/task_ABC_D/validation/statistics.yaml \
#     --embedding_dim $C \
#     --action_dim 7 \
#     --use_instruction 1 \
#     --rotation_parametrization 6D \
#     --diffusion_timesteps $diffusion_timesteps \
#     --interpolation_length $interpolation_length \
#     --num_history $num_history \
#     --relative_action $relative_action \
#     --fps_subsampling_factor $fps_subsampling_factor \
#     --lang_enhanced $lang_enhanced \
#     --save_video 0 \
#     --base_log_dir train_logs/${main_dir}/${run_log_dir}/eval_logs_pt_100_asy60_2/ \
#     --quaternion_format $quaternion_format \
#     --checkpoint /storage/dingpengxiang/3d_diffuser_actor/train_logs/Planner_Calvin/10201547prompttuninglcbABC_D-gpu8-step167000-step2100000-C192-B15-lr3e-4-DI1-20-H1-DT25-backboneclip-S256,256-R1-wd5e-3/0090499.pth \
#     --llm_ckpt /storage/dingpengxiang/3d_diffuser_actor/train_logs/Planner_Calvin/10201547prompttuninglcbABC_D-gpu8-step167000-step2100000-C192-B15-lr3e-4-DI1-20-H1-DT25-backboneclip-S256,256-R1-wd5e-3/0090499

