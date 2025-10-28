import os
import json
import pandas as pd
import numpy as np


dataset_path = '/data/nvme0/zhiheng/dataset/piper-pickcube'
save_path = '/data/nvme0/zhiheng/dataset/piper-pickcube-jointctrl1'

def modify_state(state_list):
    return np.concatenate([state_list[:6], state_list[-1:]])    # 关节角（6维）+ 夹爪宽度（1维度）
    
def modify_action(action_list):
    return np.concatenate([action_list[:6], action_list[-1:]])
    
os.system(f'rm -r {save_path}')
os.system(f'cp -r {dataset_path} {save_path}')
parquet_files_path = os.path.join(save_path, 'data/chunk-000')
for file in os.listdir(parquet_files_path):
    if file.endswith('.parquet'):
        file_path = os.path.join(parquet_files_path, file)
        df = pd.read_parquet(file_path)
        # 修改state和action
        df['observation.state']= df['observation.state'].apply(modify_state)
        df["action"] = df["action"].apply(modify_state)
        df.to_parquet(file_path)

# 修改统计量
stats_save = []
stats_path = os.path.join(save_path, 'meta/episodes_stats.jsonl')
with open(stats_path, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue  # 跳过空行
        obj = json.loads(line)  # 每行是一个独立的 JSON 对象
        parqurt_file = os.path.join(save_path, 'data/chunk-000', f"episode_{obj['episode_index']:06d}" + '.parquet')
        df = pd.read_parquet(parqurt_file)
        state_array = np.stack(df["observation.state"].to_numpy())
        action_array = np.stack(df["action"].to_numpy())
        obj['stats']['observation.state']['max'] = np.max(state_array,axis=0).tolist()
        obj['stats']['observation.state']['min'] = np.min(state_array,axis=0).tolist()
        obj['stats']['observation.state']['mean'] = np.mean(state_array,axis=0).tolist()
        obj['stats']['observation.state']['std'] = np.std(state_array,axis=0).tolist()
        obj['stats']['action']['max'] = np.max(action_array,axis=0).tolist()
        obj['stats']['action']['min'] = np.min(action_array,axis=0).tolist()
        obj['stats']['action']['mean'] = np.mean(action_array,axis=0).tolist()
        obj['stats']['action']['std'] = np.std(action_array,axis=0).tolist()
        stats_save.append(obj)

with open(stats_path, "w", encoding="utf-8") as f:
    for obj in stats_save:
        json_line = json.dumps(obj, ensure_ascii=False)
        f.write(json_line + "\n")

