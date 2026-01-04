import os
import json
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer

# 选择 tokenizer
tokenizer = AutoTokenizer.from_pretrained("/attached/remote-home2/yy/code/BLIP/BLIP/pretrain_pth/bert-base-uncased/bert-base-uncased")

# JSON 路径列表（替换为你本地实际路径）
json_paths =  [
  '/attached/remote-home2/zxk/sorted_dataset/data_labeled/QL_plane/crop_1024/QL_plane_1024_dollar_new.json',
  '/attached/remote-home2/zxk/sorted_dataset/data_labeled/QL_plane/crop_2048/QL_plane_2048_dollar_new.json',
  '/attached/remote-home2/zxk/sorted_dataset/data_labeled/QL_ship/crop_2048/QL_ship_2048_dollar_new.json',
  '/attached/remote-home2/zxk/sorted_dataset/data_labeled/QL_ship/crop_4096/QL_ship_4096_dollar_new.json',
  '/attached/remote-home2/zxk/sorted_dataset/data_labeled/gf3_plane/crop_512/GF3_plane_512_dollar_new.json',
  '/attached/remote-home2/yy/data/MultimodalLearning/airport/emw_GF3/final_refine_dollar.json',
  # # gf3 民船(3007+3838)
  '/attached/remote-home2/zxk/sorted_dataset/data_labeled/gf3_ship/crop_512/sar_image_analysis_gf3_ship_512_dollar_new_1.json',
  '/attached/remote-home2/zxk/sorted_dataset/data_labeled/gf3_ship/crop_1024/sar_image_analysis_gf3_ship_1024_dollar_new_1.json',
  # gf3 地物(610+5555+4109+22759+14299)
  '/attached/remote-home2/zxk/sorted_dataset/FUSAR-Map_label/sar_image_analysis_official_fusarmap-label_1024_toserver.json',
  '/attached/remote-home2/zxk/sorted_dataset/fusar_map/fusarmap/1024/sar_image_analysis_official_toserver.json',
  '/attached/remote-home2/zxk/sorted_dataset/gf3_airport/1024/sar_image_analysis_official_civil_toserver.json',
  '/attached/remote-home2/zxk/sorted_dataset/fusar_map/fusarmap/512_new/sar_image_analysis_official_fusarmap_512_toserver.json',
  '/attached/remote-home2/zxk/sorted_dataset/gf3_airport/512/sar_image_analysis_official_gf3-airport_512_toserver-1.json',

  # 齐鲁 地物（127+484+1133+430+4776+1325）
  '/attached/remote-home2/zxk/sorted_dataset/ql_port/5120/sar_image_analysis_official_ql_port_toserver.json',
  '/attached/remote-home2/zxk/sorted_dataset/ql_airport/5120/sar_image_analysis_official_ql_airport_toserver.json',
  '/attached/remote-home2/zxk/sorted_dataset/ql_airport/2560/sar_image_analysis_official_ql-airport_2560_toserver.json',
  '/attached/remote-home2/zxk/sorted_dataset/ql_port/2560/sar_image_analysis_official_qlport_2560_toserver.json',
  '/attached/remote-home2/zxk/sorted_dataset/ql_airport/1280/sar_image_analysis_qlairport_midpic_toserver_1280.json',
  '/attached/remote-home2/zxk/sorted_dataset/ql_port/1280/sar_image_analysis_qlport_midpic_toserver_1280.json',
  
  # X-SAR 地物（3715+14355）
  '/attached/remote-home2/zxk/data_second/X-SAR_toserver_1024/sar_image_analysis_official_x-sar_1024_toserver.json',
#   '/attached/remote-home2/zxk/data_second/X-SAR_toserver_512/sar_image_analysis_x-sar_512_toserver.json',

#   # gf3 地物第二批（42095+11390）
#   '/attached/remote-home2/zxk/data_second/gf3_server_512/sar_image_analysis_gf3-second_midpic_toserver_512.json',
#   '/attached/remote-home2/zxk/data_second/gf3_server_1024/sar_image_analysis_merged_gf3_toserver_1024.json'
  ]

all_lengths = []

for path in json_paths:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                caption = item.get("caption", "")
                tokens = tokenizer.tokenize(caption)
                all_lengths.append(len(tokens))
    except Exception as e:
        print(f"Error reading {path}: {e}")

# 统计信息
arr = np.array(all_lengths)
print(f"总样本数: {len(arr)}")
print(f"平均长度: {np.mean(arr):.2f}")
print(f"最大长度: {np.max(arr)}")
print(f"90% 样本长度 ≤ {np.percentile(arr, 90):.0f}")
print(f"95% 样本长度 ≤ {np.percentile(arr, 95):.0f}")
print(f"✅ 推荐 max_length = {int(np.percentile(arr, 95) + 10)}")

# 绘图
plt.hist(arr, bins=30, color='skyblue', edgecolor='black')
plt.title("Tokenized Caption Length Distribution")
plt.xlabel("Token Count")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.savefig("token_length_distribution_all.png")
print("📊 图已保存为 token_length_distribution_all.png")
