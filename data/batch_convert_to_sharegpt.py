import json
import os
import glob

# Configuration
DATASET_DIR = "/Users/xiezejing/PycharmProjects/prototypical networks/TV_Assistant/data/dataset/"

def convert_to_sharegpt(input_file: str, output_file: str) -> None:
    """
    将单个QA格式的JSON文件转换为ShareGPT格式
    
    Args:
        input_file: 输入JSON文件路径，格式为[{"Q": "问题", "A": "回答"}, ...]
        output_file: 输出ShareGPT格式JSON文件路径
    """
    # 读取原始数据
    with open(input_file, 'r', encoding='utf-8') as f:
        qa_data = json.load(f)
    
    print(f"  读取到 {len(qa_data)} 条QA数据")
    
    # 转换为ShareGPT格式
    sharegpt_data = []
    category = os.path.basename(input_file).replace('.json', '')
    
    for idx, qa_item in enumerate(qa_data):
        # 创建对话结构
        conversation = {
            "id": f"conv_{category}_{idx+1:04d}",
            "conversations": [
                {
                    "from": "human",
                    "value": qa_item["Q"]
                },
                {
                    "from": "gpt",
                    "value": qa_item["A"]
                }
            ],
            "category": category
        }
        
        sharegpt_data.append(conversation)
    
    print(f"  转换完成，生成 {len(sharegpt_data)} 条ShareGPT格式数据")
    
    # 保存转换后的数据
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(sharegpt_data, f, ensure_ascii=False, indent=2)
    
    print(f"  数据已保存到: {output_file}")

def batch_convert() -> None:
    """
    批量转换所有QA格式的JSON文件为ShareGPT格式
    """
    print(f"开始批量转换...")
    print(f"数据集目录: {DATASET_DIR}")
    
    # 获取所有需要转换的JSON文件
    json_files = glob.glob(os.path.join(DATASET_DIR, "*.json"))
    
    # 过滤掉已转换的文件（文件名包含_sharegpt）
    files_to_convert = [f for f in json_files if "_sharegpt" not in f]
    
    print(f"\n找到 {len(files_to_convert)} 个需要转换的文件:")
    for f in files_to_convert:
        print(f"  - {os.path.basename(f)}")
    
    # 逐个转换
    for input_file in files_to_convert:
        print(f"\n转换文件: {os.path.basename(input_file)}")
        output_file = input_file.replace(".json", "_sharegpt.json")
        convert_to_sharegpt(input_file, output_file)
    
    print(f"\n✅ 批量转换完成！")
    
    # 显示转换结果统计
    print(f"\n转换结果统计:")
    sharegpt_files = glob.glob(os.path.join(DATASET_DIR, "*_sharegpt.json"))
    for f in sharegpt_files:
        with open(f, 'r', encoding='utf-8') as fp:
            data = json.load(fp)
        print(f"  - {os.path.basename(f)}: {len(data)} 条数据")

if __name__ == "__main__":
    batch_convert()
