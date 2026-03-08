import json
import os

# Configuration
INPUT_FILE = "/Users/xiezejing/PycharmProjects/prototypical networks/TV_Assistant/data/dataset/历史人文.json"
OUTPUT_FILE = "/Users/xiezejing/PycharmProjects/prototypical networks/TV_Assistant/data/dataset/历史人文_sharegpt.json"

def convert_to_sharegpt(input_file: str, output_file: str) -> None:
    """
    将QA格式的JSON文件转换为ShareGPT格式
    
    Args:
        input_file: 输入JSON文件路径，格式为[{"Q": "问题", "A": "回答"}, ...]
        output_file: 输出ShareGPT格式JSON文件路径
    """
    # 读取原始数据
    with open(input_file, 'r', encoding='utf-8') as f:
        qa_data = json.load(f)
    
    print(f"读取到 {len(qa_data)} 条QA数据")
    
    # 转换为ShareGPT格式
    sharegpt_data = []
    
    for idx, qa_item in enumerate(qa_data):
        # 创建对话结构
        conversation = {
            "id": f"conv_{idx+1:04d}",
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
            "category": "历史人文"
        }
        
        sharegpt_data.append(conversation)
    
    print(f"转换完成，生成 {len(sharegpt_data)} 条ShareGPT格式数据")
    
    # 保存转换后的数据
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(sharegpt_data, f, ensure_ascii=False, indent=2)
    
    print(f"数据已保存到: {output_file}")
    
    # 验证输出文件
    with open(output_file, 'r', encoding='utf-8') as f:
       验证数据 = json.load(f)
    
    print(f"输出文件验证成功，包含 {len(验证数据)} 条数据")
    print(f"第一条数据示例:")
    print(json.dumps(验证数据[0], ensure_ascii=False, indent=2))

if __name__ == "__main__":
    convert_to_sharegpt(INPUT_FILE, OUTPUT_FILE)
