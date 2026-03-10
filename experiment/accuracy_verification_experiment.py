import json
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import os
from typing import List, Dict, Tuple
from tqdm import tqdm

class QwenSelfValidator:
    def __init__(self, model_path: str = "/root/autodl-tmp/cache/qwen/Qwen2.5-VL-7B-Instruct"):
        """
        初始化模型和处理器
        """
        print(f"正在加载模型: {model_path}")
        
        # 加载模型
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        self.model.eval()
        
        # 加载处理器
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        print("模型加载完成！")

    def generate_answer(self, question: str, image_path: str = None, max_new_tokens: int = 512) -> str:
        """
        第一阶段：生成答案
        """
        # 构建消息格式
        if image_path and os.path.exists(image_path):
            # 多模态输入（图文）
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": question},
                    ],
                }
            ]
        else:
            # 纯文本输入
            messages = [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": question}],
                }
            ]

        # 处理输入
        text = self.processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        # 生成答案
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # 使用贪婪解码确保确定性
            )
        
        # 解码输出
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        answer = self.processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]
        
        return answer.strip()

    def validate_answer(self, question: str, generated_answer: str, ground_truth: str = None) -> Tuple[bool, str]:
        """
        第二阶段：验证答案正确性
        使用模型自身判断生成的答案是否正确
        """
        # 构建验证提示
        validation_prompt = f"""请判断以下答案是否正确。如果正确请回答"正确"，如果错误请回答"错误"并简要说明原因。

问题：{question}

生成的答案：{generated_answer}"""

        if ground_truth:
            validation_prompt += f"\n\n标准答案：{ground_truth}"

        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": validation_prompt}],
            }
        ]

        # 处理输入
        text = self.processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        inputs = self.processor(
            text=[text],
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        # 生成验证结果
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
            )
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        validation_result = self.processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0].strip()

        # 解析验证结果
        is_correct = "正确" in validation_result and "错误" not in validation_result[:10]
        
        return is_correct, validation_result

    def process_single_item(self, item: Dict) -> Dict:
        """
        处理单个问题
        """
        question = item.get("Question", item.get("question", ""))
        image_path = item.get("image", item.get("image_path", None))
        ground_truth = item.get("Answer", item.get("answer", item.get("ground_truth", None)))
        
        print(f"\n处理问题: {question[:50]}...")
        
        # 阶段1：生成答案
        generated_answer = self.generate_answer(question, image_path)
        print(f"生成答案: {generated_answer[:100]}...")
        
        # 阶段2：验证答案
        is_correct, validation_detail = self.validate_answer(question, generated_answer, ground_truth)
        print(f"验证结果: {'正确' if is_correct else '错误'}")
        
        return {
            "question": question,
            "generated_answer": generated_answer,
            "ground_truth": ground_truth,
            "is_correct": is_correct,
            "validation_detail": validation_detail
        }

    def process_dataset(self, json_path: str) -> List[Dict]:
        """
        处理整个数据集
        """
        # 读取JSON文件
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 支持多种格式：单个对象或列表
        if isinstance(data, dict):
            data = [data]
        
        results = []
        correct_count = 0
        
        print(f"共加载 {len(data)} 条数据")
        
        for idx, item in enumerate(tqdm(data, desc="处理进度")):
            try:
                result = self.process_single_item(item)
                results.append(result)
                
                if result["is_correct"]:
                    correct_count += 1
                    
                # 每处理10条打印一次当前准确率
                if (idx + 1) % 10 == 0:
                    current_acc = correct_count / (idx + 1)
                    print(f"\n当前进度: {idx + 1}/{len(data)}, 准确率: {current_acc:.2%}")
                    
            except Exception as e:
                print(f"处理第 {idx} 条数据时出错: {e}")
                results.append({
                    "question": item.get("Question", "unknown"),
                    "error": str(e),
                    "is_correct": False
                })

        # 计算最终准确率
        total = len([r for r in results if "error" not in r])
        accuracy = correct_count / total if total > 0 else 0
        
        summary = {
            "total_processed": len(data),
            "successful": total,
            "correct_count": correct_count,
            "accuracy": accuracy,
            "accuracy_percentage": f"{accuracy:.2%}"
        }
        
        return results, summary


def main():
    # 配置路径
    json_path = "1.json"
    model_path = "/root/autodl-tmp/cache/qwen/Qwen2.5-VL-7B-Instruct"
    output_path = "validation_results.json"
    
    # 检查文件是否存在
    if not os.path.exists(json_path):
        print(f"错误: 找不到文件 {json_path}")
        print("请确保1.json存在于当前目录")
        return
    
    if not os.path.exists(model_path):
        print(f"错误: 找不到模型 {model_path}")
        print("请检查模型路径是否正确")
        return
    
    # 初始化验证器
    validator = QwenSelfValidator(model_path)
    
    # 处理数据
    results, summary = validator.process_dataset(json_path)
    
    # 保存详细结果
    output_data = {
        "summary": summary,
        "detailed_results": results
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print("\n" + "="*50)
    print("验证完成！")
    print("="*50)
    print(f"总处理数: {summary['total_processed']}")
    print(f"成功处理: {summary['successful']}")
    print(f"正确数量: {summary['correct_count']}")
    print(f"准确率: {summary['accuracy_percentage']}")
    print(f"详细结果已保存至: {output_path}")
    print("="*50)

if __name__ == "__main__":
    main()