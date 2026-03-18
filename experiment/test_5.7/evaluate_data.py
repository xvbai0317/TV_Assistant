import json
import re
import random
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import os

@dataclass
class EvaluationResult:
    """评估结果数据类"""
    content_accuracy: float = 0.0  # 内容事实准确度 (0-10)
    interaction_fit: float = 0.0    # 交互画像拟合度 (0-10)
    rule_compliance: float = 0.0    # 显式规则遵循度 (0-10)
    diversity_safety: float = 0.0   # 多样性与安全 (0-10)
    overall_score: float = 0.0      # 综合平均分 (0-10)

    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            "content_accuracy": round(self.content_accuracy, 2),
            "interaction_fit": round(self.interaction_fit, 2),
            "rule_compliance": round(self.rule_compliance, 2),
            "diversity_safety": round(self.diversity_safety, 2),
            "overall_score": round(self.overall_score, 2)
        }

class DataEvaluator:
    def __init__(self, 
                 model_path: str = "/root/autodl-tmp/cache/qwen/Qwen2.5-VL-7B-Instruct",
                 max_new_tokens: int = 200):
        """
        初始化评估器
        """
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        
        # 加载模型用于内容事实评估
        print(f"正在加载评估模型: {model_path}")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        self.model.eval()
        
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        print("评估模型加载完成！")
        
        # 定义评估规则
        self.opening_rules = [
            r'^亲爱的观众朋友',
            r'^各位看官',
            r'^老铁们',
            r'^家人们',
            r'^各位朋友晚上好',
            r'^欢迎来到本期深度解读',
            r'^各位学霸'
        ]
        
        self.closing_rules = [
            r'我们下期再见$',
            r'记得点赞关注哦$',
            r'这就是今天的全部内容$',
            r'以上观点仅供参考$',
            r'感谢各位的耐心观看$',
            r'在评论区留下你的看法$',
            r'咱们不见不散$'
        ]
        
    def evaluate_content_accuracy(self, question: str, answer: str) -> float:
        """
        评估内容事实准确度 (0-10)
        使用模型判断回答的事实准确性
        """
        prompt = f"请判断以下回答的内容是否准确。如果内容准确请回答'分数：X'，其中X是0-10的分数，10分为完全准确，0分为完全错误。问题：{question} 答案：{answer}"
        
        # 调用模型进行评估
        try:
            messages = [
                {"role": "system", "content": "你是一个专业的内容评估专家，擅长判断回答的事实准确性。"},
                {"role": "user", "content": [{"type": "text", "text": prompt}]}
            ]
            
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
            
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=0.1,
                    top_p=0.9,
                    do_sample=False,
                    pad_token_id=self.processor.tokenizer.eos_token_id
                )
            
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            result = self.processor.batch_decode(
                generated_ids_trimmed, 
                skip_special_tokens=True
            )[0].strip()
            
            # 提取分数
            match = re.search(r'分数：(\d+(?:\.\d+)?)', result)
            if match:
                score = float(match.group(1))
                return max(0.0, min(10.0, score))
            else:
                return 5.0  # 默认中等分数
                
        except Exception as e:
            print(f"内容评估失败: {e}")
            return 5.0  # 异常情况下返回中等分数
    
    def evaluate_interaction_fit(self, answer: str, style_profile: Dict = None) -> float:
        """
        评估交互画像拟合度 (0-10)
        基于风格配置评估回答是否符合预期风格
        """
        if not style_profile:
            return 5.0  # 没有风格配置时返回中等分数
        
        try:
            # 构建风格描述
            style_desc = []
            if style_profile["info_depth"] > 0.7:
                style_desc.append("高信息密度，专业术语丰富")
            elif style_profile["info_depth"] > 0.4:
                style_desc.append("中等信息密度，兼顾专业与可读")
            else:
                style_desc.append("低信息密度，简洁明了")
                
            if style_profile["emotional_resonance"] > 0.7:
                style_desc.append("高情感共鸣，故事化叙述")
            elif style_profile["emotional_resonance"] > 0.4:
                style_desc.append("中等情感共鸣，适当亲和力")
            else:
                style_desc.append("低情感共鸣，客观理性")
                
            style_description = "；".join(style_desc)
            
            prompt = f"请评估以下回答是否符合指定风格。如果符合请回答'分数：X'，其中X是0-10的分数，10分为完全符合，0分为完全不符合。风格要求：{style_description} 回答：{answer}"
            # 调用模型进行评估
            messages = [
                {"role": "system", "content": "你是一个专业的风格评估专家，擅长判断回答是否符合指定风格。"},
                {"role": "user", "content": [{"type": "text", "text": prompt}]}
            ]
            
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
            
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=0.1,
                    top_p=0.9,
                    do_sample=False,
                    pad_token_id=self.processor.tokenizer.eos_token_id
                )
            
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            result = self.processor.batch_decode(
                generated_ids_trimmed, 
                skip_special_tokens=True
            )[0].strip()
            
            # 提取分数
            match = re.search(r'分数：(\d+(?:\.\d+)?)', result)
            if match:
                score = float(match.group(1))
                return max(0.0, min(10.0, score))
            else:
                return 5.0  # 默认中等分数
                
        except Exception as e:
            print(f"风格评估失败: {e}")
            return 5.0  # 异常情况下返回中等分数
    
    def evaluate_rule_compliance(self, answer: str, constraint: str = None) -> float:
        """
        评估显式规则遵循度 (0-10)
        检查回答是否遵循了特定的规则要求
        """
        if not constraint:
            return 10.0  # 没有约束时默认满分
        
        score = 10.0
        
        # 检查开头规则
        opening_match = any(re.match(rule, answer) for rule in self.opening_rules)
        # 检查结尾规则
        closing_match = any(re.search(rule, answer) for rule in self.closing_rules)
        
        # 根据约束类型扣分
        if "开头" in constraint and not opening_match:
            score -= 5.0
        if "结尾" in constraint and not closing_match:
            score -= 5.0
            
        return max(0.0, min(10.0, score))
    
    def evaluate_diversity_safety(self, answers: List[str], current_answer: str, index: int) -> float:
        """
        评估多样性与安全 (0-10)
        - 多样性：避免重复内容
        - 安全性：避免有害内容
        """
        score = 10.0
        
        # 检查多样性
        if index > 0:
            # 与前面10个回答比较相似度
            for prev_answer in answers[:index][-10:]:
                # 简单的相似度检查：计算相同词的比例
                current_words = set(current_answer.split())
                prev_words = set(prev_answer.split())
                if current_words and prev_words:
                    similarity = len(current_words.intersection(prev_words)) / len(current_words.union(prev_words))
                    if similarity > 0.8:  # 如果相似度超过80%
                        score -= 2.0
        
        # 检查安全性（简单的关键词匹配）
        harmful_keywords = [
            "色情", "暴力", "恐怖", "极端", "歧视", "仇恨", 
            "自杀", "犯罪", "毒品", "赌博", "诈骗"
        ]
        
        for keyword in harmful_keywords:
            if keyword in current_answer:
                score -= 5.0
                break
        
        return max(0.0, min(10.0, score))
    
    def evaluate_single_item(self, item: Dict, answers: List[str], index: int) -> EvaluationResult:
        """
        评估单个问答对
        """
        question = item.get("Q", "")
        answer = item.get("A", "")
        metadata = item.get("metadata", {})
        
        # 评估各个维度
        content_accuracy = self.evaluate_content_accuracy(question, answer)
        interaction_fit = self.evaluate_interaction_fit(answer, metadata.get("style_profile"))
        rule_compliance = self.evaluate_rule_compliance(answer, metadata.get("constraint"))
        diversity_safety = self.evaluate_diversity_safety(answers, answer, index)
        
        # 计算综合分数（等权重）
        overall_score = (content_accuracy + interaction_fit + rule_compliance + diversity_safety) / 4
        
        return EvaluationResult(
            content_accuracy=content_accuracy,
            interaction_fit=interaction_fit,
            rule_compliance=rule_compliance,
            diversity_safety=diversity_safety,
            overall_score=overall_score
        )
    
    def evaluate_dataset(self, json_path: str, sample_size: int = None) -> Tuple[List[EvaluationResult], Dict]:
        """
        评估整个数据集
        """
        # 读取数据集
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 如果提供了样本大小，则随机采样
        if sample_size and sample_size < len(data):
            data = random.sample(data, sample_size)
            print(f"随机采样 {sample_size} 条数据进行评估...")
        else:
            print(f"评估全部 {len(data)} 条数据...")
        
        results = []
        answers = []
        
        # 评估每条数据
        for i, item in enumerate(data):
            print(f"正在评估第 {i+1}/{len(data)} 条数据...")
            answer = item.get("A", "")
            results.append(self.evaluate_single_item(item, answers, i))
            answers.append(answer)
        
        # 计算整体统计
        stats = {
            "total_items": len(results),
            "avg_content_accuracy": np.mean([r.content_accuracy for r in results]),
            "avg_interaction_fit": np.mean([r.interaction_fit for r in results]),
            "avg_rule_compliance": np.mean([r.rule_compliance for r in results]),
            "avg_diversity_safety": np.mean([r.diversity_safety for r in results]),
            "avg_overall_score": np.mean([r.overall_score for r in results]),
            "std_overall_score": np.std([r.overall_score for r in results])
        }
        
        return results, stats
    
    def save_results(self, results: List[EvaluationResult], stats: Dict, output_path: str):
        """
        保存评估结果
        """
        output_data = {
            "statistics": {
                "total_items": stats["total_items"],
                "content_accuracy": {
                    "mean": round(stats["avg_content_accuracy"], 2),
                    "description": "内容事实准确度 (0-10)"
                },
                "interaction_fit": {
                    "mean": round(stats["avg_interaction_fit"], 2),
                    "description": "交互画像拟合度 (0-10)"
                },
                "rule_compliance": {
                    "mean": round(stats["avg_rule_compliance"], 2),
                    "description": "显式规则遵循度 (0-10)"
                },
                "diversity_safety": {
                    "mean": round(stats["avg_diversity_safety"], 2),
                    "description": "多样性与安全 (0-10)"
                },
                "overall_score": {
                    "mean": round(stats["avg_overall_score"], 2),
                    "std": round(stats["std_overall_score"], 2),
                    "description": "综合平均分 (0-10)"
                }
            },
            "individual_results": [r.to_dict() for r in results]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"评估结果已保存到: {output_path}")

def main():
    # 配置参数
    DATASET_PATH = "/root/autodl-tmp/TV_Assistant/experiment/test_5.7/ood_test_set_400.json"
    DETAILED_DATASET_PATH = "/root/autodl-tmp/TV_Assistant/experiment/test_5.7/ood_test_set_400_detailed.json"
    OUTPUT_PATH = "/root/autodl-tmp/TV_Assistant/experiment/test_5.7/evaluation_results.json"
    MODEL_PATH = "/root/autodl-tmp/cache/qwen/Qwen2.5-VL-7B-Instruct"
    SAMPLE_SIZE = None  # 可以设置为较小的数字进行快速测试，如10
    
    # 选择使用详细数据集（包含metadata）
    if os.path.exists(DETAILED_DATASET_PATH):
        dataset_path = DETAILED_DATASET_PATH
    else:
        dataset_path = DATASET_PATH
        print("警告：使用标准数据集，缺少metadata信息，部分评估可能不准确")
    
    # 初始化评估器
    evaluator = DataEvaluator(model_path=MODEL_PATH)
    
    # 评估数据集
    results, stats = evaluator.evaluate_dataset(dataset_path, SAMPLE_SIZE)
    
    # 保存结果
    evaluator.save_results(results, stats, OUTPUT_PATH)
    
    # 打印统计信息
    print("\n=== 评估结果统计 ===")
    print(f"总评估条数: {stats['total_items']}")
    print(f"内容事实准确度: {stats['avg_content_accuracy']:.2f}/10")
    print(f"交互画像拟合度: {stats['avg_interaction_fit']:.2f}/10")
    print(f"显式规则遵循度: {stats['avg_rule_compliance']:.2f}/10")
    print(f"多样性与安全: {stats['avg_diversity_safety']:.2f}/10")
    print(f"综合平均分: {stats['avg_overall_score']:.2f}/10")
    print(f"综合分标准差: {stats['std_overall_score']:.2f}")
    print("==================")

if __name__ == "__main__":
    main()
