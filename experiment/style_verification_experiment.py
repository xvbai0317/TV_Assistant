import json
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import os
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
from dataclasses import dataclass
from enum import Enum

class StyleType(Enum):
    """预定义风格类型"""
    ACADEMIC = "academic"           # 学术严谨风格
    CONCISE = "concise"             # 简洁明了风格
    CREATIVE = "creative"           # 创意生动风格
    PROFESSIONAL = "professional"   # 商务专业风格
    FRIENDLY = "friendly"           # 友好亲切风格
    TECHNICAL = "technical"         # 技术详细风格
    CUSTOM = "custom"               # 自定义风格


@dataclass
class StyleConfig:
    """风格配置"""
    style_type: StyleType
    name: str
    description: str
    criteria: List[str]  # 评判标准
    examples: Optional[Dict[str, str]] = None  # 正负例
    
    def get_prompt(self) -> str:
        """生成风格验证的system prompt"""
        criteria_text = "\n".join([f"{i+1}. {c}" for i, c in enumerate(self.criteria)])
        
        prompt = f"""你需要评估以下回答是否符合【{self.name}】风格。

风格定义：{self.description}

评判标准：
{criteria_text}

请按以下格式输出评估结果：
- 符合度评分：0-10分（10分为完全符合）
- 评估理由：简要说明评分依据
- 改进建议：如何更好地符合该风格（如不符合）

最终判断：如果评分≥7分，回答"风格符合"；否则回答"风格不符合"。"""

        return prompt


# 预定义风格库
STYLE_LIBRARY = {
    StyleType.ACADEMIC: StyleConfig(
        style_type=StyleType.ACADEMIC,
        name="学术严谨风格",
        description="使用专业术语，逻辑严密，引用规范，客观中立，结构完整",
        criteria=[
            "使用准确的专业术语和学术表达",
            "论证逻辑严密，条理清晰",
            "观点客观中立，避免主观臆断",
            "包含必要的背景信息和上下文",
            "结构完整（引言-论证-结论）"
        ],
        examples={
            "good": "基于上述实验数据，我们可以观察到变量X与Y呈现显著正相关关系（r=0.85, p<0.01）。这一发现与Smith等人(2020)的研究结论相一致...",
            "bad": "我觉得X和Y应该有关系，看起来挺明显的，大家都这么觉得。"
        }
    ),
    StyleType.CONCISE: StyleConfig(
        style_type=StyleType.CONCISE,
        name="简洁明了风格",
        description="言简意赅，直击要点，避免冗余，信息密度高",
        criteria=[
            "直接回答核心问题，不绕弯子",
            "删除不必要的修饰词和重复内容",
            "每句话都承载有效信息",
            "总字数控制在必要范围内",
            "避免'众所周知'、'显而易见'等空话"
        ],
        examples={
            "good": "结论：模型准确率达到94.2%，满足生产环境要求。",
            "bad": "关于这个模型呢，我们要从很多角度来看，首先呢，众所周知，机器学习模型是很重要的，那么我们的这个模型经过了一系列的测试和验证，最终呢，我们发现它的准确率，也就是accuracy，达到了一个相当不错的水平，具体数字是94.2%，这个成绩呢，从各方面来看都是令人满意的，基本上是可以用的。"
        }
    ),
    StyleType.CREATIVE: StyleConfig(
        style_type=StyleType.CREATIVE,
        name="创意生动风格",
        description="比喻形象，语言生动，角度新颖，富有感染力",
        criteria=[
            "使用恰当的比喻和类比",
            "语言生动形象，避免枯燥",
            "提供新颖独特的视角",
            "能够引发读者兴趣和共鸣",
            "在准确性的基础上增加趣味性"
        ]
    ),
    StyleType.PROFESSIONAL: StyleConfig(
        style_type=StyleType.PROFESSIONAL,
        name="商务专业风格",
        description="正式得体，数据支撑，行动导向，结果明确",
        criteria=[
            "使用商务场合的正式用语",
            "关键论点有数据或事实支撑",
            "提供明确的行动建议或结论",
            "考虑商业影响和可行性",
            "语气自信但不夸大"
        ]
    ),
    StyleType.TECHNICAL: StyleConfig(
        style_type=StyleType.TECHNICAL,
        name="技术详细风格",
        description="步骤清晰，细节完整，代码/参数准确，可复现性强",
        criteria=[
            "技术细节描述准确完整",
            "包含必要的参数、版本信息",
            "步骤清晰，可复现性强",
            "代码示例规范（如适用）",
            "错误处理和边界情况说明"
        ]
    ),
    StyleType.FRIENDLY: StyleConfig(
        style_type=StyleType.FRIENDLY,
        name="友好亲切风格",
        description="语气温暖，易于理解，耐心解释，鼓励互动",
        criteria=[
            "使用亲切自然的称呼和语气",
            "复杂概念用简单语言解释",
            "展现耐心和鼓励的态度",
            "适当使用表情符号（如适用）",
            "邀请用户进一步提问"
        ]
    )
}


class ValidationResult:
    """验证结果数据类"""
    def __init__(self):
        self.question: str = ""
        self.generated_answer: str = ""
        self.ground_truth: Optional[str] = None
        
        # 内容验证
        self.content_correct: bool = False
        self.content_validation_detail: str = ""
        
        # 风格验证
        self.style_score: float = 0.0  # 0-10
        self.style_compliant: bool = False
        self.style_validation_detail: str = ""
        self.style_suggestions: str = ""
        
        # 综合评分
        self.overall_score: float = 0.0  # 综合得分
        
    def to_dict(self) -> Dict:
        return {
            "question": self.question,
            "generated_answer": self.generated_answer,
            "ground_truth": self.ground_truth,
            "content_validation": {
                "is_correct": self.content_correct,
                "detail": self.content_validation_detail
            },
            "style_validation": {
                "style_score": self.style_score,
                "is_compliant": self.style_compliant,
                "detail": self.style_validation_detail,
                "suggestions": self.style_suggestions
            },
            "overall_score": self.overall_score,
            "final_pass": self.content_correct and self.style_compliant
        }


class QwenStyleValidator:
    def __init__(self, 
                 model_path: str = "/root/autodl-tmp/TV_Assistant/train/LlamaFactory/saves/TV_sft_merged",
                 style_config: Optional[StyleConfig] = None):
        """
        初始化模型和风格配置
        """
        self.style_config = style_config or STYLE_LIBRARY[StyleType.ACADEMIC]
        print(f"正在加载模型: {model_path}")
        print(f"当前风格设定: {self.style_config.name}")
        
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

    def generate_answer(self, question: str, image_path: str = None, 
                       style_hint: str = None, max_new_tokens: int = 512) -> str:
        """
        生成答案，可选加入风格提示
        """
        # 构建system message加入风格要求
        system_msg = f"你是一个有帮助的AI助手。回答时请采用【{self.style_config.name}】。"
        if style_hint:
            system_msg += f"\n具体要求：{style_hint}"
        
        # 构建消息
        content = []
        if image_path and os.path.exists(image_path):
            content.append({"type": "image", "image": image_path})
        content.append({"type": "text", "text": question})
        
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": content}
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

        # 生成
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        answer = self.processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0].strip()
        
        return answer

    def validate_content(self, question: str, answer: str, ground_truth: str = None) -> Tuple[bool, str]:
        """
        内容正确性验证
        """
        prompt = f"""请判断以下答案的内容是否正确。如果内容正确请回答"正确"，如果错误请回答"错误"并简要说明原因。

问题：{question}

答案：{answer}"""

        if ground_truth:
            prompt += f"\n\n参考答案：{ground_truth}"

        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        
        result = self._generate(messages, max_new_tokens=256)
        
        is_correct = "正确" in result and "错误" not in result[:10]
        return is_correct, result

    def validate_style(self, answer: str) -> Tuple[float, bool, str, str]:
        """
        风格验证：返回 (分数, 是否合规, 详细评估, 改进建议)
        """
        style_prompt = self.style_config.get_prompt()
        
        full_prompt = f"{style_prompt}\n\n需要评估的回答：\n{answer}\n\n请给出评估结果："
        
        messages = [{"role": "user", "content": [{"type": "text", "text": full_prompt}]}]
        
        result = self._generate(messages, max_new_tokens=512)
        
        # 解析结果
        score = self._extract_score(result)
        is_compliant = "风格符合" in result or score >= 7.0
        
        # 提取建议部分
        suggestions = ""
        if "改进建议：" in result:
            suggestions = result.split("改进建议：")[-1].split("\n")[0]
        
        return score, is_compliant, result, suggestions

    def _generate(self, messages: List[Dict], max_new_tokens: int = 256) -> str:
        """内部生成方法"""
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

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        return self.processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True
        )[0].strip()

    def _extract_score(self, text: str) -> float:
        """从文本中提取评分"""
        import re
        # 匹配 "8分"、"评分：8"、"8/10" 等格式
        patterns = [
            r'(\d+(\.\d+)?)\s*分',
            r'评分[：:]\s*(\d+(\.\d+)?)',
            r'(\d+(\.\d+)?)\s*/\s*10',
            r'符合度[：:]\s*(\d+(\.\d+)?)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                score = float(match.group(1))
                return min(max(score, 0), 10)  # 限制在0-10
        return 5.0  # 默认中等分数

    def process_item(self, item: Dict, enforce_style: bool = True) -> ValidationResult:
        """
        处理单个问题：生成 -> 内容验证 -> 风格验证
        """
        result = ValidationResult()
        result.question = item.get("Question", item.get("question", ""))
        result.ground_truth = item.get("Answer", item.get("answer", item.get("ground_truth", None)))
        image_path = item.get("image", item.get("image_path", None))
        
        print(f"\n问题: {result.question[:60]}...")
        
        # 1. 生成答案（如enforce_style=True，会在生成时加入风格提示）
        style_hint = self.style_config.description if enforce_style else None
        result.generated_answer = self.generate_answer(
            result.question, 
            image_path, 
            style_hint=style_hint
        )
        print(f"生成答案: {result.generated_answer[:80]}...")
        
        # 2. 内容验证
        result.content_correct, result.content_validation_detail = self.validate_content(
            result.question, 
            result.generated_answer, 
            result.ground_truth
        )
        print(f"内容验证: {'✓ 正确' if result.content_correct else '✗ 错误'}")
        
        # 3. 风格验证
        result.style_score, result.style_compliant, detail, result.style_suggestions = \
            self.validate_style(result.generated_answer)
        result.style_validation_detail = detail
        print(f"风格评分: {result.style_score}/10 - {'✓ 符合' if result.style_compliant else '✗ 不符合'}")
        
        # 4. 计算综合得分（内容正确性权重0.6，风格权重0.4）
        content_score = 10.0 if result.content_correct else 0.0
        result.overall_score = content_score * 0.6 + result.style_score * 0.4
        
        return result

    def process_dataset(self, json_path: str, enforce_style: bool = True) -> Tuple[List[ValidationResult], Dict]:
        """
        处理整个数据集
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, dict):
            data = [data]
        
        results = []
        stats = {
            "total": len(data),
            "content_correct": 0,
            "style_compliant": 0,
            "both_pass": 0,
            "avg_style_score": 0.0,
            "avg_overall_score": 0.0
        }
        
        print(f"\n开始处理 {len(data)} 条数据...")
        print(f"风格设定: {self.style_config.name}")
        print("="*60)
        
        for idx, item in enumerate(tqdm(data, desc="验证进度")):
            try:
                result = self.process_item(item, enforce_style)
                results.append(result)
                
                # 更新统计
                if result.content_correct:
                    stats["content_correct"] += 1
                if result.style_compliant:
                    stats["style_compliant"] += 1
                if result.content_correct and result.style_compliant:
                    stats["both_pass"] += 1
                stats["avg_style_score"] += result.style_score
                stats["avg_overall_score"] += result.overall_score
                
                # 进度打印
                if (idx + 1) % 5 == 0:
                    current_content_acc = stats["content_correct"] / (idx + 1)
                    current_style_acc = stats["style_compliant"] / (idx + 1)
                    print(f"\n进度: {idx+1}/{len(data)} | 内容正确率: {current_content_acc:.1%} | 风格合规率: {current_style_acc:.1%}")
                    
            except Exception as e:
                print(f"\n处理第 {idx} 条时出错: {e}")
                # 创建错误结果
                err_result = ValidationResult()
                err_result.question = str(item.get("Question", "unknown"))
                err_result.content_validation_detail = f"Error: {str(e)}"
                results.append(err_result)
        
        # 计算最终统计
        valid_count = len([r for r in results if r.overall_score > 0])
        if valid_count > 0:
            stats["avg_style_score"] /= valid_count
            stats["avg_overall_score"] /= valid_count
        
        stats["content_accuracy"] = stats["content_correct"] / stats["total"]
        stats["style_compliance_rate"] = stats["style_compliant"] / stats["total"]
        stats["combined_pass_rate"] = stats["both_pass"] / stats["total"]
        
        return results, stats


def print_style_menu():
    """打印风格选择菜单"""
    print("\n" + "="*60)
    print("可选风格类型：")
    print("="*60)
    for i, (style_type, config) in enumerate(STYLE_LIBRARY.items(), 1):
        print(f"{i}. {config.name}")
        print(f"   描述: {config.description}")
        print(f"   标准: {', '.join(config.criteria[:2])}...")
        print()
    print("6. 自定义风格 (Custom)")
    print("="*60)


def create_custom_style() -> StyleConfig:
    """创建自定义风格"""
    print("\n创建自定义风格：")
    name = input("风格名称: ")
    description = input("风格描述: ")
    print("输入评判标准（每行一个，输入空行结束）：")
    criteria = []
    while True:
        c = input("> ")
        if not c:
            break
        criteria.append(c)
    
    return StyleConfig(
        style_type=StyleType.CUSTOM,
        name=name,
        description=description,
        criteria=criteria
    )


def main():
    # 配置
    json_path = "/root/autodl-tmp/TV_Assistant/experiment/test/test2/test_data.json"
    model_path = "/root/autodl-tmp/TV_Assistant/train/LlamaFactory/saves/TV_sft_merged"
    
    # 检查文件
    if not os.path.exists(json_path):
        print(f"错误: 找不到 {json_path}")
        return
    if not os.path.exists(model_path):
        print(f"错误: 找不到模型 {model_path}")
        return
    
    # 选择风格
    print_style_menu()
    # choice = input("请选择风格类型 (1-6): ").strip()
    choice = "6"
    
    style_map = {
        "1": StyleType.ACADEMIC,
        "2": StyleType.CONCISE,
        "3": StyleType.CREATIVE,
        "4": StyleType.PROFESSIONAL,
        "5": StyleType.TECHNICAL,
        "6": StyleType.FRIENDLY
    }
    
    if choice == "7" or choice not in style_map:
        style_config = create_custom_style()
    else:
        style_config = STYLE_LIBRARY[style_map[choice]]
    
    # 是否强制生成时遵循风格
    # enforce = input("\n是否强制模型在生成时遵循该风格? (y/n, 默认y): ").strip().lower() != "n"
    enforce = True
    
    # 初始化验证器
    validator = QwenStyleValidator(model_path, style_config)
    
    # 处理数据
    results, stats = validator.process_dataset(json_path, enforce_style=enforce)
    
    # 保存结果
    output = {
        "config": {
            "style_name": style_config.name,
            "style_description": style_config.description,
            "enforce_style_in_generation": enforce
        },
        "statistics": stats,
        "results": [r.to_dict() for r in results]
    }
    
    output_path = f"validation_results_{style_config.style_type.value}.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    # 打印报告
    print("\n" + "="*60)
    print("验证完成！最终报告")
    print("="*60)
    print(f"风格设定: {style_config.name}")
    print(f"强制风格生成: {'是' if enforce else '否'}")
    print("-"*60)
    print(f"总样本数: {stats['total']}")
    print(f"内容正确率: {stats['content_accuracy']:.2%} ({stats['content_correct']}/{stats['total']})")
    print(f"风格合规率: {stats['style_compliance_rate']:.2%} ({stats['style_compliant']}/{stats['total']})")
    print(f"双重达标率: {stats['combined_pass_rate']:.2%} ({stats['both_pass']}/{stats['total']})")
    print(f"平均风格评分: {stats['avg_style_score']:.2f}/10")
    print(f"平均综合得分: {stats['avg_overall_score']:.2f}/10")
    print("="*60)
    print(f"详细结果已保存: {output_path}")
    
    # 打印失败案例摘要
    failures = [r for r in results if not (r.content_correct and r.style_compliant)]
    if failures:
        print(f"\n未达标案例 ({len(failures)}个):")
        for i, f in enumerate(failures[:3], 1):
            print(f"{i}. 内容: {'✓' if f.content_correct else '✗'} | 风格: {f.style_score}分")
            print(f"   问题: {f.question[:50]}...")
            if f.style_suggestions:
                print(f"   建议: {f.style_suggestions[:60]}...")


if __name__ == "__main__":
    main()