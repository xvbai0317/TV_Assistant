import json
import random
from typing import Dict, List, Tuple
from dataclasses import dataclass
import numpy as np
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoTokenizer
from qwen_vl_utils import process_vision_info
import torch

@dataclass
class StyleProfile:
    """风格配置文件：四个维度的百分比权重"""
    info_depth: float      # 信息深度 (0-1)
    emotional_resonance: float  # 情感共鸣 (0-1)
    presentation_structure: float  # 呈现结构 (0-1)
    language_fun: float    # 语言趣味性 (0-1)
    
    def to_prompt_description(self) -> str:
        """将风格配置转换为prompt描述"""
        parts = []
        
        # 信息深度
        if self.info_depth > 0.7:
            parts.append("回答需具备极高的信息密度，包含专业术语、技术细节、学术引用和深层原理剖析")
        elif self.info_depth > 0.4:
            parts.append("回答应提供充分的技术细节和背景知识，兼顾专业性与可读性")
        else:
            parts.append("回答保持简洁明了，聚焦核心概念，避免过度技术化")
            
        # 情感共鸣
        if self.emotional_resonance > 0.7:
            parts.append("融入强烈的人文关怀和情感温度，使用故事化叙述，引发用户情感共鸣")
        elif self.emotional_resonance > 0.4:
            parts.append("适当加入生活化比喻和温和语气，让内容更有亲和力")
        else:
            parts.append("保持客观理性的陈述风格，减少情感色彩，注重事实准确性")
            
        # 呈现结构
        if self.presentation_structure > 0.7:
            parts.append("采用严谨的学术结构：背景引入→核心论证→多维分析→总结升华，层次分明")
        elif self.presentation_structure > 0.4:
            parts.append("使用清晰的段落结构，配合要点列举和逻辑递进，便于理解记忆")
        else:
            parts.append("采用自由流畅的叙述方式，如朋友对话般自然展开，不拘泥于固定格式")
            
        # 语言趣味性
        if self.language_fun > 0.7:
            parts.append("大量使用网络热梗、幽默比喻、俏皮话和反转句式，让科普像脱口秀一样有趣")
        elif self.language_fun > 0.4:
            parts.append("适当加入轻松比喻和流行文化引用，让严肃知识变得生动活泼")
        else:
            parts.append("使用规范的书面语表达，保持专业严谨的措辞风格")
            
        return "；".join(parts)

# 话题池 - 涵盖科技、人文、生活、娱乐等多领域
TOPIC_POOL = [
    # 前沿科技
    "量子计算", "脑机接口", "可控核聚变", "基因编辑CRISPR", "元宇宙技术", 
    "自动驾驶伦理", "AI大模型原理", "6G通信技术", "室温超导", "暗物质探测",
    # 天文宇宙
    "黑洞信息悖论", "引力波探测", "系外行星宜居性", "宇宙加速膨胀", "多重宇宙理论",
    "火星殖民挑战", "小行星防御系统", "太阳风暴影响", "银河系结构", "暗能量本质",
    # 生物医学
    "mRNA疫苗机制", "端粒与衰老", "肠道菌群影响", "神经可塑性", "睡眠科学",
    "疼痛感知机制", "免疫系统记忆", "病毒进化策略", "癌症免疫疗法", "脑肠轴理论",
    # 人文社科
    "认知偏差类型", "群体决策心理", "语言影响思维", "历史蝴蝶效应", "文化模因传播",
    "注意力经济", "算法推荐茧房", "消费主义心理", "都市孤独症", "信息过载应对",
    # 生活百科
    "咖啡烘焙化学", "红酒陈年原理", "运动代谢机制", "音乐治疗原理", "色彩心理学",
    "香料分子作用", "发酵食品科学", "宠物行为语言", "植物感知能力", "梦境生成机制",
    # 财经商业
    "区块链共识机制", "量化交易原理", "平台经济模式", "行为金融学", "数字货币监管",
    "ESG投资逻辑", "供应链韧性", "技术奇点经济", "注意力货币化", "零工经济权益"
]

# 约束模板池
CONSTRAINT_TEMPLATES = {
    "opening": [
        "开头必须用'亲爱的观众朋友'称呼",
        "开头必须用'各位看官'称呼",
        "开头必须用'老铁们'称呼",
        "开头必须用'家人们'称呼",
        "开头必须用'各位朋友晚上好'作为开场",
        "开头必须说'欢迎来到本期深度解读'",
        "开头必须用'各位学霸'称呼"
    ],
    "closing": [
        "结尾必须说'我们下期再见'",
        "结尾必须说'记得点赞关注哦'",
        "结尾必须说'这就是今天的全部内容'",
        "结尾必须用'以上观点仅供参考'结束",
        "结尾必须说'感谢各位的耐心观看'",
        "结尾必须邀请观众'在评论区留下你的看法'",
        "结尾必须说'咱们不见不散'"
    ],
    "both": [
        "开头用'亲爱的观众朋友'且结尾说'我们下期再见'",
        "开头用'老铁们'且结尾说'记得点赞关注哦'",
        "开头用'家人们'且结尾说'感谢各位的耐心观看'",
        "开头说'欢迎来到本期深度解读'且结尾邀请评论"
    ]
}

def generate_random_style() -> StyleProfile:
    """生成随机风格配置"""
    return StyleProfile(
        info_depth=random.uniform(0, 1),
        emotional_resonance=random.uniform(0, 1),
        presentation_structure=random.uniform(0, 1),
        language_fun=random.uniform(0, 1)
    )

def get_constraint() -> str:
    """30%概率返回约束条件"""
    if random.random() < 0.3:
        constraint_type = random.choice(["opening", "closing", "both"])
        return random.choice(CONSTRAINT_TEMPLATES[constraint_type])
    return ""

# 全局变量：缓存加载的模型和processor
MODEL_CACHE = {}


def load_model(model_path: str):
    """
    加载本地Qwen2.5-VL模型和processor，使用缓存避免重复加载
    """
    if model_path not in MODEL_CACHE:
        print(f"正在加载模型: {model_path}...")
        
        # 加载模型（使用Qwen2.5-VL专用的模型类）
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        model.eval()
        
        # 加载处理器
        processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        MODEL_CACHE[model_path] = (model, processor)
        print(f"模型加载完成: {model_path}")
    return MODEL_CACHE[model_path]


def call_qwen_api(prompt: str, model_path: str = "/root/autodl-tmp/TV_Assistant/train/LlamaFactory/saves/TV_sft_merged") -> str:
    """
    直接调用本地部署的Qwen2.5-VL模型
    """
    try:
        # 加载模型和processor（使用缓存）
        model, processor = load_model(model_path)
        
        # 构建对话消息
        messages = [
            {"role": "system", "content": "你是一个专业的电视内容助手，擅长生成高质量的科普问答内容。"},
            {"role": "user", "content": [{"type": "text", "text": prompt}]}
        ]
        
        # 应用对话模板
        text = processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # 处理视觉信息（即使没有图片也需要调用）
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)
        
        # 生成文本
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=800,
                temperature=0.8,
                top_p=0.9,
                do_sample=True,
                pad_token_id=processor.tokenizer.eos_token_id
            )
        
        # 提取生成的文本（只保留新生成的部分）
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        answer = processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0].strip()
        
        return answer
        
    except Exception as e:
        print(f"模型调用失败: {e}")
        # 备用：返回模拟数据以便测试脚本
        return "[API_ERROR]"

def generate_question(topic: str) -> str:
    """生成问题prompt"""
    question_types = [
        f"请生成一个关于'{topic}'的深度科普问题，要求：",
        f"请生成一个关于'{topic}'的电视观众可能感兴趣的提问，要求：",
        f"请生成一个关于'{topic}'的引发思考的开场问题，要求："
    ]
    
    requirements = [
        "问题要有吸引力，适合电视口播场景",
        "问题应该具体而非空泛，避免过于学术化",
        "问题长度控制在30-50字之间",
        "直接返回问题文本，不要有任何解释或前缀"
    ]
    
    prompt = random.choice(question_types) + "；".join(requirements)
    return prompt

def generate_answer(question: str, style: StyleProfile, constraint: str) -> str:
    """生成答案prompt"""
    base_prompt = f"""请回答以下问题，并严格遵守指定的风格要求：

【问题】{question}

【风格要求】{style.to_prompt_description()}
"""
    if constraint:
        base_prompt += f"\n【强制约束】{constraint}"
    
    base_prompt += "\n\n请直接输出回答内容，不要包含'根据您的要求'等前缀。"
    return base_prompt

def generate_single_qa(model_path: str, index: int = None) -> Dict[str, str]:
    """
    生成单个QA对
    """
    # 随机选择话题
    topic = random.choice(TOPIC_POOL)
    if index is not None:
        print(f"正在生成第 {index+1} 条数据 - 话题: {topic}")
    else:
        print(f"正在生成数据 - 话题: {topic}")
    
    # 生成问题
    q_prompt = generate_question(topic)
    print("正在生成问题...")
    question = call_qwen_api(q_prompt, model_path)
    
    # 清理问题文本
    question = question.strip().strip('"').strip("'").strip()
    if question.startswith("Q：") or question.startswith("问："):
        question = question[2:].strip()
    print(f"生成的问题: {question[:60]}...")
    
    # 生成风格配置和约束
    style = generate_random_style()
    constraint = get_constraint()
    
    # 生成答案
    a_prompt = generate_answer(question, style, constraint)
    print("正在生成答案...")
    answer = call_qwen_api(a_prompt, model_path)
    
    # 清理答案文本
    answer = answer.strip()
    print(f"生成的答案: {answer[:80]}...")
    
    return {
        "Q": question,
        "A": answer,
        "metadata": {
            "topic": topic,
            "style_profile": {
                "info_depth": round(style.info_depth, 2),
                "emotional_resonance": round(style.emotional_resonance, 2),
                "presentation_structure": round(style.presentation_structure, 2),
                "language_fun": round(style.language_fun, 2)
            },
            "constraint": constraint if constraint else None
        }
    }

def generate_dataset(num_samples: int = 400, model_path: str = "/root/autodl-tmp/TV_Assistant/train/LlamaFactory/saves/TV_sft_merged", output_path: str = "/root/autodl-tmp/TV_Assistant/experiment/test_5.7/ood_test_set.json") -> List[Dict]:
    """
    生成完整的OOD测试集
    
    Args:
        num_samples: 生成样本数量（默认400）
        model_path: 模型路径
        output_path: 输出JSON文件路径
    """
    dataset = []
    
    print(f"开始生成 {num_samples} 条OOD测试数据...")
    print(f"使用模型: {model_path}")
    
    # 统计信息
    stats = {
        "with_constraint": 0,
        "style_distribution": {"high_depth": 0, "high_emotion": 0, "high_structure": 0, "high_fun": 0}
    }
    
    for i in range(num_samples):
        try:
            qa_item = generate_single_qa(model_path, index=i)
            
            # 更新统计
            if qa_item["metadata"]["constraint"]:
                stats["with_constraint"] += 1
            if qa_item["metadata"]["style_profile"]["info_depth"] > 0.7:
                stats["style_distribution"]["high_depth"] += 1
            if qa_item["metadata"]["style_profile"]["emotional_resonance"] > 0.7:
                stats["style_distribution"]["high_emotion"] += 1
            if qa_item["metadata"]["style_profile"]["presentation_structure"] > 0.7:
                stats["style_distribution"]["high_structure"] += 1
            if qa_item["metadata"]["style_profile"]["language_fun"] > 0.7:
                stats["style_distribution"]["high_fun"] += 1
            
            # 只保留Q和A在最终输出中，metadata用于分析
            output_item = {
                "Q": qa_item["Q"],
                "A": qa_item["A"]
            }
            dataset.append(output_item)
            
            if (i + 1) % 50 == 0:
                print(f"已生成 {i+1}/{num_samples} 条数据...")
                
        except Exception as e:
            print(f"生成第 {i+1} 条数据时出错: {e}")
            continue
    
    # 保存JSON文件
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    
    # 保存带metadata的详细版本用于分析
    detailed_path = output_path.replace('.json', '_detailed.json')
    with open(detailed_path, 'w', encoding='utf-8') as f:
        json.dump([{**item, "metadata": generate_single_qa(model_path)["metadata"]} for item in dataset], 
                  f, ensure_ascii=False, indent=2)
    
    print(f"\n生成完成！")
    print(f"标准输出文件: {output_path}")
    print(f"详细分析文件: {detailed_path}")
    print(f"\n统计信息:")
    print(f"- 带约束的样本: {stats['with_constraint']}/{num_samples} ({stats['with_constraint']/num_samples*100:.1f}%)")
    print(f"- 高信息深度(>0.7): {stats['style_distribution']['high_depth']}条")
    print(f"- 高情感共鸣(>0.7): {stats['style_distribution']['high_emotion']}条")
    print(f"- 高呈现结构(>0.7): {stats['style_distribution']['high_structure']}条")
    print(f"- 高语言趣味(>0.7): {stats['style_distribution']['high_fun']}条")
    
    return dataset

# 批量生成时的优化：使用异步或批量API调用
def generate_batch_qa(topics: List[str], model_path: str) -> List[Dict]:
    """批量生成（如果API支持）"""
    # 这里可以实现批量调用逻辑以提高效率
    pass

if __name__ == "__main__":
    # 配置参数
    MODEL_PATH = "/root/autodl-tmp/TV_Assistant/train/LlamaFactory/saves/TV_sft_merged"
    OUTPUT_FILE = "./ood_test_set_400.json"
    NUM_SAMPLES = 400
    
    # 生成数据集
    dataset = generate_dataset(
        num_samples=NUM_SAMPLES,
        model_path=MODEL_PATH,
        output_path=OUTPUT_FILE
    )
    
    # 打印前3个样例
    print("\n=== 生成样例预览 ===")
    for i, item in enumerate(dataset[:3], 1):
        print(f"\n--- 样例 {i} ---")
        print(f"Q: {item['Q'][:80]}...")
        print(f"A: {item['A'][:100]}...")