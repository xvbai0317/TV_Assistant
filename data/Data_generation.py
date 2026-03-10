import json
import time
import random
import requests
from tqdm import tqdm
from datetime import datetime
from typing import List, Dict, Optional

# ===================== 配置区 =====================
API_KEY = "" 
API_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
MODEL_NAME = "qwen-vl-plus"

# 生成配置
CATEGORIES = {
    "影视评论与深度推荐": 2000,
    "娱乐八卦与行业资讯": 2000,
    "科普百科": 2000,
    "历史人文": 2000,
    "情感陪伴": 2000,
    "幽默段子与脑筋急转弯": 2000,
    "生活方式建议": 2000
}
OUTPUT_DIR = "./qa_datasets"           # 输出目录
BATCH_SIZE = 10                        # 每批生成数量（避免单次请求过多）
RETRY_TIMES = 3                        # 失败重试次数
DELAY_RANGE = (1, 3)                   # 请求间隔（秒），避免请求过快

# ===================== 工具函数 =====================
def init_output_dir():
    """初始化输出目录"""
    import os
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    return OUTPUT_DIR

def generate_prompt(category: str) -> str:
    """生成针对特定类别的QA生成提示词"""
    prompt_templates = {
        "影视评论与深度推荐": """请生成一个关于电影/电视剧的高质量QA问答对，要求：
        1. 每次生成不同电影/电视剧的内容，涵盖不同类型、国家和年代
        2. 问题要具体，涵盖影评、推荐、剧情分析、演员表现等维度
        3. 回答要专业、详细，有深度，不少于50字
        4. 避免过于简单或通用的问题
        示例格式：
        Q: 为什么《肖申克的救赎》被评为影史最佳电影之一？
        A: 《肖申克的救赎》之所以被奉为经典，不仅因为其精湛的叙事手法和蒂姆·罗宾斯、摩根·弗里曼的出色表演，更在于它深刻探讨了希望、自由与人性的主题...""",
        
        "娱乐八卦与行业资讯": """请生成一个关于娱乐圈八卦或行业资讯的高质量QA问答对，要求：
        1. 问题要贴近当前娱乐热点或行业动态
        2. 回答要真实可信，有事实依据，不少于50字
        3. 避免编造虚假信息
        示例格式：
        Q: 2025年春节档电影票房冠军是哪部作品？
        A: 2025年春节档票房冠军是《XX》，该片凭借精良的制作和贴近观众的剧情，最终斩获XX亿票房，成为春节档最大赢家...""",
        
        "科普百科": """请生成一个关于科普知识的高质量QA问答对，要求：
        1. 问题覆盖自然科学、社会科学等领域的知识点
        2. 回答准确易懂，有科学依据，不少于50字
        3. 避免过于简单的常识性问题
        示例格式：
        Q: 为什么黑洞会被称为"宇宙中的饕餮"？
        A: 黑洞之所以被称为"宇宙中的饕餮"，是因为它具有极强的引力，甚至连光都无法逃脱其引力范围...""",
        
        "历史人文": """请生成一个关于历史人文的高质量QA问答对，要求：
        1. 问题涵盖中外历史、文化、艺术等领域
        2. 回答基于史实，客观准确，不少于50字
        3. 避免历史虚无主义和错误信息
        示例格式：
        Q: 为什么说丝绸之路是古代东西方文明交流的重要通道？
        A: 丝绸之路不仅是商品贸易的通道，更是文化、宗教、技术交流的桥梁，它促进了中西方文明的相互了解和融合...""",
        
        "情感陪伴": """请生成一个关于情感陪伴的高质量QA问答对，要求：
        1. 问题贴近普通人的情感困扰和心理需求
        2. 回答温暖有同理心，提供实际的情感支持，不少于50字
        3. 避免生硬的说教
        示例格式：
        Q: 感到孤独无助时，应该如何自我调节？
        A: 感到孤独无助是很正常的情绪体验，首先要接纳自己的感受，然后可以尝试和信任的人倾诉，或者通过阅读、运动等方式转移注意力...""",
        
        "幽默段子与脑筋急转弯": """请生成一个关于幽默段子或脑筋急转弯的高质量QA问答对，要求：
        1. 内容健康向上，幽默风趣
        2. 回答贴合问题，有笑点或趣味性
        3. 避免低俗和攻击性内容
        示例格式：
        Q: 什么门永远关不上？
        A: 球门！因为球门是用来进球的，永远不需要关上，这也是足球运动的魅力所在...""",
        
        "生活方式建议": """请生成一个关于生活方式建议的高质量QA问答对，要求：
        1. 问题覆盖饮食、运动、睡眠、职场、社交等生活场景
        2. 回答实用可行，有科学依据，不少于50字
        3. 避免不切实际的建议
        示例格式：
        Q: 如何养成健康的睡眠习惯？
        A: 养成健康的睡眠习惯需要规律作息，建议每天固定时间上床和起床，睡前1小时远离电子设备，保持卧室安静、黑暗和凉爽..."""
    }
    
    return prompt_templates.get(category, 
        "请生成一个高质量的QA问答对，问题要具体，回答要详细且有价值，不少于50字。")

def call_qwen_api(prompt: str) -> Optional[str]:
    """调用Qwen2.5VL API生成回答"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "你是一个专业的QA问答数据生成助手，需要按照要求生成高质量的问答对。"},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,    # 控制生成多样性
        "max_tokens": 500      # 最大生成长度
    }
    
    for attempt in range(RETRY_TIMES):
        try:
            response = requests.post(
                API_BASE_URL,
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"].strip()
            
            else:
                print(f"API请求失败，状态码：{response.status_code}，尝试次数：{attempt+1}")
                time.sleep(2 ** attempt)  # 指数退避
            
        except Exception as e:
            print(f"API调用异常：{str(e)}，尝试次数：{attempt+1}")
            time.sleep(2 ** attempt)
    
    return None

def parse_qa_pair(raw_text: str) -> Optional[Dict]:
    """解析生成的文本为QA对"""
    try:
        # 提取Q和A部分
        q_start = raw_text.find("Q:")
        a_start = raw_text.find("A:")
        
        if q_start == -1 or a_start == -1:
            return None
        
        question = raw_text[q_start+2:a_start].strip()
        answer = raw_text[a_start+2:].strip()
        
        # 过滤空内容或过短的问答
        if len(question) < 10 or len(answer) < 50:
            return None
        
        return {
            "Q": question,
            "A": answer
        }
    
    except Exception as e:
        print(f"解析QA失败：{str(e)}")
        return None

def load_existing_data(category: str) -> List[Dict]:
    """加载已有的数据（用于去重）"""
    file_path = f"{OUTPUT_DIR}/{category}.json"
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return []

def save_qa_data(category: str, qa_list: List[Dict]):
    """保存QA数据到JSON文件"""
    file_path = f"{OUTPUT_DIR}/{category}.json"
    
    # 加载已有数据并去重
    existing_data = load_existing_data(category)
    existing_questions = set()
    for item in existing_data:
        if "question" in item:
            existing_questions.add(item["question"])
        elif "Q" in item:
            existing_questions.add(item["Q"])
    
    # 过滤重复数据
    new_qa_list = []
    for qa in qa_list:
        if qa["Q"] not in existing_questions:
            new_qa_list.append(qa)
            existing_questions.add(qa["Q"])
    
    all_data = []
    for item in existing_data + new_qa_list:
        filtered_item = {k: v for k, v in item.items() if k != "category"}
        all_data.append(filtered_item)
    
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n已保存 {category} 数据：总计 {len(all_data)} 条，新增 {len(new_qa_list)} 条")

def generate_category_data(category: str, target_count: int):
    """生成指定类别的QA数据"""
    print(f"\n开始生成【{category}】数据，目标：{target_count} 条")
    
    # 初始化
    init_output_dir()
    existing_data = load_existing_data(category)
    current_count = len(existing_data)
    need_generate = max(0, target_count - current_count)
    
    if need_generate <= 0:
        print(f"【{category}】已有 {current_count} 条数据，无需生成")
        return
    
    # 批量生成
    generated_count = 0
    pbar = tqdm(total=need_generate, desc=f"生成{category}")
    
    while generated_count < need_generate:
        # 每批生成BATCH_SIZE条
        batch_need = min(BATCH_SIZE, need_generate - generated_count)
        batch_qa = []
        
        for _ in range(batch_need):
            # 生成提示词并调用API
            prompt = generate_prompt(category)
            raw_response = call_qwen_api(prompt)
            
            if raw_response:
                qa_pair = parse_qa_pair(raw_response)
                if qa_pair:
                    batch_qa.append(qa_pair)
                    generated_count += 1
                    pbar.update(1)
            
            # 随机延迟，避免请求过快
            time.sleep(random.uniform(*DELAY_RANGE))
        
        # 保存批次数据
        if batch_qa:
            save_qa_data(category, batch_qa)
    
    pbar.close()
    print(f"【{category}】数据生成完成，最终数量：{len(load_existing_data(category))} 条")

def main():
    """主函数"""
    print("=== Qwen2.5VL QA数据生成脚本 ===")
    print(f"开始时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"目标类别：{list(CATEGORIES.keys())}")
    
    # 按类别生成数据
    for category, target_count in CATEGORIES.items():
        generate_category_data(category, target_count)
        # 类别间增加间隔
        time.sleep(5)
    
    print(f"\n=== 所有类别数据生成完成 ===")
    print(f"结束时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"输出目录：{OUTPUT_DIR}")
    
    # 输出统计信息
    print("\n=== 数据统计 ===")
    total = 0
    for category in CATEGORIES.keys():
        count = len(load_existing_data(category))
        total += count
        print(f"{category}: {count} 条")
    print(f"总计：{total} 条")

if __name__ == "__main__":
    # 检查API Key配置
    if API_KEY == "your_api_key_here":
        print("\033[91m警告：请先配置你的API Key！\033[0m")
    else:
        main()