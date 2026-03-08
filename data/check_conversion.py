import json

# 读取转换后的ShareGPT文件
with open('/Users/xiezejing/PycharmProjects/prototypical networks/TV_Assistant/data/dataset/历史人文_sharegpt.json', 'r', encoding='utf-8') as f:
    sharegpt_data = json.load(f)

print(f"Total entries: {len(sharegpt_data)}")

# 检查第100条数据
print("\n第100条数据:")
print(json.dumps(sharegpt_data[99], ensure_ascii=False, indent=2))

# 检查第500条数据
print("\n第500条数据:")
print(json.dumps(sharegpt_data[499], ensure_ascii=False, indent=2))

# 检查最后几条数据
print("\n最后3条数据:")
for i in range(-3, 0):
    print(f"\n第{len(sharegpt_data) + i + 1}条:")
    print(json.dumps(sharegpt_data[i], ensure_ascii=False, indent=2))
