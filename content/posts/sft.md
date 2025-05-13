+++
title = '使用deepseek微调原神角色基本信息的大模型'
date = 2025-05-13T15:56:41+08:00
draft = false
+++

### 简介

最近两个月，业余时间研究了大语言模型的微调技巧，也成功微调出了一个原神的大模型

我微调的大模型要达到的效果是

```
问:钟离是什么性别?
答:钟离是男性

问:钟离来自什么国家?
答:钟离来自璃月

问:钟离掌握什么元素力？
答:岩元素

问：钟离的身份是什么?
答:往生堂客卿，岩神

问: 钟离的性格特点是什么？
答: 沉稳，深知璃月文化底蕴
```



### 准备事项

#### 数据集

我的原始数据集是从huggingface下载的，原始数据格式大致如下

```json
{
    "丽莎": {
        "性别": "成年女性",
        "国籍": "蒙德",
        "元素力": "雷元素",
        "身份": "西风骑士团图书管理员",
        "性格特征": "喜欢挑逗人的魔法师大姐姐"
    },
    "行秋": {
        "性别": "少年男性",
        "国籍": "璃月",
        "元素力": "水元素",
        "身份": "飞云商会少爷",
        "性格特征": "行侠仗义的侠客"
    },
    "钟离": {
        "性别": "成年男性",
        "国籍": "璃月",
        "元素力": "岩元素",
        "身份": "往生堂客卿，岩神",
        "性格特征": "沉稳，深知璃月文化底蕴"
    },
    "温迪": {
        "性别": "少年男性",
        "国籍": "蒙德",
        "元素力": "风元素",
        "身份": "吟游诗人，风神",
        "性格特征": "喜欢开玩笑的轻佻少年"
    }
}
```

这样的数据集不能直接喂给大模型，我们需要处理，对于每一个角色，都做如下处理，以钟离为例

```json
[{
        "role":"user",
        "content":"请问钟离的性别是什么？"
    },
    {
        "role":"assistant",
        "content":"钟离是男性"
    },
    {
        "role":"user",
        "content":"钟离是男性还是女性？"
    },
    {
        "role":"assistant",
        "content":"钟离是男性"
    },
    {
        "role":"user",
        "content":"钟离是男性还是女性？"
    },
    {
        "role":"assistant",
        "content":"钟离是男性"
    },
    {
        "role":"user",
        "content":"钟离是男人还是女人？"
    },
    {
        "role":"assistant",
        "content":"钟离是男性"
    },
    {
        "role":"user",
        "content":"钟离先生,他是男孩还是女孩？"
    },
    {
        "role":"assistant",
        "content":"钟离是男性"
    },
    {
        "role":"user",
        "content":"钟离先生,他是什么性别？"
    },
    {
        "role":"assistant",
        "content":"钟离是男性"
    },
    {
        "role":"user",
        "content":"钟离先生,是什么性别呢?"
    },
    {
        "role":"assistant",
        "content":"钟离是男性"
    },
    {
        "role":"user",
        "content":"钟离什么性别"
    },
    {
        "role":"assistant",
        "content":"钟离是男性"
    },
    {
        "role":"user",
        "content":"钟离是男还是女啊"
    },
    {
        "role":"assistant",
        "content":"钟离是男性"
    },
    {
        "role":"user",
        "content":"钟离是男孩还是女孩？"
    },
    {
        "role":"assistant",
        "content":"钟离是男性"
    },
    {
        "role":"user",
        "content":"钟离先生是男孩还是女孩？"
    },
    {
        "role":"assistant",
        "content":"钟离是男性"
    },
    {
        "role":"user",
        "content":"钟离先生是什么性别？"
    },
    {
        "role":"assistant",
        "content":"钟离是男性"
    },
    {
        "role":"user",
        "content":"请问钟离的国籍是什么？"
    },
    {
        "role":"assistant",
        "content":"钟离来自璃月"
    },
    {
        "role":"user",
        "content":"钟离先生是属于哪个国家的?"
    },
    {
        "role":"assistant",
        "content":"钟离来自璃月"
    },
    {
        "role":"user",
        "content":"钟离先生来自哪个国家?"
    },
    {
        "role":"assistant",
        "content":"钟离来自璃月"
    },
    {
        "role":"user",
        "content":"你好,请问钟离先生来自哪里?"
    },
    {
        "role":"assistant",
        "content":"钟离来自璃月"
    },
    {
        "role":"user",
        "content":"钟离先生从哪里来的?"
    },
    {
        "role":"assistant",
        "content":"钟离来自璃月"
    },
    {
        "role":"user",
        "content":"钟离从哪里来的?"
    },
    {
        "role":"assistant",
        "content":"钟离来自璃月"
    },
    {
        "role":"user",
        "content":"钟离先生,来自何方?"
    },
    {
        "role":"assistant",
        "content":"钟离来自璃月"
    },
    {
        "role":"user",
        "content":"钟离来自哪个国家?"
    },
    {
        "role":"assistant",
        "content":"钟离来自璃月"
    },
    {
        "role":"user",
        "content":"钟离先生来自什么国家?"
    },
    {
        "role":"assistant",
        "content":"钟离来自璃月"
    },
    {
        "role":"user",
        "content":"请问钟离的元素力是什么？"
    },
    {
        "role":"assistant",
        "content":"岩元素"
    },
    {
        "role":"user",
        "content":"钟离先生掌握哪种元素力?"
    },
    {
        "role":"assistant",
        "content":"岩元素"
    },
    {
        "role":"user",
        "content":"钟离先生能控制什么元素力?"
    },
    {
        "role":"assistant",
        "content":"岩元素"
    },
    {
        "role":"user",
        "content":"你好,请问钟离先生使用哪种元素?"
    },
    {
        "role":"assistant",
        "content":"岩元素"
    },
    {
        "role":"user",
        "content":"你好,钟离先生使用哪种元素?"
    },
    {
        "role":"assistant",
        "content":"岩元素"
    },
    {
        "role":"user",
        "content":"钟离使用哪种元素?"
    },
    {
        "role":"assistant",
        "content":"岩元素"
    },
    {
        "role":"user",
        "content":"钟离先生会用哪种元素?"
    },
    {
        "role":"assistant",
        "content":"岩元素"
    },
    {
        "role":"user",
        "content":"钟离使用的元素是什么?"
    },
    {
        "role":"assistant",
        "content":"岩元素"
    },
    {
        "role":"user",
        "content":"钟离的元素是什么?"
    },
    {
        "role":"assistant",
        "content":"岩元素"
    },
    {
        "role":"user",
        "content":"钟离先生的元素是什么?"
    },
    {
        "role":"assistant",
        "content":"岩元素"
    },
    {
        "role":"user",
        "content":"钟离用的是什么元素?"
    },
    {
        "role":"assistant",
        "content":"岩元素"
    },
    {
        "role":"user",
        "content":"钟离的元素属性是什么?"
    },
    {
        "role":"assistant",
        "content":"岩元素"
    },
    {
        "role":"user",
        "content":"钟离能控制什么元素力?"
    },
    {
        "role":"assistant",
        "content":"钟离控制岩元素"
    },
    {
        "role":"user",
        "content":"钟离会用哪种元素?"
    },
    {
        "role":"assistant",
        "content":"岩元素"
    },
    {
        "role":"user",
        "content":"钟离什么元素?"
    },
    {
        "role":"assistant",
        "content":"岩元素"
    },
    {
        "role":"user",
        "content":"钟离使用什么元素力?"
    },
    {
        "role":"assistant",
        "content":"钟离使用岩元素"
    },
    {
        "role":"user",
        "content":"钟离会用哪种元素?"
    },
    {
        "role":"assistant",
        "content":"岩元素"
    },
    {
        "role":"user",
        "content":"钟离的元素是什么?"
    },
    {
        "role":"assistant",
        "content":"岩元素"
    },
    {
        "role":"user",
        "content":"请问钟离的身份是什么？"
    },
    {
        "role":"assistant",
        "content":"钟离是往生堂客卿，岩神"
    },
    {
        "role":"user",
        "content":"钟离先生是什么身份?"
    },
    {
        "role":"assistant",
        "content":"钟离是往生堂客卿，岩神"
    },
    {
        "role":"user",
        "content":"钟离先生的职责是什么?"
    },
    {
        "role":"assistant",
        "content":"钟离是往生堂客卿，岩神"
    },
    {
        "role":"user",
        "content":"你好,请问钟离先生是什么职务?"
    },
    {
        "role":"assistant",
        "content":"钟离是往生堂客卿，岩神"
    },
    {
        "role":"user",
        "content":"你好,请问钟离先生是做什么工作的?"
    },
    {
        "role":"assistant",
        "content":"钟离是往生堂客卿，岩神"
    },
    {
        "role":"user",
        "content":"钟离先生是做什么工作的?"
    },
    {
        "role":"assistant",
        "content":"钟离是往生堂客卿，岩神"
    },
    {
        "role":"user",
        "content":"钟离先生的身份是什么?"
    },
    {
        "role":"assistant",
        "content":"钟离是往生堂客卿，岩神"
    },
    {
        "role":"user",
        "content":"钟离先生的职业是什么?"
    },
    {
        "role":"assistant",
        "content":"钟离是往生堂客卿，岩神"
    },
    {
        "role":"user",
        "content":"钟离的职业是什么?"
    },
    {
        "role":"assistant",
        "content":"钟离是往生堂客卿，岩神"
    },
    {
        "role":"user",
        "content":"钟离的工作是什么?"
    },
    {
        "role":"assistant",
        "content":"钟离是往生堂客卿，岩神"
    },
    {
        "role":"user",
        "content":"钟离先生的工作是什么?"
    },
    {
        "role":"assistant",
        "content":"钟离是往生堂客卿，岩神"
    },
    {
        "role":"user",
        "content":"钟离是什么身份?"
    },
    {
        "role":"assistant",
        "content":"钟离是往生堂客卿，岩神"
    },
    {
        "role":"user",
        "content":"钟离的职责是什么?"
    },
    {
        "role":"assistant",
        "content":"钟离是往生堂客卿，岩神"
    },
    {
        "role":"user",
        "content":"钟离是什么职务?"
    },
    {
        "role":"assistant",
        "content":"钟离是往生堂客卿，岩神"
    },
    {
        "role":"user",
        "content":"请问钟离的性格特征是什么？"
    },
    {
        "role":"assistant",
        "content":"钟离,沉稳，深知璃月文化底蕴"
    },
    {
        "role":"user",
        "content":"钟离先生是什么性格?"
    },
    {
        "role":"assistant",
        "content":"钟离,沉稳，深知璃月文化底蕴"
    },
    {
        "role":"user",
        "content":"钟离先生是什么性格的人?"
    },
    {
        "role":"assistant",
        "content":"钟离,沉稳，深知璃月文化底蕴"
    },
    {
        "role":"user",
        "content":"你好,请问钟离先生的性格特点是什么?"
    },
    {
        "role":"assistant",
        "content":"钟离,沉稳，深知璃月文化底蕴"
    },
    {
        "role":"user",
        "content":"你好,请问钟离先生有什么性格特点呢?"
    },
    {
        "role":"assistant",
        "content":"钟离,沉稳，深知璃月文化底蕴"
    },
    {
        "role":"user",
        "content":"你好,请问钟离有什么性格特点呢?"
    },
    {
        "role":"assistant",
        "content":"钟离,沉稳，深知璃月文化底蕴"
    },
    {
        "role":"user",
        "content":"你好,请问钟离的性格特点是什么?"
    },
    {
        "role":"assistant",
        "content":"钟离,沉稳，深知璃月文化底蕴"
    },
    {
        "role":"user",
        "content":"钟离是什么性格?"
    },
    {
        "role":"assistant",
        "content":"钟离,沉稳，深知璃月文化底蕴"
    },
    {
        "role":"user",
        "content":"钟离是什么性格的人?"
    },
    {
        "role":"assistant",
        "content":"钟离,沉稳，深知璃月文化底蕴"
    },
    {
        "role":"user",
        "content":"钟离什么性格?"
    },
    {
        "role":"assistant",
        "content":"沉稳，深知璃月文化底蕴"
    },
    {
        "role":"user",
        "content":"钟离是什么性格?"
    },
    {
        "role":"assistant",
        "content":"沉稳，深知璃月文化底蕴"
    },
    {
        "role":"user",
        "content":"钟离有什么性格特征?"
    },
    {
        "role":"assistant",
        "content":"沉稳，深知璃月文化底蕴"
    },
    {
        "role":"user",
        "content":"钟离的性格特征是什么?"
    },
    {
        "role":"assistant",
        "content":"钟离,沉稳，深知璃月文化底蕴"
    },
    {
        "role":"user",
        "content":"钟离的性格特点是啥?"
    },
    {
        "role":"assistant",
        "content":"沉稳，深知璃月文化底蕴"
    },
    {
        "role":"user",
        "content":"钟离的性格是什么?"
    },
    {
        "role":"assistant",
        "content":"钟离,沉稳，深知璃月文化底蕴"
    }]
```

对于每个问题，我设计多样的提问方式，这样可以让模型更好地理解我的提问

#### 训练

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
check_point='deepseek-ai/DeepSeek-R1-Distill-Qwen-7B'  
# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(check_point) 
# 加载大模型
model = AutoModelForCausalLM.from_pretrained(check_point,torch_dtype = torch.float16, device_map ="auto")
print(tokenizer)
```

首先加载模型以及它的tokenizer，tokenizer是分词器的意思。tokenizer在大语言模型中是一个非常重要的概念。比如 “钟离来自什么国家”，会被分词为

```
["<｜begin▁of▁sentence｜>","钟"，"离"，"来自","什么","国家"]
```
并且转换成一串数字

```
[151646,  75061,  99372, 101919,  99245,  99599]
```

这些都是分词器提前设置好的，在大模型目录下的tokenizer.json中可以查询到具体的分词详情


```python
tokenizer.add_special_tokens({
    "additional_special_tokens": ["<｜User｜>", "<｜Assistant｜>"]
})
tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
tokenizer.pad_token = '<|pad|>'
print(tokenizer.pad_token_id)
print(tokenizer.eos_token_id)
model.resize_token_embeddings(len(tokenizer))
```

添加几个special_tokens，这个后面再讲

我的数据集托管在了huggingface，从这个地址下载
```python
# 加载数据集
from datasets import load_dataset
raw_datasets=load_dataset('maxwell60701/genshin-impact-role-chat-model')
raw_datasets
```


```
DatasetDict({
    train: Dataset({
        features: ['messages'],
        num_rows: 3480
    })
    test: Dataset({
        features: ['messages'],
        num_rows: 870
    })
})
```

可以看到我的训练集为3480行，测试集为870行

```python
raw_datasets['train'][0]
```
打印一笔训练数据集看看格式
```
{'messages': [{'content': '砂糖女士来自什么国家?', 'role': 'user'},
  {'content': '砂糖来自蒙德', 'role': 'assistant'}]}
```

```python
def tokenizer_convert(example):
  prompt=tokenizer.apply_chat_template(example['messages'],tokenize=False)
  return {'text':prompt}
```