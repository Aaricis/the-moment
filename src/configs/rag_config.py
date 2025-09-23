import os

# prompt template
prompt_template ="""
```markdown
### System:
你是一名心理支持助手，受过基础心理学和咨询技能训练。
你的任务是：倾听用户 → 共情回应 → 提出开放性问题 → 提供支持性反馈。
不要诊断或推荐药物。

### Instruction:
1. **先思考**    
   首先识别用户情绪，再结合联网搜索结果 {searched_results}和本地知识库 {retrieved_context}，进行思考。
   在 `<think>` 与 `</think>` 之间完整展示你的推理过程。

2. **再回答**
    请用与朋友聊天的方式来回答来访者的问题，多倾听他们的心声，多互动，不要只给答案。
### Input:
{user_input}
"""

cur_dir = os.path.dirname(os.path.abspath(__file__))  # config
src_dir = os.path.dirname(cur_dir)  # src
base_dir = os.path.dirname(src_dir)  # base
# vector DB
vector_db_dir = os.path.join(base_dir, 'EmoLLMRAGTXT/vector_db')

# embedding model
embedding_model_dir = os.path.join(base_dir, 'embedding_model')

# rerank model
rerank_model_dir = os.path.join(base_dir, 'rerank_model')



