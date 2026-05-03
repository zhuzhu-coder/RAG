# Recipe RAG

这是一个基于 LangChain 1.x 的食谱 RAG 示例项目。系统会读取 Markdown 食谱，构建 FAISS 向量索引，并结合 BM25 与向量检索回答做菜相关问题。

## 目录结构

```text
code/
  main.py
  config.py
  rag_modules/
  tests/
data/
  cook/
```

默认读取 `data/cook` 下的 Markdown 文件，索引默认保存到 `vector_index`。路径由 `config.py` 按项目根目录解析，所以从项目根或 `code/` 目录运行都能得到一致结果。

## 安装

```powershell
cd E:\RAG\code
python -m pip install -r requirements.txt
```

## 配置

复制 `.env.example` 为 `.env`，然后填入 DashScope API Key。

```env
DASHSCOPE_API_KEY=your_dashscope_api_key_here
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

可选的 `RAG_*` 配置项：

```env
RAG_DATA_PATH=data/cook
RAG_INDEX_SAVE_PATH=vector_index
RAG_EMBEDDING_MODEL=text-embedding-v4
RAG_LLM_MODEL=qwen-plus
RAG_TOP_K=3
RAG_RETRIEVAL_CANDIDATE_K=10
RAG_TEMPERATURE=0.1
RAG_MAX_TOKENS=2048
```

## 准备数据

把食谱 Markdown 放到 `data/cook` 目录，建议按分类子目录组织，例如：

```text
data/cook/vegetable_dish/番茄炒蛋.md
data/cook/meat_dish/红烧肉.md
```

分类目录会被映射成菜品分类，难度会根据正文里的星级标记识别。

## 运行

```powershell
cd E:\RAG\code
python main.py
```

## 测试

```powershell
cd E:\RAG\code
python -m pytest tests -q -p no:cacheprovider
```
