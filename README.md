# 多角度评论情感评价工具 (Multi-angle Review Sentiment Evaluation)

一个基于**方面情感分析（ABSA, Aspect-Based Sentiment Analysis）**的评论评价脚本。
给定一份商品 / 服务评论数据，可按多个业务「角度」（方面 / aspect）配置关键词，
对命中关键词的评论进行细粒度情感打分，并输出统计结果与可视化图表。

> 本仓库整理自一份早期的个人练手脚本，已做可配置化与清理，便于分享与复用。

## ✨ 功能

- 按多个业务角度（如「价格」「物流」「售后服务」）配置关键词；
- 对命中关键词的评论调用情感分类模型，输出 `强烈正面 / 正面 / 中性 / 负面 / 强烈负面` 五档评分；
- 抽取命中的关键词上下文与模型识别出的相关短语；
- 汇总各角度的情感分布，并生成三张可视化图表（情感得分、提及次数、情感分布）。

## 📦 安装

```bash
pip install -r requirements.txt
```

情感模型默认使用 [PyABSA](https://github.com/yangheng95/PyABSA) 的中文 checkpoint，
首次运行时会自动下载，请保持网络畅通。

## 🚀 使用

### 1. 准备数据

输入为 Excel 文件，需至少包含一列评论文本（脚本会把**第一列**当作评论内容）。
仓库内置了一份示例数据 `data/sample_reviews.xlsx`，可直接体验：

```bash
python multi_angle_evaluate.py --input data/sample_reviews.xlsx
```

### 2. 使用自定义配置

内置的 `EXAMPLE_ASPECTS` 是一份「电商水果评论」的示例配置。你可以：

- 直接修改脚本中的 `EXAMPLE_ASPECTS`；或
- 把方面配置写成 JSON（参考 `aspects_config.example.json`），通过 `--config` 传入。

```bash
python multi_angle_evaluate.py \
    --input my_data.xlsx \
    --output-dir result \
    --checkpoint chinese \
    --config aspects_config.example.json
```

### 命令行参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--input` | `data/sample_reviews.xlsx` | 输入评论 Excel 路径 |
| `--output-dir` | `result` | 分析结果输出目录 |
| `--checkpoint` | `chinese` | PyABSA 情感模型 checkpoint（或本地路径） |
| `--config` | 无 | 可选的方面配置 JSON，覆盖内置示例 |

## 📂 输出

- `result/<角度名>_analysis.tsv`：完整结果（含每条评论的评分、标签、关键词、上下文）；
- `result/<角度名>_analysis_<角度名>_提取.tsv`：仅命中该角度的评论；
- `可视化结果/`：三张汇总图表（PNG）。

## 📝 说明

- 关键词匹配为「包含即命中」的朴素策略，适合做初步的角度下钻；
- 情感打分依赖预训练模型，结果仅供分析参考；
- 中文字体使用 `SimHei` 等，若在 Linux 服务器运行缺少中文字体，图表中文可能显示为方块，
  可安装对应字体或在脚本顶部 `plt.rcParams["font.family"]` 处调整。

## 📄 License

仅供学习与分享用途，欢迎 Fork 与改进。
