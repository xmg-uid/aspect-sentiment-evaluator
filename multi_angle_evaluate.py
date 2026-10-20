"""多角度评论情感评价工具 (Multi-angle Review Sentiment Evaluation)

基于方面情感分析（ABSA, Aspect-Based Sentiment Analysis）的评论评价脚本。

功能概述
--------
给定一份商品 / 服务评论数据（Excel），本工具可以：

1. 按多个业务「角度」（方面 / aspect）配置关键词；
2. 对命中关键词的评论调用情感分类模型，进行细粒度情感打分；
3. 输出每条评论的情感得分、情感标签、命中关键词与上下文；
4. 汇总各角度的情感分布，并生成可视化图表。

依赖
----
pandas, numpy, matplotlib, seaborn, pyabsa

快速开始
--------
    # 使用内置的示例配置（某电商水果评论）
    python multi_angle_evaluate.py --input data/sample_reviews.xlsx

    # 使用自定义的方面配置
    python multi_angle_evaluate.py --input my_data.xlsx \
        --output-dir result --checkpoint chinese --config aspects_config.json

说明
----
- 输入 Excel 需至少包含一列评论文本，脚本会自动把第一列当作评论内容。
- 情感模型默认使用 PyABSA 的中文 checkpoint（首次运行会自动下载）。
"""

import os
import argparse
import json
import logging
import re
import pandas as pd
import numpy as np
from pyabsa import APCCheckpointManager
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------
# 日志与可视化样式
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

SEPARATOR = "=" * 60

plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题
sns.set(font_scale=1.1)
sns.set_style("whitegrid")

# ---------------------------------------------------------------------------
# 默认配置（可通过命令行参数覆盖）
# ---------------------------------------------------------------------------
DEFAULT_INPUT = "data/sample_reviews.xlsx"
DEFAULT_OUTPUT_DIR = "result"
DEFAULT_VIS_DIR = "可视化结果"
DEFAULT_CHECKPOINT = "chinese"


# ---------------------------------------------------------------------------
# 1. 模型加载
# ---------------------------------------------------------------------------
def load_sentiment_classifier(checkpoint):
    """加载 PyABSA 的情感分类模型（checkpoint 可为预训练名或本地路径）。"""
    print(f"\n{SEPARATOR}")
    print(f"开始加载情感分类模型: {checkpoint}")
    try:
        sentiment_classifier = APCCheckpointManager.get_sentiment_classifier(
            checkpoint=checkpoint,
            auto_device=False,  # 使用 CPU
        )
        print("情感分类模型加载完成")
        return sentiment_classifier
    except Exception as e:
        logger.error(f"加载模型时发生错误: {e}")
        return None


# ---------------------------------------------------------------------------
# 2. 关键词上下文提取
# ---------------------------------------------------------------------------
def find_keyword_context(text, keywords, window=30):
    """提取包含关键词的上下文片段，并把命中的关键词用【】标记出来。"""
    contexts = []
    for kw in keywords:
        for match in re.finditer(re.escape(kw), text):
            start = max(0, match.start() - window)
            end = min(len(text), match.end() + window)
            context = text[start:end].replace(kw, f"【{kw}】")  # 标记关键词
            contexts.append(context)
    return " | ".join(contexts) if contexts else ""


# ---------------------------------------------------------------------------
# 3. 单条评论情感打分与信息抽取
# ---------------------------------------------------------------------------
def score_and_extract(sentiment_classifier, df, keywords, aspect):
    """对 DataFrame 中的评论逐条打分，并抽取关键词上下文等信息。"""
    print(f"\n{SEPARATOR}")
    print(f"开始分析评论情感和提取信息")
    print(f"分析角度: {aspect}, 关键词: {', '.join(keywords)}")

    scores = []
    extracted_info = []
    total = len(df) if df is not None else 0

    if df is not None and sentiment_classifier is not None:
        for idx, text in enumerate(df["review"].astype(str)):
            if idx % 10 == 0 and idx > 0:
                print(f"已处理 {idx}/{total} 条评论")

            review_id = df["review_id"].iloc[idx]
            info = {
                "review_id": review_id,
                "full_text": text,
                "is_target": False,
                "keywords_found": [],
                "contexts": "",
                "aspect_terms": [],
                "sentiment_score": 0,
                "sentiment_label": "中性",
            }

            found_keywords = [kw for kw in keywords if kw in text]
            if found_keywords:
                info["is_target"] = True
                info["keywords_found"] = found_keywords
                info["contexts"] = find_keyword_context(text, found_keywords)

                try:
                    result = sentiment_classifier.predict(text, aspects=aspect, infer_aspect=True)
                    info["aspect_terms"] = result.get("aspect", [])

                    probs = result.get("probs", [])
                    if probs:
                        probs = probs[0]
                        positive_prob = probs[1] if len(probs) > 1 else 0
                        negative_prob = probs[0] if len(probs) > 0 else 0

                        if positive_prob > 0.8:
                            score, info["sentiment_label"] = 2, "强烈正面"
                        elif positive_prob > 0.5:
                            score, info["sentiment_label"] = 1, "正面"
                        elif negative_prob > 0.8:
                            score, info["sentiment_label"] = -2, "强烈负面"
                        elif negative_prob > 0.5:
                            score, info["sentiment_label"] = -1, "负面"
                        else:
                            score, info["sentiment_label"] = 0, "中性"

                        info["sentiment_score"] = score
                    else:
                        score = 0
                except Exception as e:
                    logger.error(f"处理评论ID {review_id} 时发生错误: {e}")
                    score = 0
            else:
                score = 0

            scores.append(score)
            extracted_info.append(info)

        print(f"评论分析完成，共处理 {total} 条评论")

    return scores, extracted_info


# ---------------------------------------------------------------------------
# 4. 结果保存
# ---------------------------------------------------------------------------
def save_results(df, output_path, scores, extracted_info, aspect_name):
    """把打分结果写回 DataFrame，并分别保存完整结果与目标角度提取结果。"""
    print(f"\n{SEPARATOR}")
    print(f"开始保存分析结果到文件")
    if df is None:
        print("无数据可保存")
        return None

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df["分数"] = scores
    df[f"{aspect_name}_情感标签"] = [info["sentiment_label"] for info in extracted_info]
    df[f"{aspect_name}_关键词"] = [",".join(info["keywords_found"]) for info in extracted_info]
    df[f"{aspect_name}_上下文"] = [info["contexts"] for info in extracted_info]
    df[f"{aspect_name}_相关短语"] = [",".join(info["aspect_terms"]) for info in extracted_info]

    try:
        df.to_csv(output_path, sep="\t", na_rep="nan", index=False)
        print(f"完整结果已保存: {output_path}")

        target_output = output_path.replace(".tsv", f"_{aspect_name}_提取.tsv")
        target_df = pd.DataFrame([info for info in extracted_info if info["is_target"]])
        if not target_df.empty:
            target_df.to_csv(target_output, sep="\t", na_rep="nan", index=False)
            print(f"提取信息已保存: {target_output}")
        else:
            print(f"未找到相关{aspect_name}评论，跳过保存提取文件")
        return target_df
    except Exception as e:
        logger.error(f"保存文件时发生错误: {e}")
        return None


# ---------------------------------------------------------------------------
# 5. 单维度分析（供独立调用）
# ---------------------------------------------------------------------------
def analyze_aspect(input_path, output_path, aspect_name, checkpoint="chinese", keywords=None):
    """分析某个单一角度的评论。"""
    print(f"\n{SEPARATOR}")
    print(f"开始进行【{aspect_name}】角度的评论分析")
    print(f"输入文件: {input_path}")
    print(f"输出文件: {output_path}")
    print(SEPARATOR)

    if keywords is None:
        keywords = []
    df = read_review_data(input_path)
    sentiment_classifier = load_sentiment_classifier(checkpoint)
    scores, extracted_info = score_and_extract(sentiment_classifier, df, keywords, [aspect_name])
    save_results(df, output_path, scores, extracted_info, aspect_name)

    if df is not None:
        stats = {
            "强烈正面 (2分)": (df["分数"] == 2).sum(),
            "正面 (1分)": (df["分数"] == 1).sum(),
            "中性 (0分)": (df["分数"] == 0).sum(),
            "负面 (-1分)": (df["分数"] == -1).sum(),
            "强烈负面 (-2分)": (df["分数"] == -2).sum(),
        }
        print(f"\n【{aspect_name}】评分统计结果:")
        for k, v in stats.items():
            print(f"  {k}: {v}")
    print(f"\n【{aspect_name}】角度的评论分析已完成")
    return df


# ---------------------------------------------------------------------------
# 6. 数据读取
# ---------------------------------------------------------------------------
def read_review_data(input_path):
    """读取评论 Excel，自动把第一列当作评论文本，并补充 review_id。"""
    print(f"\n{SEPARATOR}")
    print(f"开始读取评论数据: {input_path}")
    try:
        df = pd.read_excel(input_path)
        df.rename(columns={df.columns[0]: "review"}, inplace=True)
        df["review_id"] = range(1, len(df) + 1)  # 新增评论ID
        print(f"评论数据读取完成，共 {len(df)} 条记录")
        return df
    except FileNotFoundError:
        logger.error(f"未找到输入文件: {input_path}")
        return None
    except Exception as e:
        logger.error(f"读取文件时发生错误: {e}")
        return None


# ---------------------------------------------------------------------------
# 7. 批量多维度分析
# ---------------------------------------------------------------------------
def analyze_multiple_aspects(input_file_path, base_output_dir, aspects_config, checkpoint="chinese"):
    """依次分析多个维度，复用同一份数据与同一个模型，提升效率。"""
    print(f"\n{SEPARATOR}")
    print(f"开始批量分析多个维度（共 {len(aspects_config)} 个维度）")
    print(f"维度列表: {[cfg['aspect_name'] for cfg in aspects_config]}")
    print(SEPARATOR)

    df = read_review_data(input_file_path)
    if df is None:
        print("数据读取失败，无法进行批量分析")
        return None

    sentiment_classifier = load_sentiment_classifier(checkpoint)
    if sentiment_classifier is None:
        print("模型加载失败，无法进行批量分析")
        return None

    all_results = []
    for idx, config in enumerate(aspects_config, 1):
        aspect_name = config["aspect_name"]
        keywords = config.get("keywords", [])
        output_path = os.path.join(base_output_dir, f"{aspect_name}_analysis.tsv")

        print(f"\n{SEPARATOR}")
        print(f"正在分析维度 {idx}/{len(aspects_config)}: {aspect_name}")
        print(SEPARATOR)

        scores, extracted_info = score_and_extract(
            sentiment_classifier, df.copy(), keywords, [aspect_name]
        )
        save_results(df.copy(), output_path, scores, extracted_info, aspect_name)

        all_results.append({
            "aspect_name": aspect_name,
            "score_2_count": sum(1 for s in scores if s == 2),
            "score_1_count": sum(1 for s in scores if s == 1),
            "score_0_count": sum(1 for s in scores if s == 0),
            "score_neg1_count": sum(1 for s in scores if s == -1),
            "score_neg2_count": sum(1 for s in scores if s == -2),
            "target_count": int(((df["分数"] != 0)).sum()) if "分数" in df else 0,
        })

    print(f"\n{SEPARATOR}")
    print(f"所有 {len(aspects_config)} 个维度分析已完成")
    print(f"结果保存目录: {base_output_dir}")
    print(SEPARATOR)
    return all_results


# ---------------------------------------------------------------------------
# 8. 可视化
# ---------------------------------------------------------------------------
def plot_aspect_sentiment_scores(df, output_dir=DEFAULT_VIS_DIR):
    """各维度平均情感得分条形图。"""
    if df.empty:
        print("没有足够数据生成情感得分图")
        return
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(12, 8))
    df_sorted = df.sort_values("平均情感得分", ascending=False)
    bars = plt.bar(
        df_sorted["维度"],
        df_sorted["平均情感得分"],
        color=np.where(df_sorted["平均情感得分"] > 0, "skyblue", "salmon"),
    )
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, height, f"{height:.2f}",
                 ha="center", va="bottom", fontsize=10)
    plt.axhline(y=0, color="gray", linestyle="--", alpha=0.7)
    plt.title("各维度平均情感得分", fontsize=16)
    plt.xlabel("分析维度", fontsize=14)
    plt.ylabel("平均情感得分", fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/各维度情感得分.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"各维度情感得分图已保存至 {output_dir}")


def plot_aspect_mentions(df, output_dir=DEFAULT_VIS_DIR):
    """各维度提及次数条形图。"""
    if df.empty:
        print("没有足够数据生成提及次数图")
        return
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(12, 8))
    df_sorted = df.sort_values("提及次数", ascending=False)
    bars = plt.bar(df_sorted["维度"], df_sorted["提及次数"], color="lightgreen")
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, height, f"{height}",
                 ha="center", va="bottom", fontsize=10)
    plt.title("各维度提及次数", fontsize=16)
    plt.xlabel("分析维度", fontsize=14)
    plt.ylabel("提及次数", fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/各维度提及次数.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"各维度提及次数图已保存至 {output_dir}")


def plot_aspect_sentiment_distribution(df, output_dir=DEFAULT_VIS_DIR):
    """各维度情感分布堆叠条形图（正面/中性/负面）。"""
    if df.empty:
        print("没有足够数据生成情感分布图")
        return
    os.makedirs(output_dir, exist_ok=True)
    df_plot = df.set_index("维度")[["正面评论数", "中性评论数", "负面评论数"]]
    plt.figure(figsize=(14, 8))
    df_plot.plot(kind="bar", stacked=True, color=["green", "gray", "red"], figsize=(14, 8))
    plt.title("各维度情感分布", fontsize=16)
    plt.xlabel("分析维度", fontsize=14)
    plt.ylabel("评论数量", fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="情感类型")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/各维度情感分布.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"各维度情感分布图已保存至 {output_dir}")


def process_aspect_sentiment_data(results):
    """把原始分析结果整理成 DataFrame，便于可视化。"""
    aspect_data = []
    for res in results:
        aspect_name = res["aspect_name"]
        total_mentions = res["target_count"]
        positive_count = res["score_2_count"] + res["score_1_count"]
        negative_count = res["score_neg2_count"] + res["score_neg1_count"]
        neutral_count = res["score_0_count"]

        if total_mentions > 0:
            total_score = (
                res["score_2_count"] * 2
                + res["score_1_count"] * 1
                + res["score_neg1_count"] * -1
                + res["score_neg2_count"] * -2
            )
            avg_score = total_score / total_mentions
        else:
            avg_score = 0

        aspect_data.append({
            "维度": aspect_name,
            "平均情感得分": round(avg_score, 2),
            "提及次数": total_mentions,
            "正面评论数": positive_count,
            "负面评论数": negative_count,
            "中性评论数": neutral_count,
        })
    return pd.DataFrame(aspect_data)


# ---------------------------------------------------------------------------
# 9. 入口
# ---------------------------------------------------------------------------
# 示例配置：某电商「水果」评论的多角度分析。
# 使用时请替换为你的业务角度与关键词。
EXAMPLE_ASPECTS = [
    {"aspect_name": "品牌对消费者反馈的响应速度",
     "keywords": ["回复", "及时", "响应", "客服", "有问必答", "客服不回复", "处理", "解决问题", "解决问题高效"]},
    {"aspect_name": "购买的便利性",
     "keywords": ["发货", "速度慢", "顺丰", "物流", "送货", "取件码", "才到", "到家", "准时", "快速", "驿站", "物流慢"]},
    {"aspect_name": "质量和新鲜度",
     "keywords": ["新鲜", "完好", "黑", "坏", "磕", "损", "烂", "刚摘", "无坏果", "无磕碰", "果香浓郁", "保质期新", "现摘现发", "无异味", "保存完好", "腐烂", "发霉", "大", "小", "足", "分量", "饱满", "称", "偏", "超", "个头", "径", "个大质优", "大小均匀", "符合规格", "偏小", "大果", "足斤足两", "个头适中", "单个分量足", "比预期小"]},
    {"aspect_name": "品牌提供的个性化体验",
     "keywords": ["独特", "老人", "宝宝", "糖心", "粉面特性", "可脆可面", "辅食", "纯天然", "特殊", "特别的", "适合", "小孩"]},
    {"aspect_name": "品牌的形象和口碑",
     "keywords": ["回购", "再买", "继续", "推荐", "上当", "老客户", "第二次购买", "习惯", "一直", "还会买", "就爱", "多次购买", "继续光顾", "下次还来", "认准这家", "习惯性购买", "二次回购", "复购率高", "推荐给同事", "避雷", "不会回购", "不会再买", "失望", "黑心", "再次购买"]},
    {"aspect_name": "价格",
     "keywords": ["值", "不值", "高", "贵", "便宜", "划算", "实惠", "物超所值", "性价比", "价格", "超值", "超划算", "不划算", "物有所值", "性价比极高", "比实体店便宜", "价格高", "一分钱一分货", "买不起", "吃不起"]},
    {"aspect_name": "品牌的售后服务",
     "keywords": ["店家", "商家", "服务", "卖家", "态度差", "态度极差", "补偿", "态度好", "售后", "保障", "坏果补偿"]},
    {"aspect_name": "包装和使用便利性",
     "keywords": ["破", "简陋", "严实", "破损", "送货上门", "时效快", "速度快", "包装", "给力", "挤压", "带箱称重", "保护"]},
    {"aspect_name": "品牌的市场影响力",
     "keywords": ["市场", "比之前买的好", "不如第一次", "认准这家", "习惯性购买", "复购率高", "推荐给同事"]},
    {"aspect_name": "产地特色和地域文化",
     "keywords": ["源", "虚假", "正宗", "原产地", "特产", "货源", "天水花牛", "产地", "天水", "发货地点", "地理标志产品", "一点都不正宗", "花牛苹果", "地理标志", "果香", "果香浓郁", "就是香", "就是", "这样的", "这个苹果", "这种苹果"]},
]


def main():
    parser = argparse.ArgumentParser(
        description="多角度评论情感评价工具：按业务角度对评论进行细粒度情感分析。"
    )
    parser.add_argument("--input", default=DEFAULT_INPUT, help="输入评论 Excel 路径")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="结果输出目录")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="PyABSA 情感模型 checkpoint")
    parser.add_argument("--config", default=None,
                        help="可选：包含 aspects 配置的 JSON 文件路径（覆盖内置示例配置）")
    args = parser.parse_args()

    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            aspects_config = json.load(f)
    else:
        aspects_config = EXAMPLE_ASPECTS

    print(SEPARATOR)
    print("启动多角度评价分析程序")
    print(f"启动时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(SEPARATOR)

    analysis_results = analyze_multiple_aspects(
        input_file_path=args.input,
        base_output_dir=args.output_dir,
        aspects_config=aspects_config,
        checkpoint=args.checkpoint,
    )

    if analysis_results:
        aspect_df = process_aspect_sentiment_data(analysis_results)
        if not aspect_df.empty:
            print("\n各维度情感分析摘要:")
            print(aspect_df[["维度", "平均情感得分", "提及次数"]].to_string(index=False))
            plot_aspect_sentiment_scores(aspect_df)
            plot_aspect_mentions(aspect_df)
            plot_aspect_sentiment_distribution(aspect_df)
        else:
            print("没有生成有效的情感分析数据，无法进行可视化")
    else:
        print("分析结果为空，无法进行可视化")

    print(f"\n{SEPARATOR}")
    print("所有分析任务已完成")
    print(f"结束时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(SEPARATOR)


if __name__ == "__main__":
    main()
