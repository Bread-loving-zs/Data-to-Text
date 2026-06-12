import argparse
import json
from pathlib import Path

from src.config import OUTPUT_DIR, setup_logging
from src.data.loader import DataLoader
from src.agent.intent import IntentRecognizer
from src.agent.query import DataQuerier
from src.agent.statistics import StatisticsEngine
from src.agent.generator import ReportGenerator
from src.report.template import ReportTemplate
from src.training.prepare_data import TrainingDataPreparer

logger = setup_logging("cli")


def cmd_training_stats(args):
    preparer = TrainingDataPreparer()
    stats = preparer.get_statistics(args.file)
    logger.info("训练数据统计:")
    for k, v in stats.items():
        logger.info(f"  {k}: {v}")


def cmd_training_import(args):
    preparer = TrainingDataPreparer()
    filepath = Path(args.input)
    if not filepath.exists():
        logger.error(f"文件不存在: {filepath}")
        return

    if filepath.suffix == ".jsonl":
        samples = preparer.load_samples_from_jsonl(filepath)
    elif filepath.suffix == ".csv":
        if not args.output_csv:
            logger.error("CSV配对模式需要 --output-csv 指定输出CSV文件")
            return
        samples = preparer.load_samples_from_csv_pairs(filepath, Path(args.output_csv))
    else:
        logger.error(f"不支持的文件格式: {filepath.suffix}")
        return

    if not samples:
        logger.warning("未加载到任何样本")
        return

    output_name = args.output_name or "training_data.jsonl"
    preparer.process_real_samples(samples, output_name, mode=args.mode)
    logger.info("训练数据导入完成")


def cmd_training_export(args):
    preparer = TrainingDataPreparer()
    output = preparer.export_for_finetuning(args.input, args.output)
    if output.exists():
        logger.info(f"微调格式数据已导出: {output}")


def cmd_evaluate(args):
    from src.training.evaluate import Evaluator

    test_path = Path(args.input)
    if not test_path.exists():
        logger.error(f"测试数据不存在: {test_path}")
        return

    generator = ReportGenerator(
        model=args.model,
        host=args.host,
        backend=args.backend,
    )

    evaluator = Evaluator(test_path)
    results = evaluator.evaluate_batch(generator)

    if not results:
        logger.warning("无评估结果")
        return

    avg_overall = sum(r.overall_score for r in results) / len(results)
    avg_fact = sum(r.fact_accuracy for r in results) / len(results)
    avg_struct = sum(r.structure_score for r in results) / len(results)

    logger.info(f"评估结果: overall={avg_overall:.4f}, fact_accuracy={avg_fact:.4f}, structure={avg_struct:.4f}")

    if args.output:
        output_data = {
            "summary": {
                "num_samples": len(results),
                "avg_overall_score": round(avg_overall, 4),
                "avg_fact_accuracy": round(avg_fact, 4),
                "avg_structure_score": round(avg_struct, 4),
            },
            "details": [
                {
                    "index": r.sample_index,
                    "overall_score": r.overall_score,
                    "fact_accuracy": r.fact_accuracy,
                    "structure_score": r.structure_score,
                    "word_overlap": r.word_overlap_ratio,
                }
                for r in results
            ]
        }
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        logger.info(f"评估结果已保存: {output_path}")


def cmd_generate_report(args):
    logger.info(f"正在处理: {args.prompt}")

    logger.info("[1/4] 识别意图...")
    recognizer = IntentRecognizer()
    intent = recognizer.recognize(args.prompt)
    logger.info(f"  意图: {intent}")

    logger.info("[2/4] 查询数据...")
    querier = DataQuerier()
    results = querier.query_by_intent(intent)
    table_count = sum(1 for v in results.values() if v is not None and not v.empty)
    logger.info(f"  获取到 {table_count} 张数据表")

    logger.info("[3/4] 统计计算...")
    stats_engine = StatisticsEngine()
    stats_results = {}
    for key, df in results.items():
        if df is not None and not df.empty:
            try:
                rate_cols = [c for c in df.columns if c in ("rate", "bhgl", "niandu_bhgl", "bhgl_new")]
                if rate_cols and len(df) >= 2:
                    stats_results[key] = stats_engine.summarize_rates(df, rate_col=rate_cols[0])
                    logger.debug(f"  统计计算 [{key}]: mean_rate={stats_results[key].get('mean_rate')}")
            except Exception as e:
                logger.debug(f"  统计计算 [{key}] 跳过: {e}")

    if stats_results:
        results["_statistics"] = stats_results
        logger.info(f"  完成 {len(stats_results)} 张表的统计计算")

    logger.info("[4/4] 生成报告...")
    generator = ReportGenerator()
    report_md = generator.generate(intent, results)
    logger.info(f"  报告长度: {len(report_md)} 字符")

    md_path = OUTPUT_DIR / "report.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(report_md, encoding="utf-8")
    logger.info(f"  Markdown 已保存: {md_path}")

    if args.docx:
        logger.info("[5/5] 生成 Word 文档...")
        template = ReportTemplate()
        docx_path = template.build_report_with_charts(report_md, results, args.name)
        logger.info(f"  Word 文档已保存: {docx_path}")

    logger.info("报告生成完成!")


def cmd_convert_to_docx(args):
    from src.report.converter import markdown_to_docx
    md_text = Path(args.input).read_text(encoding="utf-8")
    output = Path(args.output) if args.output else Path(args.input).with_suffix(".docx")
    result = markdown_to_docx(md_text, output)
    logger.info(f"Word 文档已保存: {result}")


def cmd_explore(args):
    loader = DataLoader()
    available = loader.list_available()
    logger.info(f"可用数据表 ({len(available)}):")
    for key in available:
        try:
            df = loader.load(key)
            logger.info(f"  {key}: {df.shape[0]} 行 x {df.shape[1]} 列")
        except Exception as e:
            logger.error(f"  {key}: 加载失败 - {e}")


def main():
    parser = argparse.ArgumentParser(description="Data-to-Text 食品安全抽检报告自动生成系统")
    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    training_parser = subparsers.add_parser("training", help="训练数据管理")
    training_sub = training_parser.add_subparsers(dest="training_action")

    train_stats = training_sub.add_parser("stats", help="查看训练数据统计")
    train_stats.add_argument("-f", "--file", default="training_data.jsonl", help="训练数据文件")
    train_stats.set_defaults(func=cmd_training_stats)

    train_import = training_sub.add_parser("import", help="导入真实训练样本")
    train_import.add_argument("input", help="输入文件路径 (JSONL或CSV)")
    train_import.add_argument("--output-csv", default=None, help="CSV配对模式的输出CSV")
    train_import.add_argument("--output-name", default="training_data.jsonl", help="输出文件名")
    train_import.add_argument("--mode", default="a", choices=["w", "a"],
                              help="写入模式: w=覆盖, a=追加(默认)")
    train_import.set_defaults(func=cmd_training_import)

    train_export = training_sub.add_parser("export", help="导出为微调格式")
    train_export.add_argument("-i", "--input", default="training_data.jsonl", help="输入文件")
    train_export.add_argument("-o", "--output", default="training_data_formatted.jsonl", help="输出文件")
    train_export.set_defaults(func=cmd_training_export)

    evaluate_parser = subparsers.add_parser("evaluate", help="评估模型报告质量")
    evaluate_parser.add_argument("input", help="测试数据文件 (JSONL)")
    evaluate_parser.add_argument("-o", "--output", default=None, help="评估结果输出文件 (JSON)")
    evaluate_parser.add_argument("--model", default=None, help="模型名称")
    evaluate_parser.add_argument("--host", default=None, help="API地址")
    evaluate_parser.add_argument("--backend", default=None, choices=["ollama", "vllm"],
                                  help="推理后端: ollama(默认) 或 vllm")
    evaluate_parser.set_defaults(func=cmd_evaluate)

    generate_parser = subparsers.add_parser("generate", help="生成抽检报告")
    generate_parser.add_argument("prompt", help="报告需求描述")
    generate_parser.add_argument("--docx", action=argparse.BooleanOptionalAction, default=True, help="生成Word文档")
    generate_parser.add_argument("--name", default="抽检分析报告", help="报告名称")
    generate_parser.set_defaults(func=cmd_generate_report)

    convert_parser = subparsers.add_parser("convert", help="Markdown转Word")
    convert_parser.add_argument("input", help="输入的Markdown文件")
    convert_parser.add_argument("-o", "--output", default=None, help="输出的Word文件")
    convert_parser.set_defaults(func=cmd_convert_to_docx)

    explore_parser = subparsers.add_parser("explore", help="查看可用数据表")
    explore_parser.set_defaults(func=cmd_explore)

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        return

    args.func(args)


if __name__ == "__main__":
    main()
