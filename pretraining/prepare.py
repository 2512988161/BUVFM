#!/usr/bin/env python3
"""
prepare.py - 一键生成训练数据 CSV、YAML 配置和启动脚本。

用法:
  python prepare.py \
    --pretrain_dataset /path/to/pretrain_data \
    --finetune_dataset_train /path/to/train \
    --finetune_dataset_val /path/to/val \
    --model vitg \
    --devices "cuda:0 cuda:1 cuda:2 cuda:3"
"""

import argparse
import os
import sys
from pathlib import Path

import yaml


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm"}

# model_name, out_layers（最后一层索引）
MODEL_CONFIGS = {
    "vitg": {"model_name": "vit_giant_xformers", "out_layers": [39]},
    "vith": {"model_name": "vit_huge",           "out_layers": [31]},
    "vitl": {"model_name": "vit_large",          "out_layers": [23]},
}


def parse_mapping(mapping_str):
    """Parse 'class_0:0,class_1:1,class_2:2' into {'class_0': '0', 'class_1': '1', ...}"""
    if not mapping_str:
        return {}
    result = {}
    for pair in mapping_str.split(","):
        pair = pair.strip()
        if ":" not in pair:
            print(f"[WARN] 忽略无效映射项: {pair}")
            continue
        key, val = pair.split(":", 1)
        result[key.strip()] = val.strip()
    return result


def find_videos(root):
    """递归查找目录下所有视频文件，返回 [(abs_path, relative_subfolder), ...]"""
    root = Path(root).resolve()
    videos = []
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if Path(f).suffix.lower() in VIDEO_EXTS:
                abs_path = os.path.join(dirpath, f)
                rel = os.path.relpath(dirpath, root)
                videos.append((abs_path, rel))
    return videos


def generate_pretrain_csv(dataset_path, mapping, output_path):
    """
    生成 pretrain.csv。
    - 有 mapping: 根据子文件夹名匹配，匹配不上的写 0
    - 无 mapping: 所有行写 0
    """
    videos = find_videos(dataset_path)
    if not videos:
        print(f"[ERROR] 在 {dataset_path} 下未找到任何视频文件")
        sys.exit(1)

    with open(output_path, "w") as f:
        for abs_path, rel_subfolder in videos:
            label = mapping.get(rel_subfolder, "0")
            f.write(f"{abs_path}::{label}\n")

    print(f"[OK] pretrain.csv: {len(videos)} 条记录 -> {output_path}")
    return len(videos)


def generate_split_csv(dataset_path, mapping, default_prefix, output_path):
    """
    生成 train.csv 或 val.csv。
    数据集必须有子文件夹结构（ImageNet-style）。
    默认映射: class_0→0, class_1→1, ...
    """
    root = Path(dataset_path).resolve()
    if not root.is_dir():
        print(f"[ERROR] 目录不存在: {dataset_path}")
        sys.exit(1)

    # 如果没有自定义映射，自动从子文件夹生成
    if not mapping:
        subdirs = sorted([d.name for d in root.iterdir() if d.is_dir()])
        mapping = {}
        for d in subdirs:
            # class_0 → 0, class_1 → 1, ...
            if d.startswith(default_prefix):
                try:
                    idx = int(d[len(default_prefix):])
                    mapping[d] = str(idx)
                except ValueError:
                    mapping[d] = d
            else:
                mapping[d] = d

    videos = []
    for subfolder, label in mapping.items():
        subfolder_path = root / subfolder
        if not subfolder_path.is_dir():
            print(f"[WARN] 子文件夹不存在，跳过: {subfolder_path}")
            continue
        for dirpath, _, filenames in os.walk(subfolder_path):
            for f in filenames:
                if Path(f).suffix.lower() in VIDEO_EXTS:
                    abs_path = os.path.join(dirpath, f)
                    videos.append((abs_path, label))

    if not videos:
        print(f"[ERROR] 在 {dataset_path} 下未找到任何视频文件")
        sys.exit(1)

    with open(output_path, "w") as f:
        for abs_path, label in videos:
            f.write(f"{abs_path}::{label}\n")

    print(f"[OK] {Path(output_path).name}: {len(videos)} 条记录 -> {output_path}")
    return len(videos)


def generate_pretrain_yaml(template_path, folder, csv_path, checkpoint_path, model_key, num_samples, output_path):
    """基于模板生成 pretrain.yaml，修改 folder / datasets / read_checkpoint / model / ipe。"""
    with open(template_path, "r") as f:
        cfg = yaml.safe_load(f)

    cfg["folder"] = str(Path(folder).resolve())
    cfg["data"]["datasets"] = [str(Path(csv_path).resolve())]
    cfg["meta"]["read_checkpoint"] = str(Path(checkpoint_path).resolve())

    # 覆盖 model 配置
    mcfg = MODEL_CONFIGS[model_key]
    cfg["model"]["model_name"] = mcfg["model_name"]

    # ipe = pretrain dataset 总数 / batch_size
    batch_size = cfg["data"]["batch_size"]
    cfg["optimization"]["ipe"] = num_samples // batch_size

    with open(output_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    print(f"[OK] pretrain.yaml -> {output_path}")


def generate_finetune_yaml(template_path, folder, pretrain_folder, train_csv, val_csv, num_classes, model_key, output_path):
    """基于模板生成 finetune.yaml，修改 folder / dataset / num_classes / checkpoint / encoder model。"""
    with open(template_path, "r") as f:
        cfg = yaml.safe_load(f)

    cfg["folder"] = str(Path(folder).resolve())
    cfg["experiment"]["data"]["dataset_train"] = str(Path(train_csv).resolve())
    cfg["experiment"]["data"]["dataset_val"] = str(Path(val_csv).resolve())
    cfg["experiment"]["data"]["num_classes"] = num_classes
    # finetune 的 checkpoint 指向 pretrain 输出的 latest.pt
    cfg["model_kwargs"]["checkpoint"] = str(Path(pretrain_folder).resolve() / "latest.pt")

    # 覆盖 encoder model_name 和 out_layers
    mcfg = MODEL_CONFIGS[model_key]
    cfg["model_kwargs"]["pretrain_kwargs"]["encoder"]["model_name"] = mcfg["model_name"]
    cfg["model_kwargs"]["wrapper_kwargs"]["out_layers"] = mcfg["out_layers"]

    with open(output_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    print(f"[OK] finetune.yaml -> {output_path}")


def generate_shell_scripts(pretrain_yaml, finetune_yaml, devices, output_dir):
    """生成 pretrain.sh 和 finetune.sh。"""
    pretrain_sh = Path(output_dir) / "pretrain.sh"
    with open(pretrain_sh, "w") as f:
        f.write("#!/bin/bash\n")
        f.write(f'python -m app.main \\\n  --fname {pretrain_yaml} \\\n  --devices {devices}\n')
    os.chmod(pretrain_sh, 0o755)
    print(f"[OK] pretrain.sh -> {pretrain_sh}")

    finetune_sh = Path(output_dir) / "finetune.sh"
    with open(finetune_sh, "w") as f:
        f.write("#!/bin/bash\n")
        f.write(f'PYTHONPATH=. python -m evals.main \\\n  --fname {finetune_yaml} \\\n  --devices {devices}\n')
    os.chmod(finetune_sh, 0o755)
    print(f"[OK] finetune.sh -> {finetune_sh}")


def main():
    parser = argparse.ArgumentParser(description="一键生成训练数据 CSV、YAML 配置和启动脚本")

    # 数据集路径
    parser.add_argument("--pretrain_dataset", required=True, help="预训练数据集根目录（递归扫描视频）")
    parser.add_argument("--finetune_dataset_train", required=True, help="微调训练集目录（ImageNet-style 子文件夹）")
    parser.add_argument("--finetune_dataset_val", required=True, help="微调验证集目录（ImageNet-style 子文件夹）")

    # 模型名
    parser.add_argument("--model", required=True, choices=["vitg", "vith", "vitl"], help="训练模型")

    # 可选参数
    parser.add_argument("--devices", default="cuda:0 cuda:1 cuda:2 cuda:3", help="GPU 设备列表")
    parser.add_argument("--pretrain_mapping", default="", help="预训练子文件夹→类别映射，如 'cls0:0,cls1:1'")
    parser.add_argument("--finetune_mapping", default='class_0:0,class_1:1,class_2:2', help="微调子文件夹→类别映射，如 'class_0:0,class_1:1'")
    parser.add_argument("--num_classes", type=int, default=3, help="微调类别数（默认 3）")
    parser.add_argument("--pretrain_csv", default="pretrain.csv", help="预训练 CSV 文件名")
    parser.add_argument("--train_csv", default="train.csv", help="训练 CSV 文件名")
    parser.add_argument("--val_csv", default="val.csv", help="验证 CSV 文件名")

    args = parser.parse_args()

    pretrain_folder = f'outputs/{args.model}/pretrain'
    finetune_folder = f'outputs/{args.model}/finetune'
    pretrain_checkpoint = f'configs/{args.model}/{args.model}.pt'
    # 定位 configs 目录（prepare.py 同级）
    script_dir = Path(__file__).resolve().parent
    configs_dir = script_dir / "configs"
    configs_dir.mkdir(exist_ok=True)
    model_configs_dir = configs_dir / args.model
    model_configs_dir.mkdir(exist_ok=True)

    # Step 1: 生成 CSV
    print("=" * 50)
    print("Step 1: 生成 CSV 文件")
    print("=" * 50)

    pretrain_csv_path = model_configs_dir / args.pretrain_csv
    pretrain_num = generate_pretrain_csv(
        args.pretrain_dataset,
        parse_mapping(args.pretrain_mapping),
        pretrain_csv_path,
    )

    train_csv_path = model_configs_dir / args.train_csv
    generate_split_csv(
        args.finetune_dataset_train,
        parse_mapping(args.finetune_mapping),
        "class_",
        train_csv_path,
    )

    val_csv_path = model_configs_dir / args.val_csv
    generate_split_csv(
        args.finetune_dataset_val,
        parse_mapping(args.finetune_mapping),
        "class_",
        val_csv_path,
    )

    # Step 2: 生成 YAML 配置
    print()
    print("=" * 50)
    print("Step 2: 生成 YAML 配置")
    print("=" * 50)

    pretrain_yaml_path = model_configs_dir / "pretrain.yaml"
    generate_pretrain_yaml(
        configs_dir / "pretrain-tample.yaml",
        pretrain_folder,
        pretrain_csv_path,
        pretrain_checkpoint,
        args.model,
        pretrain_num,
        pretrain_yaml_path,
    )

    finetune_yaml_path = model_configs_dir / "finetune.yaml"
    generate_finetune_yaml(
        configs_dir / "fintune-tample.yaml",
        finetune_folder,
        pretrain_folder,
        train_csv_path,
        val_csv_path,
        args.num_classes,
        args.model,
        finetune_yaml_path,
    )

    # Step 3: 生成启动脚本
    print()
    print("=" * 50)
    print("Step 3: 生成启动脚本")
    print("=" * 50)

    generate_shell_scripts(
        pretrain_yaml_path,
        finetune_yaml_path,
        args.devices,
        model_configs_dir,
    )

    print()
    print("=" * 50)
    print("完成！生成的文件：")
    print(f"  CSV:   {model_configs_dir}/{args.pretrain_csv}, {args.train_csv}, {args.val_csv}")
    print(f"  YAML:  {pretrain_yaml_path}, {finetune_yaml_path}")
    print(f"  Shell: {model_configs_dir}/pretrain.sh, {model_configs_dir}/finetune.sh")
    print()
    print(f"运行预训练: bash {model_configs_dir}/pretrain.sh")
    print(f"运行微调:   bash {model_configs_dir}/finetune.sh")


if __name__ == "__main__":
    main()
