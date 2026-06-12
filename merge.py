import pandas as pd
import argparse

parser = argparse.ArgumentParser(description="合并短视频预测结果到原始长视频级别")
parser.add_argument("--input_csv", help="输入 CSV 路径")
parser.add_argument("--output_csv", help="输出 CSV 路径")
parser.add_argument("--mode", type=int, default=1, choices=[1, 2],
                    help="选择模式: 1=优先级(有2选p2最大，有1选p1最大，都没有选p2最大), 2=直接选p2最大的")
args = parser.parse_args()

df = pd.read_csv(args.input_csv)

# 解析 video（'/' 前的部分）
df['video'] = df['video_name'].apply(lambda x: x.rsplit('/', 1)[0])

# 每条短视频的预测类别
df['pre'] = df[['p0', 'p1', 'p2']].values.argmax(axis=1)

if args.mode == 1:
    # 模式1：有2选p2最大的，有1选p1最大的，都没有选p2最大的
    # 全是 NonLesion 的视频直接认定为 0 类
    def pick_representative(group):
        if not group['video_name'].str.contains(r'(?<!Non)Lesion').any():
            # 所有 clip 都是 NonLesion，直接认定为 0 类
            row = group.iloc[0].copy()
            row['p0'], row['p1'], row['p2'], row['pre'] = 1.0, 0.0, 0.0, 0
            return row
        if (group['pre'] == 2).any():
            return group.loc[group['p2'].idxmax()]
        elif (group['pre'] == 1).any():
            return group.loc[group['p1'].idxmax()]
        else:
            return group.loc[group['p2'].idxmax()]
else:
    # 模式2：直接选p2最大的
    def pick_representative(group):
        if not group['video_name'].str.contains(r'(?<!Non)Lesion').any():
            # 所有 clip 都是 NonLesion，直接认定为 0 类
            row = group.iloc[0].copy()
            row['p0'], row['p1'], row['p2'], row['pre'] = 1.0, 0.0, 0.0, 0
            return row
        return group.loc[group['p2'].idxmax()]

result = df.groupby('video').apply(pick_representative).reset_index(level='video')
result = result[['video', 'p0', 'p1', 'p2', 'pre']]

result.to_csv(args.output_csv, index=False)
print(f"合并完成 (mode={args.mode}): {len(df)} 条短视频 → {len(result)} 个原始视频")
print(f"结果已保存至: {args.output_csv}")
print(result.head())
