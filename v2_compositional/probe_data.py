# -*- coding: utf-8 -*-
"""
probe_data.py —— 在【服务器】上跑，诊断 TinyStories 数据格式并给出接入参数

用法：
    python probe_data.py /root/private_data/snowflake/tinystories_100mb.txt

输出：
  1. 文件大小 / 行数 / 字符数
  2. 字符集大小（判断是否字符级）
  3. 按空格分词后的词表大小（判断词级规模）
  4. 建议的 vocab_size、seq_len、dataset 规模
  5. 直接给出可粘贴进 train_lock60.py 的 load_data() 代码

★ 关键提醒：vocab_size 决定 embedding 参数量
    字符级 (vocab≈100) : embed = 100 × 128 = 12.8k   可忽略
    词级   (vocab≈50k) : embed = 50k × 128 = 6.4M    ⇒ 会淹没 3.6M 的主体模型！

    对 tiny 档（主体 ~3.6M），词级 embedding 会成为参数主体，
    "拼专家 vs 标准 MoE" 的对比会被稀释。这点必须在论文里说明。
"""
import collections
import os
import sys


def main(path):
    if not os.path.exists(path):
        print(f"[FATAL] 找不到 {path}")
        sys.exit(1)

    nbytes = os.path.getsize(path)
    print("=" * 88)
    print(f"[文件] {path}")
    print("=" * 88)
    print(f"  字节数: {nbytes:,}  ({nbytes/1024**2:.1f} MB)")

    # 只读前 8MB 做统计，避免大文件全读
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        head = f.read(8 * 1024 * 1024)

    print(f"  采样: 前 {len(head):,} 字符")

    # ---- 字符级 ----
    chars = collections.Counter(head)
    vocab_char = len(chars)
    print("\n" + "-" * 88)
    print("【1】字符级")
    print("-" * 88)
    print(f"  不同字符数: {vocab_char}")
    top = chars.most_common(12)
    print(f"  最常见: {[(repr(c), n) for c, n in top[:8]]}")
    printable = sum(1 for c in chars if c.isprintable() or c in "\n\t ")
    print(f"  可打印字符占比: {printable/max(vocab_char,1):.1%}")

    # ---- 词级 ----
    words = head.split()
    vocab_word = len(set(words))
    print("\n" + "-" * 88)
    print("【2】词级（按空白分词）")
    print("-" * 88)
    print(f"  总词元数(采样): {len(words):,}")
    print(f"  不同词数: {vocab_word:,}")
    print(f"  平均每词字符: {len(head)/max(len(words),1):.2f}")

    # ---- 结构 ----
    lines = head.split("\n")
    print("\n" + "-" * 88)
    print("【3】结构")
    print("-" * 88)
    print(f"  行数: {len(lines):,}")
    lens = [len(l) for l in lines[:5000]]
    print(f"  行长: 平均 {sum(lens)/max(len(lens),1):.0f}, "
          f"最大 {max(lens) if lens else 0}")
    print(f"  首行前 200 字符:\n    {lines[0][:200]!r}")
    if len(lines) > 1:
        print(f"  第二行前 200 字符:\n    {lines[1][:200]!r}")

    # ---- 判定与建议 ----
    print("\n" + "=" * 88)
    print("【4】判定与建议")
    print("=" * 88)
    est_tokens_char = nbytes
    est_tokens_word = nbytes / (len(head) / max(len(words), 1))

    print(f"  若字符级: 总 token ≈ {est_tokens_char:,}, vocab ≈ {vocab_char}")
    print(f"  若词级  : 总 token ≈ {est_tokens_word:,.0f}, vocab ≈ {vocab_word:,}")

    print("\n  ★ embedding 参数量对比（d=128）:")
    print(f"    字符级 {vocab_char:>7} × 128 = {vocab_char*128:>12,}")
    print(f"    词级   {vocab_word:>7} × 128 = {vocab_word*128:>12,}")
    print(f"    tiny 档主体模型约 3.6M")
    if vocab_word * 128 > 3.6e6:
        print("    ⚠ 词级 embedding 会超过主体模型 ⇒ 架构对比被稀释，"
              "论文需说明，或考虑 tied embedding")
    else:
        print("    ✅ 可接受")

    print("\n" + "=" * 88)
    print("【5】可直接粘贴进 train_lock60.py 的 load_data()")
    print("=" * 88)
    print(f'''
def load_data():
    """自动接入：{os.path.basename(path)}"""
    import torch
    from torch.utils.data import DataLoader
    txt = open(r"{path}", encoding="utf-8").read()
    chars = sorted(set(txt))
    stoi = {{c: i for i, c in enumerate(chars)}}
    itos = {{i: c for i, c in enumerate(chars)}}
    ids = torch.tensor([stoi[c] for c in txt], dtype=torch.long)
    V = len(chars)                      # = {vocab_char}
    n = int(len(ids) * 0.9)
    tr, va = ids[:n], ids[n:]

    def get_batch(split, bs, seq):
        src = tr if split == "train" else va
        ix = torch.randint(len(src) - seq, (bs,))
        x = torch.stack([src[i:i+seq] for i in ix])
        y = torch.stack([src[i+1:i+1+seq] for i in ix])
        return x, y

    class Iter:
        def __init__(self, split): self.split = split
        def __iter__(self): return self
        def __next__(self): return get_batch(self.split, BS, SEQ)

    return Iter("train"), Iter("val"), V
''')
    print("  注意：把 BS / SEQ 换成实际值（在 load_data 前定义，或从 args 取）")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1
         else "/root/private_data/snowflake/tinystories_100mb.txt")
