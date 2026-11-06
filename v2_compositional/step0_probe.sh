#!/bin/bash
echo "==== [1] 脚本存在性 ===="
ls -1 train_v3.py train_v4.py dense_baseline.py snowflake_B.py 2>/dev/null
echo "==== [2] train_v4 参数名 ===="
grep -oE '"--[a-z0-9-]+"' train_v4.py | sort -u
echo "==== [3] 聚合方式 (mean/sum/softmax/stack) ===="
grep -nE "\.mean\(|\.sum\(|softmax|torch\.stack|einsum" snowflake_B.py | head -30
echo "==== [4] 调制相加 (W1_base 累加) ===="
grep -nE "W1_base|W1_V|mod|delta|@ V" snowflake_B.py | head -30
echo "==== [5] scaffold-top-k 默认值 ===="
grep -nE "scaffold.top.k|scaffold_top_k" train_v4.py | head -10
