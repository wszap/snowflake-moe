# Snowflake MoE 实验记录

## 一、关键洞察（贯穿全程）

1. **忆点是接线控制器，不是输出加法项**（锁12后）
2. **均匀平均陷阱**：熵正则推高熵→wiring_ent=ln(8)=2.0794锁死，模型退化为昂贵的Dense层
3. **软硬分离骗局**：wiring_ent正常但connection_strength飙到16.0（topk硬选择坍缩）
4. **赢者通吃**：单一细胞垄断所有token，其余15个细胞饿死（ΔPPL≈0）
5. **反向蒸馏**：好学生（teacher）被差生（student）同化拖下水（teacher权重萎缩）
6. **单向ENT是延迟引信**：前期被ce_loss压制，后期突然爆发，直接冲破黄金区间

## 二、实验数据总览

### 锁3.7（LAMBDA_ENT=0.01）
- 端点: wiring_ent=1.88, connection_strength=1.0001
- ΔPPL mean=+0.0059, max=+0.0900, >0.1的cell: 0个
- 结论：断路（无正常电路）

### 锁3.7.1（LAMBDA_ENT=0.1）
- 端点: wiring_ent=0.0000, connection_strength=1.7969
- ΔPPL mean=+0.0453, max=+0.3995@cell1, >0.1的cell: 1个
- 结论：赢者通吃（短路）

### 锁3.7.2（LAMBDA_ENT=0.03）
- 端点: wiring_ent=0.003, connection_strength=1.09
- 中段（step7600~8200）短暂进入黄金区间（1.47→1.33）
- ΔPPL mean=+0.0060, max=+0.0572@cell15, >0.1的cell: 0个
- 结论：两段式延迟坍缩

### 锁3.8（LAMBDA_ENT=0.05 + KD=0.05 + 分层bias）
- 坍缩点：step 1800（比3.7.2更早）
- teacher（cell1）权重萎缩，student（cell0/2/3）扩张 → 反向蒸馏
- 结论：KD未阻止坍缩，反而加速

### 锁3.9（双向熵控制 + 硬偏置 + KD=0）
- 端点: wiring_ent=1.8674（全程1.86±0.01，零坍缩）
- connection_strength=1.0001（均匀）
- 逐cell强制接线ΔPPL：cell0=+36.5, cell1=+34.6, cell2=+24.4, cell3=+33.3
- 结论：接线稳定性改善，细胞已学会分工（缺一不可），但仍断路

### 锁4.0（内容寻址接线 + 双向控制）——已完成
- 核心改动：organelle_query + 内容寻址
- step 200~1600: wiring_ent 1.8569→1.8113（缓慢下降），终点 1.6635
- connection_strength 全程恒 1.0001（step1200 前即收敛），val_ppl=10.0277
- 逐cell强制接线ΔPPL：cell0=+32.89, cell1=+32.94, cell2=+29.46, cell3=+32.84（差异<4 PPL）
- 结论：内容寻址未打破断路；组织零正常电路（connection_strength 恒 1.0001），4个cell互相接近、缺一不可但未按内容分工

## 三、判据（黄金区间）

| 指标 | 目标 | 说明 |
|------|------|------|
| wiring_ent | 1.2~1.6 | 单样本自信 |
| connection_strength | 1.5~2.0 | 跨样本正常电路 |
| 逐cell ΔPPL | ≥3个cell > 0.1 | 多细胞分工成立 |
| val_ppl | ≤10.03 | 主任务未崩 |

## 四、核心发现（用于论文）

1. 在Softmax+TopK框架下，单向熵正则无法实现真正常电路
2. 模型只会在"断路"（wiring_ent=1.88）与"短路"（wiring_ent=0）之间震荡
3. 需要引入内容寻址（架构级改动）+ 双向控制（宏观约束）
4. 边缘算力（RTX 5060 8GB）可复现大模型级MoE病理
