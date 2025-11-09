# Transformer 作业项目说明

本仓库实现了一个用于英文到德文翻译的小规模 Transformer，并完成完整的训练与验证流程。仓库内容满足课程作业要求：提供源代码、运行脚本、训练曲线日志与 README 说明。

## 1. 项目结构
- `src/`：核心源码，包含 `Transformer` 框架、`Encoder/Decoder`、多头注意力、前馈网络、位置编码、数据加载与训练脚本等模块。
- `results/fast_run/`：最终实验结果（配置、最优模型权重、TensorBoard 日志、指标 JSON）。
- `requirements.txt`：运行所需 Python 包。
- `IWSLT2017/`：本地化的 IWSLT 2017 EN-DE 数据集（需提前准备）。
- `opus_mt_en_de_tokenizer/`：与模型配套的 SentencePiece 分词器。

## 2. 环境与依赖
建议使用 Python 3.9+。安装依赖：

```bash
pip install -r requirements.txt
```

如需 GPU 训练，请确保已正确安装支持 CUDA 的 PyTorch 版本。

## 3. 数据与分词器准备
1. 在项目根目录下放置 `IWSLT2017/` 文件夹，包含：
   - `train.tags.en-de.en`、`train.tags.en-de.de`
   - `IWSLT17.TED.dev2010.en-de.en.xml`、`IWSLT17.TED.dev2010.en-de.de.xml`
   - `IWSLT17.TED.tst2010.en-de.en.xml`、`IWSLT17.TED.tst2010.en-de.de.xml`
2. 将分词模型放在 `opus_mt_en_de_tokenizer/`，需包含 `config.json`、`tokenizer_config.json`、`vocab.json`、`source.spm`、`target.spm` 等文件。

所有路径可通过命令行参数显式指定，默认使用上述相对路径。

## 4. 训练与验证
项目提供 `src/train.py` 作为统一入口，支持完整训练、学习率调度、梯度裁剪与 TensorBoard 记录。

### 4.1 复现实验命令
以下命令可复现 `results/fast_run/` 中的最终实验（如使用 PowerShell，可先激活对应虚拟环境）：

```bash
python -m src.train ^
  --d_model 128 ^
  --num_layers 2 ^
  --num_heads 4 ^
  --d_ff 512 ^
  --max_len 128 ^
  --batch_size 64 ^
  --lr 3e-4 ^
  --num_epochs 20 ^
  --lr_scheduler cosine ^
  --warmup_steps 2000 ^
  --tokenizer ./opus_mt_en_de_tokenizer ^
  --data_dir ./IWSLT2017 ^
  --save_dir results/fast_run
```

在类 Unix 终端中，可将 `^` 替换为 `\` 或写成单行。

### 4.2 仅评估最优模型
训练完成后，使用保存的权重进行评估：

```bash
python -m src.train ^
  --eval_only ^
  --tokenizer ./opus_mt_en_de_tokenizer ^
  --data_dir ./IWSLT2017 ^
  --save_dir results/fast_run ^
  --checkpoint_path results/fast_run/best_model.pt
```

添加 `--disable_bleu` 参数可跳过 BLEU 计算。

## 5. 结果与日志
`results/fast_run/results.json` 汇总了训练/验证损失与测试集指标：

```text
test_loss = 1.6018
test_perplexity = 4.9620
```

训练与验证损失曲线可通过 TensorBoard 查看：

```bash
tensorboard --logdir results/fast_run/logs
```

## 6. 代码亮点
- 手工实现的多头自注意力、残差连接与 LayerNorm，封装在 `src/attention.py`、`src/encoder.py`、`src/decoder.py`。
- 支持正余弦位置编码与可选的学习型位置编码（`--use_learned_pos`）。
- 数据管线基于本地 IWSLT 2017 平行语料，`src/data.py` 提供解析与 `DataLoader` 构建。
- 训练脚本集成梯度裁剪、`AdamW` 优化、两种学习率调度器（Plateau / Cosine），并可选计算 BLEU。

## 7. 常见问题
- **找不到数据文件**：检查 `IWSLT2017/` 中的文件名与路径是否正确，可通过 `--data_dir` 指向自定义目录。
- **Tokenizer 未找到**：确认 `--tokenizer` 指向包含 `tokenizer_config.json` 和 `vocab.json` 的目录。
- **CPU 训练过慢**：在支持 CUDA 的机器上重新安装 GPU 版 PyTorch，并确保驱动与 CUDA 版本匹配。

如需进一步实验（学习率调整、梯度截断阈值、更深的 Encoder/Decoder 层数等），可直接修改命令行参数或 `config.json`。


