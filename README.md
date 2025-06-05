# 项目名称
  
> 基于扩散模型实现胸部X光医学影像的生成。

### 具体改动
#### 1. 硬件配置与环境依赖
- Python == 3.10  
- PyTorch == 2.1.2
- CUDA 版本为 11.8
- GPU 为 NVIDIA RTX 4090（24GB）单卡
- 库版本变动：将原 gradio \== 5.0.0 改为 5.25.0 以适配实验环境，经验证可正常运行。
#### 2. 代码文件改动
1. 在 ./iddm/tools 下新增了 enhance_xray.py 文件，用于对生成图像进行视觉效果增强。

2. 更改了 ./iddm/model/trainers 下的 dm.py 文件：
	- 更新了 def train_in_iter() 函数，实现了梯度累计（步数为4）；
	- 记录了每轮每步的 Loss 损失值，新增了def generate_loss_curves() 函数用来生成训练损失曲线图。
### 运行
#### 训练
- 确保数据集已放入./datasets/X-Ray，可通过 https://pan.baidu.com/s/1RTaBfaCu-JgR3Rpq_TwBKg?pwd=xgni 获取训练数据集X-Ray.zip（或自定义数据集）
```bash
cd 项目名

# 安装依赖

pip install -r requirements.txt # 或使用 pip install -e .

# 运行 train.py
# 详细训练及推理参数见 train.py 与 generate.py 中的默认参数设置 或 README_Original.md

python train.py --epochs 300 --batch_size 1 --image_size 128 --result_path ./results
```
#### 推理生成
- 确保模型文件已放入./results/X-Ray_gen，可通过 https://pan.baidu.com/s/1MGSHymGGAqXcGopSAZWbFQ?pwd=mr45 获取X-Ray_gen.zip
```bash
# 运行 generate.py 生成
# 每次生成一类图像

python generate.py --class_name 0 --image_size 256 --weight_path ./results/X-Ray_gen/ckpt_last.pt --result_path ./results/X-Ray_gen/Emphysema
python generate.py --class_name 1 --image_size 256 --weight_path ./results/X-Ray_gen/ckpt_last.pt --result_path ./results/X-Ray_gen/Fibrosis

# ......依次生成所有类
```
#### 结果评估
- 需要手动修改 ./iddm/tools/FID_calculator.py 中的文件路径
```bash
# 计算FID分数

python ./iddm/tools/FID_calculator.py

# 视觉增强（同样需要修改文件路径）

python ./iddm/tools/enhance_xray.py
```
### 实验记录

- 训练时间：在默认参数与上述配置下，训练一轮需 10 分 50 秒，训练 300 轮全部用时约 54 小时。
- 推理时间：（num_images == 4 ）生成 4 张 128 * 128 图像用时约 29 秒。
- FID分数（训练 image_size 128 + epoch 300 + 生成数量 1000 张）：
	- Emphysema（肺气肿）：145.2676981951077
	- Fibrosis（肺纤维化）：143.92358086379602
	- Nodule_Mass（肺结节）：140.38330557827024
	- Pleural_Thickening（胸膜增厚）：115.03021424543965

### 致谢
> 本项目基于[IDDM: Integrated-Design-Diffusion-Model](https://github.com/chairc/Integrated-Design-Diffusion-Model)fork并改进而来。感谢原作者[chairc](https://github.com/chairc)提供的开源代码。
