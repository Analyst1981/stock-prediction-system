# 股票预测系统 (Stock Prediction System)

基于深度学习的股票预测系统，支持LSTM、GRU、Transformer模型预测桂林旅游股票(000978)未来1-8天走势，包含GPT智能解释功能。

## 🎯 项目特色

- **多模型支持**: LSTM、GRU、Transformer三种深度学习模型
- **多步预测**: 支持预测未来1-8天的股票走势
- **GPT智能分析**: 自动生成股票走势分析和投资建议
- **实时数据**: 基于MySQL数据库的实时股票数据
- **可视化界面**: Vue.js前端展示预测结果和图表
- **模型比较**: 自动选择表现最佳的模型

## 🏗️ 系统架构

### 后端架构
- **Flask RESTful API**: 提供预测接口
- **PyTorch**: 深度学习框架
- **MySQL**: 股票数据存储
- **Redis**: 缓存和任务队列

### 前端架构
- **Vue.js**: 前端框架
- **Element UI**: UI组件库
- **ECharts**: 图表可视化
- **Axios**: HTTP请求库

## 📊 支持的股票指标

- 开盘价 (Open)
- 收盘价 (Close)
- 最高价 (High)
- 最低价 (Low)
- 成交量 (Volume)
- 技术指标: MA5、RSI等

## 🚀 快速开始

### 环境要求
- Python 3.8+
- Node.js 14+
- MySQL 8.0+
- Redis 6.0+

### 安装依赖

#### 后端依赖
```bash
pip install -r requirements.txt
```

#### 前端依赖
```bash
cd frontend
npm install
```

### 配置数据库

1. 创建MySQL数据库和表
2. 配置.env文件中的数据库连接信息
3. 运行数据导入脚本

### 训练模型

```bash
# 训练LSTM模型
python train/train.py --model lstm --mode train

# 训练GRU模型
python train/train.py --model gru --mode train

# 训练Transformer模型
python train/train.py --model transformer --mode train
```

### 启动服务

#### 启动后端API
```bash
python api/api.py
# 或使用生产环境
python -m gunicorn api.api:app -b 0.0.0.0:5000
```

#### 启动前端
```bash
cd frontend
npm run serve
# 或使用生产环境
npm run build
```

### 使用预测功能

1. 打开前端界面: http://localhost:8080
2. 选择预测天数(1-30天)
3. 选择预测模型(LSTM/GRU/Transformer)
4. 查看预测结果和GPT分析

## 📁 项目结构

```
stock-prediction-system/
├── api/                    # Flask RESTful API
├── backtester/            # 回测系统
├── configs/               # 配置文件
├── data/                  # 数据处理
├── features/              # 特征工程
├── frontend/              # Vue.js前端
├── models/                # 深度学习模型
├── train/                 # 训练脚本
├── utils/                 # 工具函数
├── requirements.txt       # Python依赖
└── README.md             # 项目说明
```

## 🔧 核心功能模块

### 1. 数据收集与预处理
- 从MySQL数据库获取股票历史数据
- 技术指标计算(MA、RSI等)
- 数据清洗和异常值处理
- 特征归一化和序列生成

### 2. 模型训练
- **LSTM模型**: 长短期记忆网络
- **GRU模型**: 门控循环单元
- **Transformer模型**: 注意力机制
- 自定义损失函数和约束条件

### 3. GPT智能分析
- 基于股票数据的文本生成
- 自动生成投资建议
- 每日走势分析说明

### 4. 后端API服务
- RESTful API接口
- 模型选择和加载
- 实时预测服务
- 错误处理和日志记录

### 5. 前端界面
- 响应式设计
- 实时图表展示
- 模型性能比较
- 用户交互体验

## 📈 模型性能

系统在验证集上的平均表现:
- **LSTM**: MAE ≈ 0.85, RMSE ≈ 1.12
- **GRU**: MAE ≈ 0.82, RMSE ≈ 1.08
- **Transformer**: MAE ≈ 0.79, RMSE ≈ 1.05

## 🤝 贡献指南

欢迎提交Issue和Pull Request来改进项目！

## 📄 许可证

MIT License - 详见LICENSE文件

## 📞 联系方式

如有问题或建议，请通过GitHub Issues联系。

---

**注意**: 本项目仅供学习和研究使用，不构成投资建议。股市有风险，投资需谨慎。