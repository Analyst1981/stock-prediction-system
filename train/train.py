import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import mysql.connector
from datetime import datetime, timedelta
import os
import sys
import json
import argparse
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.lstm import LSTMModel
from models.gru import GRUModel
from models.transformer import TransformerModel
from models.gpt_tiny import TinyGPT
from configs.model_config import ModelConfig
from utils.logger import setup_logger
from utils.visualize import plot_training_results, plot_predictions
from features.preprocessor import StockDataPreprocessor

class StockTrainer:
    def __init__(self, model_type='lstm', config=None):
        self.model_type = model_type
        self.config = config or ModelConfig()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 设置日志
        self.logger = setup_logger('train', 'train/train.log')
        self.logger.info(f"使用设备: {self.device}")
        self.logger.info(f"模型类型: {model_type}")
        
    def load_data_from_mysql(self):
        """从MySQL数据库加载股票数据"""
        try:
            connection = mysql.connector.connect(
                host=self.config.DB_HOST,
                user=self.config.DB_USER,
                password=self.config.DB_PASSWORD,
                database=self.config.DB_NAME,
                port=self.config.DB_PORT
            )
            
            cursor = connection.cursor(dictionary=True)
            
            # 获取股票代码对应的数据
            query = f"""
                SELECT * FROM stock_data 
                WHERE stock_code = '{self.config.STOCK_CODE}' 
                ORDER BY date ASC
            """
            
            cursor.execute(query)
            result = cursor.fetchall()
            
            if not result:
                raise ValueError(f"未找到股票代码 {self.config.STOCK_CODE} 的数据")
            
            df = pd.DataFrame(result)
            df['date'] = pd.to_datetime(df['date'])
            
            cursor.close()
            connection.close()
            
            self.logger.info(f"成功从MySQL加载 {len(df)} 条记录")
            return df
            
        except Exception as e:
            self.logger.error(f"从MySQL加载数据失败: {e}")
            raise
    
    def prepare_features(self, df):
        """准备特征数据"""
        preprocessor = StockDataPreprocessor()
        
        # 计算技术指标
        df = preprocessor.add_technical_indicators(df)
        
        # 选择特征
        feature_columns = self.config.FEATURE_COLUMNS
        
        # 确保所有特征列都存在
        missing_cols = [col for col in feature_columns if col not in df.columns]
        if missing_cols:
            self.logger.warning(f"缺失特征列: {missing_cols}")
            feature_columns = [col for col in feature_columns if col in df.columns]
        
        # 数据清洗
        df = df.dropna(subset=feature_columns)
        
        # 特征缩放
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(df[feature_columns])
        
        return features_scaled, scaler, df
    
    def create_sequences(self, data, target, sequence_length, prediction_days):
        """创建序列数据"""
        X, y = [], []
        
        for i in range(len(data) - sequence_length - prediction_days + 1):
            X.append(data[i:i + sequence_length])
            
            # 获取未来prediction_days的收盘价
            target_values = []
            for j in range(prediction_days):
                target_idx = i + sequence_length + j
                if target_idx < len(target):
                    target_values.append(target[target_idx])
                else:
                    target_values.append(target[-1])  # 用最后一个值填充
            
            y.append(target_values)
        
        return np.array(X), np.array(y)
    
    def create_datasets(self, df):
        """创建训练和验证数据集"""
        features, scaler, df = self.prepare_features(df)
        
        # 获取目标变量（收盘价）
        close_prices = df['close'].values
        
        # 创建序列
        X, y = self.create_sequences(
            features, 
            close_prices, 
            self.config.SEQUENCE_LENGTH, 
            self.config.PREDICTION_DAYS
        )
        
        # 划分训练集和验证集
        split_idx = int(len(X) * 0.8)
        
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # 转换为PyTorch张量
        X_train = torch.FloatTensor(X_train)
        y_train = torch.FloatTensor(y_train)
        X_val = torch.FloatTensor(X_val)
        y_val = torch.FloatTensor(y_val)
        
        self.logger.info(f"训练集大小: {X_train.shape}, 验证集大小: {X_val.shape}")
        
        return X_train, y_train, X_val, y_val, scaler
    
    def create_model(self, input_size):
        """创建模型"""
        if self.model_type == 'lstm':
            model = LSTMModel(
                input_size=input_size,
                hidden_size=self.config.LSTM_HIDDEN_SIZE,
                num_layers=self.config.LSTM_NUM_LAYERS,
                output_size=self.config.PREDICTION_DAYS,
                dropout=self.config.DROPOUT
            )
        elif self.model_type == 'gru':
            model = GRUModel(
                input_size=input_size,
                hidden_size=self.config.GRU_HIDDEN_SIZE,
                num_layers=self.config.GRU_NUM_LAYERS,
                output_size=self.config.PREDICTION_DAYS,
                dropout=self.config.DROPOUT
            )
        elif self.model_type == 'transformer':
            model = TransformerModel(
                input_size=input_size,
                d_model=self.config.TRANSFORMER_D_MODEL,
                nhead=self.config.TRANSFORMER_NHEAD,
                num_layers=self.config.TRANSFORMER_NUM_LAYERS,
                output_size=self.config.PREDICTION_DAYS,
                dropout=self.config.DROPOUT
            )
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")
        
        return model.to(self.device)
    
    def train_model(self, model, X_train, y_train, X_val, y_val):
        """训练模型"""
        
        # 创建数据加载器
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=self.config.BATCH_SIZE, shuffle=True)
        
        # 定义损失函数和优化器
        criterion = nn.HuberLoss()  # 使用Huber损失，对异常值更鲁棒
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.LEARNING_RATE)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        # 训练记录
        train_losses = []
        val_losses = []
        
        self.logger.info("开始训练...")
        
        for epoch in range(self.config.EPOCHS):
            # 训练阶段
            model.train()
            train_loss = 0
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                
                # 添加价格约束损失
                price_constraint = self.calculate_price_constraint(outputs, batch_y)
                total_loss = loss + self.config.PRICE_CONSTRAINT_WEIGHT * price_constraint
                
                total_loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.GRADIENT_CLIP)
                
                optimizer.step()
                train_loss += loss.item()
            
            # 验证阶段
            val_loss = self.evaluate_model(model, X_val, y_val, criterion)
            
            train_loss /= len(train_loader)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            scheduler.step(val_loss)
            
            if epoch % 10 == 0:
                self.logger.info(f"Epoch {epoch+1}/{self.config.EPOCHS}, "
                               f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # 早停机制
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                # 保存最佳模型
                model_path = f"train/models_informations/best_{self.model_type}_{datetime.now().strftime('%Y%m%d%H%M%S')}.pth"
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'model_type': self.model_type,
                    'config': self.config.__dict__,
                    'best_val_loss': best_val_loss,
                    'epoch': epoch
                }, model_path)
                
                self.logger.info(f"保存最佳模型: {model_path}")
                
            else:
                patience_counter += 1
                if patience_counter >= self.config.PATIENCE:
                    self.logger.info(f"早停于epoch {epoch+1}")
                    break
        
        return model, train_losses, val_losses, best_val_loss
    
    def calculate_price_constraint(self, predictions, targets):
        """计算价格约束损失"""
        # 确保预测价格单调性
        diff = predictions[:, 1:] - predictions[:, :-1]
        penalty = torch.relu(-diff)  # 惩罚下降
        return penalty.mean()
    
    def evaluate_model(self, model, X_val, y_val, criterion):
        """评估模型"""
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            X_val, y_val = X_val.to(self.device), y_val.to(self.device)
            outputs = model(X_val)
            loss = criterion(outputs, y_val)
            val_loss = loss.item()
        
        return val_loss
    
    def train_gpt_model(self):
        """训练GPT模型用于文本分析"""
        self.logger.info("开始训练GPT模型...")
        
        try:
            gpt_model = TinyGPT()
            
            # 准备训练数据
            train_data = self.prepare_gpt_training_data()
            
            # 训练GPT
            gpt_model.train(
                train_data=train_data,
                output_dir="models/results/custom_gpt",
                num_epochs=3,
                batch_size=4
            )
            
            self.logger.info("GPT模型训练完成")
            
        except Exception as e:
            self.logger.error(f"GPT模型训练失败: {e}")
    
    def prepare_gpt_training_data(self):
        """准备GPT训练数据"""
        # 这里应该根据股票数据生成训练文本
        # 简化版本，实际应用中需要更复杂的文本生成
        stock_data = self.load_data_from_mysql()
        
        training_texts = []
        for i in range(len(stock_data)):
            row = stock_data.iloc[i]
            text = f"日期: {row['date']}, 股票代码: {row['stock_code']}, "
            text += f"开盘价: {row['open']}, 收盘价: {row['close']}, "
            text += f"最高价: {row['high']}, 最低价: {row['low']}, "
            text += f"成交量: {row['volume']}"
            training_texts.append(text)
        
        return training_texts
    
    def run_training(self):
        """运行完整的训练流程"""
        try:
            # 加载数据
            df = self.load_data_from_mysql()
            
            # 创建数据集
            X_train, y_train, X_val, y_val, scaler = self.create_datasets(df)
            
            # 创建模型
            model = self.create_model(X_train.shape[2])
            
            # 训练模型
            model, train_losses, val_losses, best_loss = self.train_model(
                model, X_train, y_train, X_val, y_val
            )
            
            # 保存scaler
            scaler_path = f"train/models_informations/scaler_{self.model_type}.pkl"
            import joblib
            joblib.dump(scaler, scaler_path)
            
            # 可视化训练结果
            plot_training_results(train_losses, val_losses, self.model_type)
            
            self.logger.info("训练完成！")
            
            return best_loss
            
        except Exception as e:
            self.logger.error(f"训练过程失败: {e}")
            raise

def main():
    parser = argparse.ArgumentParser(description='股票预测模型训练')
    parser.add_argument('--model', type=str, default='lstm', 
                       choices=['lstm', 'gru', 'transformer', 'gpt'],
                       help='选择模型类型')
    parser.add_argument('--mode', type=str, default='train', 
                       choices=['train', 'gpt_train'],
                       help='训练模式')
    
    args = parser.parse_args()
    
    trainer = StockTrainer(model_type=args.model)
    
    if args.mode == 'train':
        trainer.run_training()
    elif args.mode == 'gpt_train':
        trainer.train_gpt_model()

if __name__ == "__main__":
    main()