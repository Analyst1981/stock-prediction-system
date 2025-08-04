from models.gpt_tiny import TinyGPT
import torch
import os

def main():
    # 初始化GPT模型
    model_path = "D:/2100101724Graduation project/models/results/custom_gpt"
    
    if os.path.exists(model_path):
        print(f"正在从 {model_path} 加载模型...")
        gpt_model = TinyGPT(model_path)
        print("模型加载成功！")
        
        # 打印模型结构
        print("\n模型结构:")
        print(gpt_model.model)
        
        # 测试推理
        test_input = "桂林旅游股票今日走势分析"
        print(f"\n测试输入: {test_input}")
        
        try:
            output = gpt_model.generate_text(test_input, max_length=50)
            print(f"模型输出: {output}")
        except Exception as e:
            print(f"推理测试失败: {e}")
            
    else:
        print(f"模型路径不存在: {model_path}")
        print("请先训练GPT模型或检查路径设置")

if __name__ == "__main__":
    main()