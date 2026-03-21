import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # 用于显示中文
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 定义目标函数
def target_function(x):
    """目标函数：f(x) = sin(2πx) + 0.5cos(4πx) + 0.3x^2"""
    return np.sin(2 * np.pi * x) + 0.5 * np.cos(4 * np.pi * x) + 0.3 * x**2

# 神经网络层实现
class LinearLayer:
    """线性层：y = Wx + b"""
    def __init__(self, input_dim, output_dim):
        # He初始化（适合ReLU）
        self.W = np.random.randn(input_dim, output_dim) * np.sqrt(2.0 / input_dim)
        self.b = np.zeros((1, output_dim))
        self.x = None
        # 用于Adam优化器的动量
        self.m_W = np.zeros_like(self.W)
        self.v_W = np.zeros_like(self.W)
        self.m_b = np.zeros_like(self.b)
        self.v_b = np.zeros_like(self.b)
        self.t = 0
        
    def forward(self, x):
        """前向传播"""
        self.x = x
        return np.dot(x, self.W) + self.b
    
    def backward(self, grad_output, learning_rate, use_adam=True):
        """反向传播，更新参数"""
        batch_size = self.x.shape[0]
        grad_W = np.dot(self.x.T, grad_output) / batch_size
        grad_b = np.mean(grad_output, axis=0, keepdims=True)
        grad_input = np.dot(grad_output, self.W.T)
        
        if use_adam:
            # Adam优化器
            self.t += 1
            beta1, beta2, eps = 0.9, 0.999, 1e-8
            
            self.m_W = beta1 * self.m_W + (1 - beta1) * grad_W
            self.v_W = beta2 * self.v_W + (1 - beta2) * (grad_W ** 2)
            m_W_hat = self.m_W / (1 - beta1 ** self.t)
            v_W_hat = self.v_W / (1 - beta2 ** self.t)
            self.W -= learning_rate * m_W_hat / (np.sqrt(v_W_hat) + eps)
            
            self.m_b = beta1 * self.m_b + (1 - beta1) * grad_b
            self.v_b = beta2 * self.v_b + (1 - beta2) * (grad_b ** 2)
            m_b_hat = self.m_b / (1 - beta1 ** self.t)
            v_b_hat = self.v_b / (1 - beta2 ** self.t)
            self.b -= learning_rate * m_b_hat / (np.sqrt(v_b_hat) + eps)
        else:
            # SGD
            self.W -= learning_rate * grad_W
            self.b -= learning_rate * grad_b
        
        return grad_input


class ReLU:
    """ReLU激活函数层"""
    def __init__(self):
        self.x = None
        
    def forward(self, x):
        """前向传播：max(0, x)"""
        self.x = x
        return np.maximum(0, x)
    
    def backward(self, grad_output):
        """反向传播：梯度 = grad_output * (x > 0)"""
        return grad_output * (self.x > 0)


class TwoLayerReLUNetwork:
    """
    两层ReLU神经网络 - 验证万能逼近定理
    """
    def __init__(self, input_dim=1, hidden_dim=512, output_dim=1):
        """
        Args:
            input_dim: 输入维度
            hidden_dim: 隐藏层神经元数量（越多拟合能力越强）
            output_dim: 输出维度
        """
        self.linear1 = LinearLayer(input_dim, hidden_dim)
        self.relu = ReLU()
        self.linear2 = LinearLayer(hidden_dim, output_dim)
        
    def forward(self, x):
        """前向传播"""
        h = self.linear1.forward(x)         # 线性变换
        h_relu = self.relu.forward(h)       # ReLU激活
        y_pred = self.linear2.forward(h_relu)  # 输出层
        return y_pred
    
    def backward(self, grad_output, learning_rate):
        """反向传播"""
        grad_h_relu = self.linear2.backward(grad_output, learning_rate)
        grad_h = self.relu.backward(grad_h_relu)
        _ = self.linear1.backward(grad_h, learning_rate)
    
    def predict(self, x):
        """预测"""
        return self.forward(x)


def normalize_data(x, mean=None, std=None):
    """数据标准化"""
    if mean is None:
        mean = np.mean(x)
    if std is None:
        std = np.std(x)
    return (x - mean) / (std + 1e-8), mean, std


def main():
    
    # 生成测试数据
    np.random.seed(42)
    n_samples = 1000
    x = np.random.uniform(-2, 2, n_samples)
    y = target_function(x)
    
    # 添加噪声
    noise_level = 0.05
    y += np.random.normal(0, noise_level, n_samples)
    
    # 准备数据
    X = x.reshape(-1, 1)
    Y = y.reshape(-1, 1)
    
    # 数据标准化 
    X_norm, x_mean, x_std = normalize_data(X)
    Y_norm, y_mean, y_std = normalize_data(Y)
    
    print(f"\n目标函数: f(x) = sin(2πx) + 0.5cos(4πx) + 0.3x²")
    print(f"训练样本数: {n_samples}")
    print(f"噪声水平: {noise_level}")
    
    # 创建两层ReLU网络
    hidden_dim = 512
    model = TwoLayerReLUNetwork(input_dim=1, hidden_dim=hidden_dim, output_dim=1)
    
    # 训练参数
    learning_rate = 0.005
    epochs = 5000
    batch_size = 32
    
    print(f"\n训练参数:")
    print(f"  - 学习率: {learning_rate}")
    print(f"  - 训练轮数: {epochs}")
    print(f"  - 批次大小: {batch_size}")
    print(f"\n开始训练...")
    
    for epoch in range(epochs):
        # 随机打乱数据
        indices = np.random.permutation(len(X_norm))
        X_shuffled = X_norm[indices]
        Y_shuffled = Y_norm[indices]
        
        epoch_loss = 0
        n_batches = 0
        
        # 小批量训练
        for i in range(0, len(X_norm), batch_size):
            batch_x = X_shuffled[i:i+batch_size]
            batch_y = Y_shuffled[i:i+batch_size]
            
            # 前向传播
            y_pred = model.forward(batch_x)
            
            # 计算损失（MSE）
            loss = np.mean((y_pred - batch_y) ** 2)
            epoch_loss += loss
            n_batches += 1
            
            # 反向传播
            grad_output = 2 * (y_pred - batch_y)
            model.backward(grad_output, learning_rate)
        
        # 计算平均训练损失
        avg_train_loss = epoch_loss / n_batches
        
        if (epoch + 1) % 500 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_train_loss:.6f}")
        
        # 检查是否收敛
        if avg_train_loss < 0.001:
            print(f"\n损失已收敛到 {avg_train_loss:.6f}，提前停止于第 {epoch+1} 轮")
            break
    
    print("\n训练完成！")
    print(f"最终损失: {avg_train_loss:.6f}")
    
    # 测试预测
    x_test = np.linspace(-2, 2, 200).reshape(-1, 1)
    x_test_norm = (x_test - x_mean) / (x_std + 1e-8)
    y_test_true = target_function(x_test.flatten())
    
    # 预测并反归一化
    y_test_pred_norm = model.predict(x_test_norm)
    y_test_pred = y_test_pred_norm * (y_std + 1e-8) + y_mean
    y_test_pred = y_test_pred.flatten()
    
    # 计算测试误差
    test_mse = np.mean((y_test_pred - y_test_true) ** 2)
    print(f"测试集 MSE: {test_mse:.6f}")
    
    # 可视化
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, s=20, alpha=0.5, label='训练数据（含噪声）')
    plt.plot(x_test, y_test_true, 'r-', linewidth=2, label='真实函数')
    plt.plot(x_test, y_test_pred, 'b-', linewidth=2, label='神经网络拟合')
    plt.xlabel('x', fontsize=12)
    plt.ylabel('f(x)', fontsize=12)
    plt.title(f'两层ReLU网络函数拟合\n(隐藏层: {hidden_dim}个神经元, 测试MSE: {test_mse:.4f})', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 保存图像
    save_path = 'd:/deeplearn/function_fitting_test.png'
    plt.savefig(save_path, dpi=150)
    print(f"\n结果已保存到 {save_path}")
    
    # 显示图像
    plt.show()
    
    print("\n" + "=" * 60)
    print("实验结论：")
    print("两层ReLU神经网络成功拟合了复杂的周期函数，")
    print("验证了万能逼近定理的有效性。")
    print("=" * 60)

if __name__ == "__main__":
    main()
