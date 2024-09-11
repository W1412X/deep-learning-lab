在 PyTorch 中，张量（Tensor）是基本的数据结构，它类似于 NumPy 的数组。以下是 PyTorch 中常用的张量操作，按类别分组：

### 创建张量
- `torch.tensor(data)`：从数据创建张量。
- `torch.empty(size)`：创建一个未初始化的张量。
- `torch.zeros(size)`：创建一个全为零的张量。
- `torch.ones(size)`：创建一个全为一的张量。
- `torch.arange(start, end, step)`：创建一个均匀间隔的张量。
- `torch.linspace(start, end, steps)`：创建一个在指定范围内均匀分布的张量。
- `torch.rand(size)`：创建一个在 [0, 1) 区间内均匀分布的随机张量。
- `torch.randn(size)`：创建一个从标准正态分布（均值为 0，标准差为 1）中采样的张量。
- `torch.eye(n)`：创建一个 n x n 的单位矩阵。
- `torch.diag(input)`：从输入张量创建对角矩阵。
- `torch.from_numpy(ndarray)`：从 NumPy 数组创建张量。
- `torch.as_tensor(data)`：从数据创建张量，保持数据的原始类型。

### 张量操作
- `tensor.size()` 或 `tensor.shape`：获取张量的形状。
- `tensor.dim()`：获取张量的维度数量。
- `tensor.reshape(new_shape)`：调整张量的形状。
- `tensor.view(new_shape)`：返回一个具有相同数据但形状不同的张量。
- `tensor.transpose(dim0, dim1)`：交换指定维度。
- `tensor.permute(*dims)`：重新排列维度。
- `tensor.squeeze(dim)`：删除指定维度大小为 1 的维度。
- `tensor.unsqueeze(dim)`：在指定位置插入一个大小为 1 的维度。
- `tensor.flatten(start_dim, end_dim)`：展平指定维度范围的张量。
- `tensor.expand(*sizes)`：扩展张量以匹配指定的大小。

### 运算操作
- `tensor.add(tensor2)`：逐元素加法。
- `tensor.sub(tensor2)`：逐元素减法。
- `tensor.mul(tensor2)`：逐元素乘法。
- `tensor.div(tensor2)`：逐元素除法。
- `tensor.matmul(tensor2)`：矩阵乘法。
- `tensor.sum(dim)`：在指定维度上求和。
- `tensor.mean(dim)`：在指定维度上求均值。
- `tensor.max(dim)`：在指定维度上找最大值。
- `tensor.min(dim)`：在指定维度上找最小值。
- `tensor.clamp(min, max)`：限制张量的值在指定范围内。
- `tensor.abs()`：计算张量中每个元素的绝对值。

### 张量的比较操作
- `tensor.eq(tensor2)`：逐元素等于比较。
- `tensor.ne(tensor2)`：逐元素不等于比较。
- `tensor.gt(tensor2)`：逐元素大于比较。
- `tensor.lt(tensor2)`：逐元素小于比较。
- `tensor.ge(tensor2)`：逐元素大于等于比较。
- `tensor.le(tensor2)`：逐元素小于等于比较。

### 张量的索引和切片
- `tensor[index]`：索引单个元素或切片。
- `tensor[start:end]`：对张量进行切片。
- `tensor[index1, index2]`：多维张量的索引。

### 张量的变换
- `tensor.flip(dims)`：沿指定维度翻转张量。
- `tensor.roll(shifts, dims)`：沿指定维度滚动张量的元素。
- `tensor.index_select(dim, index)`：根据指定的索引选择张量的部分数据。

### 其他操作
- `tensor.to(device)`：将张量移动到指定设备（如 CPU 或 GPU）。
- `tensor.cpu()`：将张量移回 CPU。
- `tensor.cuda()`：将张量移动到 GPU（如果可用）。
- `tensor.item()`：将只有一个元素的张量转换为 Python 标量。
- `tensor.numpy()`：将张量转换为 NumPy 数组。
- `tensor.clone()`：克隆张量的内容。