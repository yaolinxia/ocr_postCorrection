# ocr post-correction

# 问题备注

对于OCR的预测结果，做进一步的矫正处理



# 研究思路

- 多模型预测出的结果，进行互补操作
- 对于预测出的结果，结合自然语言处理的方法，将单个字， 与其前后联系起来，而不是单独进行考虑

# 数据

## 训练数据

- 模型的训练数据，选取了常用字，标点符号，数字字母等共计3900个，作为字库，基于此，通过图像处理的方法，自动生成多样性的单字图片数据

## 预测数据

- 法律的卷宗pdf扫描件， 挑选分辨率较高的来做



# 采用方法









# 存在问题

