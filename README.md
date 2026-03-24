# MachineLearning
本仓库用以保存机器学习项目相关子模块的内容.

1. Knn项目包含的内容为CubeRecognition,即AI识别魔方状态，各部分划分如下：
* ColorClassify: 包含输出RGB识别的代码，把测试数据按照3阶魔方的排布现实
* GetCubeColor: 通过摄像头获取图片并圈出9个色块区域，提取出RGB值。本文件夹还包含一些测试内容
* visualization: 生成knn算法过程的可视化图片的代码，3D图形更加直观
* RubikCubePicture: get_color_from_picture识别的图片材料

2. k_means项目包含对学生物理、历史成绩的选科聚类
* subject: 包含直接调用sci-kit learn库的简单聚类实现
* visualization: 包含使用plt库对结果进行散点图展示

3. facerecognition项目包含对人脸的检测和人脸特征点的识别实现

4. neuralnetworks包含对手写数字识别的代码实现，源代码和数据集来自于“深入浅出神经网络与深度学习”