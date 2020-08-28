# 简介：

这是一个基于 stylegan(2)制作的 web 人脸编辑应用

# 功能：

1. 对图像进行人脸提取，得到 1024x1024 的以人脸为主的图像
2. 对人脸图像进行编码，得到基于模型空间的潜向量
3. 对潜向量进行属性编辑，生成编辑后的人脸图像

# 展示：
<div align="center">
    <img src="https://github.com/OpenOceanCold/gif_pic/raw/master/age.gif" width = "200" height = "200" />
    <img src="https://github.com/OpenOceanCold/gif_pic/raw/master/beauty.gif" width = "200" height = "200" />
    <img src="https://github.com/OpenOceanCold/gif_pic/raw/master/gender.gif" width = "200" height = "200" />
    <img src="https://github.com/OpenOceanCold/gif_pic/raw/master/height.gif" width = "200" height = "200" />
</div>
<div align="center">
    <img src="https://github.com/OpenOceanCold/gif_pic/raw/master/width.gif" width = "200" height = "200" />
    <img src="https://github.com/OpenOceanCold/gif_pic/raw/master/smile.gif" width = "200" height = "200" />
    <img src="https://github.com/OpenOceanCold/gif_pic/raw/master/horizontal.gif" width = "200" height = "200" />
    <img src="https://github.com/OpenOceanCold/gif_pic/raw/master/vertical.gif" width = "200" height = "200" />
</div>








# 环境配置：

## 官方要求

* Both Linux and Windows are supported. Linux is recommended for performance and compatibility reasons.
* 64-bit Python 3.6 installation. We recommend Anaconda3 with numpy 1.14.3 or newer.
* TensorFlow 1.14 or 1.15 with GPU support. The code does not support TensorFlow 2.0.
* On Windows, you need to use TensorFlow 1.14 — TensorFlow 1.15 will not work.
* One or more high-end NVIDIA GPUs, NVIDIA drivers, CUDA 10.0 toolkit and cuDNN 7.5. To reproduce the results reported in the paper, you need an NVIDIA GPU with at least 16 GB of DRAM.
* Docker users: use the provided Dockerfile to build an image with the required library dependencies.
* On Windows, the compilation requires Microsoft Visual Studio to be in PATH. We recommend installing Visual Studio Community Edition and adding into PATH using "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat".
## 我的测试环境配置为：


* Win10，1060max-q，
* Python 3.7.3 (default, Mar 27 2019, 17:13:21) [MSC v.1915 64 bit (AMD64)] :: Anaconda, Inc. on win32，
* CUDAtoolkit 10.0.130，CuDNN 7.6.0，tensorflow-gpu 1.14.0，VS2019，flask 1.1.1。

# 使用方法：


1. **下载并解压本项目，下载预训练好的 network，放到项目中的 networks 文件夹下。**

模特：[https://drive.google.com/file/d/1--kh2Em5U1qh-H7Lin9FzppkZCQ18c4W/view?usp=drivesdk](https://drive.google.com/file/d/1--kh2Em5U1qh-H7Lin9FzppkZCQ18c4W/view?usp=drivesdk)

明星：[https://drive.google.com/file/d/1-04v78_pI59M0IvhcKxsm3YhK2-plnbj/view?usp=drivesdk](https://drive.google.com/file/d/1-04v78_pI59M0IvhcKxsm3YhK2-plnbj/view?usp=drivesdk)

网红：[https://drive.google.com/file/d/1-35jUa-Y0kfda-oQgdrda4ys-UFtlPa_/view?usp=drivesdk](https://drive.google.com/file/d/1-35jUa-Y0kfda-oQgdrda4ys-UFtlPa_/view?usp=drivesdk)

黄种人：[https://drive.google.com/file/d/1-3XU6KzIVywFoKXx2zG1hW8mH4OYpyO9/view?usp=drivesdk](https://drive.google.com/file/d/1-3XU6KzIVywFoKXx2zG1hW8mH4OYpyO9/view?usp=drivesdk)

官方欧美人：[https://drive.google.com/open?id=1QHc-yF5C3DChRwSdZKcx1w6K8JvSxQi7](https://drive.google.com/open?id=1QHc-yF5C3DChRwSdZKcx1w6K8JvSxQi7)

**C:/Program Files (x86)/Microsoft Visual Studio/2019/Community/VC/Tools/MSVC/ 目录下有一个版本号的文件夹，将版本****号****替换****到****dnnlib/tflib/custom_ops.py 29****行的对应版本号14.26.28801是我的vs2019的msvc的版本号**

![图片](https://uploader.shimo.im/f/zCB2Tfh0EcX1vF0R.png!thumbnail)

2. **在终端中，进入本项目路径，运行 flask 项目**
```python
python html/routes.py
```
![图片](https://uploader.shimo.im/f/qd7DfvpLa4972AuD.png!thumbnail)

3. **将地址复制到浏览器**[http://127.0.0.1:8080/](http://127.0.0.1:8080/)

![图片](https://uploader.shimo.im/f/r5EZTmYEjpO6iUEk.png!thumbnail)

**以下所有路径操作注意 “ \ ” 与 “ / ” ，使用 “ / ”**

4. **第一步提取人脸，将要处理的图片放到 html/static/ 目录下，再将路径输入到**源图片路径 中，点击 提取人脸。等待 1-2 分钟，提取后的人脸将显示在页面右侧。

<div align="center"><img src="https://uploader.shimo.im/f/pVqaJXF0Xn6bGcRm.png!thumbnail" /></div>

5. **第二步进行人脸编码，选择使用的模型，最好与人脸所属类型相同，将刚才提**取的人脸图片路径（同在 static 目录下的 png 文件）输入到 人脸图片路径 中，点击 人脸编码 。等待 5-6 分钟，进程可在终端中观看。编码后再使用该模型生成的人脸图片将显示在页面右侧。

<div align="center"><img src="https://uploader.shimo.im/f/82MIyFytHdpvfxT9.png!thumbnail" /></div>

![图片](https://uploader.shimo.im/f/Uoo6T85sQ8vH4en4.png!thumbnail)


6. **第三步进行人脸属性编辑。选择刚才生成的人脸编码（同在 static 目录下**的.npy 文件），路径输入到 人脸编码路径 中，在下面选择使用的模型（可使用与编码不同的模型，但效果不一定好），滑块条可以选择共 7 个属性的参数。提供正反两个方向调整。建议每次调整尽可能少的属性，调整值也不要太大。点击 生成图片 ，等待 1 分钟左右即可在页面右侧查看到生成的人脸。

<div align="center"><img src="https://uploader.shimo.im/f/dOShdRusNW9A521O.png!thumbnail" /></div>

![图片](https://uploader.shimo.im/f/LcKZgdmPJcsO0xXE.png!thumbnail)

**使用过程图片：源图片，人脸提取图片，编码后生成的图片，属性调整后的图片**



7. **终端中 ctrl+c 即可结束 flask 项目**

## 参考：

**stylegan 官方：https://github.com/NVlabs/stylegan**

**stylegan2 官方：https://github.com/NVlabs/stylegan**

**stylegan 编码：https://github.com/rolux/stylegan2encoder**

**stylegan2 编辑器与训练模型：https://github.com/a312863063/generators-with-stylegan2**

**参考资料：http://www.seeprettyface.com/**