# 手把手教你使用YOLOV5做电线绝缘子缺陷检测

随着社会和经济的持续发展，电力系统的投资与建设也日益加速。在电力系统中，输电线路作为电能传输的载体，是最为关键的环节之一。而绝缘子作为输电环节中的重要设备，在支撑固定导线，保障绝缘距离的方面有着重要作用。大多数高压输电线路主要架设在非城市内地区，绝缘子在输电线路中由于数量众多、跨区分布，且长期暴露在空气中，受恶劣自然环境的影响，十分容易发生故障。随着大量输电工程的快速建设，传统依靠人工巡检的模式，已经越来越难以适应高质量运维的要求。随着国网公司智能化要求的提升，无人机技术的快速应用，采取无人机智能化巡视，能够大幅度减少运维人员及时间，提升质量，因此得到快速发展。

深度学习技术的大量应用，计算机运算性能的不断提高，为无人机准确识别和定位绝缘子，实时跟踪拍摄开辟了新的解决途径。本文对输电线路中绝缘子进行识别及定位，利用深度学习技术采取基于YOLOv5 算法的目标检测手段，结合绝缘子数据集的特点，对无人机拍摄图片进行训练，实现对绝缘子精准识别和定位，大幅提升无人机巡检时对绝缘子设备准确跟踪、判定的效率，具有十分重要的应用效果。

废话不多说，咱们先看两张实际效果。

![val_batch0_pred](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/imgs/val_batch0_pred.jpg)

![](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/imgs/image-20230315155840755.png)

## <font color='red'>注意事项</font>

1. 尽量使用英文路径，避免中文路径，中文路径可能会导致代码安装错误和图片读取错误。
2. pycharm运行代码一定要注意左下角是否在虚拟环境中。
3. 库的版本很重要，使用本教程提供的代码将会事半功倍

遇到解决不了的问题可以通过<font color='red'>私信（QQ:3045834499）</font>联系我，粉丝儿朋友远程调试该项目（包含数据集和训练好的三组模型）仅需99个圆子。

## 前期准备

项目下载地址：[ YOLOV5电线绝缘子缺陷检测数据集+代码+模型+教学视频+参考论文资源-CSDN文库](https://download.csdn.net/download/ECHOSON/87579913)

### 电脑设置

大部分小伙伴使用的电脑一般都是windows系统，在windows系统下跑代码，难免会遇到各种各样的bug，首先是编码问题，为了防止代码在运行过程中，出现编码错误，我们首先要将我们电脑的语言设置为utf-8格式，具体如下。首先在搜索框搜索语言，点击这里。

![image-20230314162708618](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/imgs/image-20230314162708618.png)

找到管理语言设置。

![image-20230314162736950](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/imgs/image-20230314162736950.png)

勾选utf-8即可。

![image-20230314162809457](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/imgs/image-20230314162809457.png)

另外有的电脑初始设置的时候内存是由电脑自行分配的，很可能在运行训练代码的时候出现显存溢出的情况，为了防止该情况的出现，我们需要在高级系统设置中对虚拟内存进行设置，如下。

首先打开高级系统设置，点开性能中的设置。

![image-20230314163125410](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/imgs/image-20230314163125410.png)

在高级中找到虚拟内存的设置。

![g](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/imgs/image-20230314163246667.png)

以d盘为例，这里我们将虚拟内存设置在4G到8G之间，其余操作一样。

![image-20230314163343142](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/imgs/image-20230314163343142.png)

其他盘也设置完成之后，点击确定之后按照电脑提示重启即可。

### 相关软件安装

* Nvidia驱动安装（可选）

  首先是驱动的安装，这个小节主要是针对电脑带有Nvidia显卡的小伙伴，如果你的电脑没有Nvidia显卡，你直接跳过就可以了，你可以通过右下方的任务栏判断是是否具有这个显卡，如果是笔者这里的绿色小眼睛图标，说明你是具有Nvidia显卡的。

  ![image-20230314163712808](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/imgs/image-20230314163712808.png)

  驱动下载的地址为：[官方驱动 | NVIDIA](https://www.nvidia.cn/Download/index.aspx?lang=cn)

  注意请按照你电脑具体的型号来选择驱动文件，不清楚的可以通过设备管理器来查看你显卡的具体型号。

  ![image-20230314163843861](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/imgs/image-20230314163843861.png)

  ![image-20230314164934119](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/imgs/image-20230314164934119.png)

  下载exe文件之后，直接下一步下一步按照默认安装就好，注意这里最好按照默认目录安装，否则可能导致安装失败的情况如下，安装完毕之后重启电脑，在命令行中输入`nvidia-smi`，如果正常输出显卡的信息说明显卡驱动安装已经成功。

  ![image-20230314164556374](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/imgs/image-20230314164556374.png)

  <font color='red'>另外，这里多叭叭两句。</font>

  很多的远古教程会教你去安装cuda和cudnn，这个过程非常麻烦，并且需要你注册nvidia的账户，由于众所周知的原因，账户注册就很繁琐。其实，在新版本的驱动安装中，就已经自带最新版本的cuda，比如我上图所示的在笔者驱动安装完毕之后已经自带了12.0的cuda，cuda这个东西是向下兼容的，后续的cudnn那些我们直接通过anaconda来安装就可以了，这样真的省心很多。

* Anaconda安装

  conda是一个开源的包、环境管理器，可以用于在同一个机器上安装不同版本的软件包及其依赖，并能够在不同的环境之间切换。我强烈推荐你使用他，他的作用类似于java中的maven和我们平时使用的虚拟机，他能够保证的项目之间互相是隔离的。举个简单的例子，如果你同时有两个项目，你一个使用的pytorch1.8，一个用的是pytorch1.10，这样一个环境肯定就不够了，这个时候anaconda就派上大用场了，他可以创建两个环境，各用各的，互不影响。

  Anaconda有完全版本和Miniconda，其中完整版太臃肿了，我这边推荐使用miniconda，下载地址在：[Index of /anaconda/miniconda/ | 北京外国语大学开源软件镜像站 | BFSU Open Source Mirror](https://mirrors.bfsu.edu.cn/anaconda/miniconda/)

  下滑到最下方，点击这个版本的下载即可。

  ![image-20230314165429511](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/imgs/image-20230314165429511.png)

  同样的，下载之后安装即可，<font color='red'>注意不要安装在C盘！！！</font>，另外，添加到系统路径这个选项也请务必选上，后面使用起来会带来很多便捷，并且这里的安装位置请你一定要记得，后面我们在Pycharm中将会使用到。

  ![image-20230314165627756](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/imgs/image-20230314165627756.png)

* Pycharm安装

  OK，Anaconda安装完成之后，我们还需要一个编辑器来写代码，这里推荐大家使用Pycharm，Pycharm有专业版和社区版，社区版是免费的，专业版如果你有教育邮箱的话也可以通过教育邮箱注册账户来使用，但是社区版的功能已经够绝大多数小伙伴来用了，Pycharm的下载地址在：[Download PyCharm: Python IDE for Professional Developers by JetBrains](https://www.jetbrains.com/pycharm/download/#section=windows)

  ![image-20230314170220451](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/imgs/image-20230314170220451.png)

  注意也是安装的时候不要安装在c盘，并且安装过程中这些可选的选项要勾上。

  ![image-20230314170445393](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/imgs/image-20230314170445393.png)

  完成之后，比如我们用pycharm打开一个项目，在新版本的下方会出现命令行无法使用的情况。请使用管理员模型打开powershell。

  ![image-20230314171029040](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/imgs/image-20230314171029040.png)

  然后执行执行指令`set-ExecutionPolicy RemoteSigned`，输入Y然后enter完成即可。

  ![image-20230314171244840](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/imgs/image-20230314171244840.png)

  另外，在Pycharm的右下方是表示的是你所处的虚拟环境，这里先简单说下在pycharm中如何使用anaconda中创建的虚拟环境。

  ![image-20230314171639225](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/imgs/image-20230314171639225.png)

  ![](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/imgs/image-20230314171823015.png)

  点击ok之后，右下角出现你的虚拟环境名称就表示成功了。

  ![image-20230314171911330](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/imgs/image-20230314171911330.png)



以上这些步骤完成之后，恭喜你，差不多一半的工作就完事了，剩下无非就是根据不同的项目来配置环境和执行代码了。

嗷，对这里我们也说一下，如何在Pycharm中选Anaconda的虚拟环境。

## 环境配置

OK，来到关键环境配置的部分，首先大家下载代码之后会得到一个压缩包，在当前文件夹解压之后，进入CMD开始我们的环境配置环节。

![image-20230315150456715](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/imgs/image-20230315150456715.png)

为了加快后期第三方库的安装速度，我们这里需要添加几个国内的源进来，直接复制粘贴下面的这些指令到你的命令行即可。

```bash
conda config --remove-key channels
conda config --add channels https://mirrors.ustc.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.ustc.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.bfsu.edu.cn/anaconda/cloud/pytorch/
conda config --set show_channel_urls yes
pip config set global.index-url https://mirrors.ustc.edu.cn/pypi/web/simple

```

执行完毕大概是下面这个样子，后面你就可以飞速下载这些库了。

![image-20230315150835331](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/imgs/image-20230315150835331.png)

### 创建虚拟环境

首先，我们需要根据我们的项目来创建一个虚拟环境，通过下面的指令创建并激活虚拟环境。

我们创建一个Python版本为3.8.5，环境名称为yolo的虚拟环境。

```bash
conda create -n yolo python==3.8.5
conda activate yolo
```

![image-20230315152037806](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/imgs/image-20230315152037806.png)

<font color='red'>切记！这里一定要激活你的虚拟环境，否则后续你的库会安装在基础环境中，前面的小括号表示你处于的虚拟环境。</font>

### Pytorch安装

注意Pyotorch和其他库不太一样，Pytorch的安装涉及到conda和cudnn，一般而言，对于30系的显卡，我们的cuda不能小于11，对于10和20系的显卡，一般使用的是cuda10.2。下面给出了30系显卡、30系以下显卡和cpu的安装指令，请大家根据自己的电脑配置自行下载。笔者这里是3060的显卡，所以执行的是第一条指令。

```bash
conda install pytorch==1.10.0 torchvision torchaudio cudatoolkit=11.3 # 30系列以上显卡gpu版本pytorch安装指令
conda install pytorch==1.8.0 torchvision torchaudio cudatoolkit=10.2 # 10系和20系以及mx系列的执行这条
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cpuonly # CPU的小伙伴直接执行这条命令即可
```

安装之后，可以和笔者一样，输入下面的指令测试以下gpu是否可用，如果输出的是true表示GPU是可用的。

![image-20230315152204221](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/imgs/image-20230315152204221.png)

### 其余库安装

其余库的安装就非常简单了，我们通过pip来进行安装，注意这里一定要确保你执行的目录下有requirements.txt这个文件，否则你将会遇到文件找不到的bug，你可以通过`dir`指令来查看是否有这个文件。

```bash
pip install -r requirements.txt
```

![image-20230315152933301](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/imgs/image-20230315152933301.png)

### Pycharm中运行

一是为了查看代码方便，二是为了运行方便，这里我们使用Pycharm打开项目，点击这里右键文件夹直接打开项目即可非常方便。

打开之后你将会看到这样的一个界面，其中左侧是文件浏览器，中间是编辑器，下方是一些工具，右下角是你所处的虚拟环境 。

![image-20230315153430378](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/imgs/image-20230315153430378.png)

之后，我们就需要为当前的项目选择虚拟环境了，这一步非常重要，有的兄弟配置好了没选环境，你将会遇到一堆奇怪的bug，选环境的步骤如下。

首先点击，添加解释器。

![image-20230315153552702](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/imgs/image-20230315153552702.png)

三步走选择我们刚才创建的虚拟环境，点击ok。

![image-20230315153636384](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/imgs/image-20230315153636384.png)

![image-20230315153728665](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/imgs/image-20230315153728665.png)

之后你可以你可以右键执行main_window.py这个文件，出现下面的画面说明你就成功了。

![image-20230315153838054](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/imgs/image-20230315153838054.png)

![image-20230315154014173](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/imgs/image-20230315154014173.png)

## 数据集准备

数据集这里我放在了CSDN中，大家可以执行标注准备数据集，或者使用这里我处理好的数据集，数据集下载之后放在和代码目录同级的data目录下。 

![image-20230315154210695](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/imgs/image-20230315154210695.png)

数据集打开之后你将会看到两个文件夹，images目录存放图片文件，labels目录存放标签文件。

![image-20230315154304336](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/imgs/image-20230315154304336.png)

之后记住你这里的数据集路径，在后面的训练中我们将会使用到，比如笔者这里的`F:\new_project\sfid\data\target_sfid`。

![image-20230315154400117](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/imgs/image-20230315154400117.png)



## 训练和测试

注：这里你可以选择去自己尝试以下，笔者在runs的train目录下已经放了训练好的模型，你是可以直接使用。



下面就是训练的过程，笔者这里已经将数据集和模型的配置文件写好了，你只需要将数据集中的数据路径替换成你的路径，执行`go_train.py`即可开始训练了。

![image-20230315154659234](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/imgs/image-20230315154659234.png)

执行go_train.py文件中，包含三条指令，分别表示yolov5中small模型、medium模型和large模型，比如我这里要训练s模型，我就将其他两个模型训练的指令注释掉就好了。

![image-20230315155028259](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/imgs/image-20230315155028259.png)

运行之后，下方会输出运行的信息，这里的红色只是日志信息，不是报错，大家不要惊慌。

![image-20230315155258835](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/imgs/image-20230315155258835.png)

以笔者这里的s模型为例，详细含义如下。

![image-20230315155441956](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/imgs/image-20230315155441956.png)

## 图形化程序

最后就是执行我们的图形化界面程序了。

![image-20230315155624329](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/imgs/image-20230315155624329.png)

直接右键执行window_main.py执行即可，这里上两章效果图。

![image-20230315155809201](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/imgs/image-20230315155809201.png)

![image-20230315155840755](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/imgs/image-20230315155840755.png)
