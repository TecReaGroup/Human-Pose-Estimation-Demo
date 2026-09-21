# TensorRT 封装与工业部署分析

本文针对多台同配置 Windows 工控机、相同 GPU 型号和相同软件环境的部署场景，分析本项目 TensorRT 产物的封装方式。文中的发布目录、部署流程和改进项属于建议方案，不代表项目已经实现对应功能。

推荐采用：**标准 Windows 环境 + 固定版本运行时 + ONNX 模型 + 预编译 TensorRT 缓存 + 桌面启动入口**。在发布准备阶段完成耗时编译，在经过验收的同配置目标设备上复用引擎。

## 1. 当前项目的推理架构

当前执行链如下：

```text
摄像头图像
    ↓
rtmlib 前处理
    ↓
ONNX Runtime
    ├─ TensorRT：执行划分出来的主要计算子图
    ├─ CUDA：执行部分其余节点
    └─ CPU：执行部分其余节点
    ↓
rtmlib 后处理
    ↓
骨架绘制与界面展示
```

实现位置见 [pose.py](../src/rtmw_preview/pose.py:98)。项目启用了 TensorRT 引擎缓存，同时注册 CUDA 和 CPU Execution Provider。

当前模型分区策略为：

| 模型 | 排除在 TensorRT 之外的算子 | 最小子图节点数 |
| --- | --- | --- |
| YOLOX-M | `TopK`、`NonMaxSuppression` | 50 |
| RTMW-X | 无显式排除项 | 1 |

没有显式排除算子，不等于模型所有节点都一定由 TensorRT 执行，实际分配仍由 ONNX Runtime 决定。

**现有 `.engine` 是 ONNX Runtime 管理的 TensorRT 子图产物。只复制两个引擎文件，不能获得完整的姿态识别程序。** 按当前实现，仍需原始 ONNX、ONNX Runtime、rtmlib 前后处理以及应用运行依赖。

## 2. 引擎复用的环境约束

“相同电脑、相同环境”应落实为可记录、可核对的发布基线。

| 项目 | 推荐约束 |
| --- | --- |
| GPU | 同具体型号、同显存规格，不能只依据 RTX 系列或 `sm86` 判断 |
| 操作系统 | 同 Windows 版本与架构，优先采用统一系统镜像 |
| NVIDIA 驱动 | 统一为经过验收的版本 |
| Python | 固定 Python 3.12 的具体补丁版本 |
| ONNX Runtime | 当前项目为 `1.22.0` |
| TensorRT | 当前项目为 `10.9.0.34` |
| CUDA、cuDNN | 锁定发布环境中的完整运行库版本 |
| 模型 | 文件内容完全一致，记录 SHA-256 |
| 构建设置 | 固定精度、workspace、分区规则和输入 shape/profile 等参数 |
| 应用及 rtmlib | 固定版本，保证前后处理与加载行为一致 |

核心依赖声明见 [pyproject.toml](../pyproject.toml:5)，完整依赖解析结果保存在 [uv.lock](../uv.lock:1)。发布时应保留并使用经过验收的锁文件。

在上述条件一致并通过目标设备验收后，可以将引擎复用作为标准部署路径。TensorRT 引擎不应视为跨 GPU、跨操作系统、跨版本通用的模型文件。

统一驱动版本的目的是减少部署差异，并不意味着任何驱动补丁变化都会使引擎失效。模型、ONNX Runtime、TensorRT、硬件或构建设置变化时，应重新评估缓存兼容性，按新发布基线重新构建和验收。

## 3. 发布包需要包含的内容

| 内容 | 用途 |
| --- | --- |
| 应用代码、摄像头模块 | 采集、推理调度、界面展示 |
| `yolox_m.onnx`、`rtmw_x.onnx` | ONNX Runtime 加载计算图并组织执行 |
| 对应 `.engine` | 复用已经编译的 TensorRT 子图 |
| 已生成的 `.profile` | 保存输入形状范围等信息，应随对应引擎交付 |
| `.timing` | 加速可能发生的重建，不能替代引擎 |
| Python、依赖包及所需 DLL | 提供实际运行环境 |
| 配置文件 | 相机参数、GPU 编号与推理设置 |
| 发布清单 | 记录版本、目标硬件、构建参数和文件校验值 |

### 3.1 推荐交付形式

第一版建议采用可离线安装、双击启动的目录式发布包，需要正式安装体验时，再封装安装程序。

```text
release/
├─ runtime/          固定版本 Python、依赖包及所需 DLL
├─ src/              应用代码
├─ device/           摄像头模块
├─ config/           现场配置
├─ data/
│  └─ model/         两个 ONNX 模型
├─ temp/
│  └─ engine/        与当前代码匹配的预编译缓存
├─ log/              运行日志
├─ manifest.json     发布版本、环境版本、文件校验值
└─ launcher.exe      或启动脚本、桌面快捷方式
```

这是结构示意，运行时组织方式、启动入口和安装流程仍需实现。

Python 环境应作为正式运行时交付，或使用 uv 在目标安装路径离线创建。普通 `.venv` 可能包含解释器路径和入口脚本路径等依赖，不能默认当作任意位置可移动的目录。

当前项目通过源文件位置推导根目录，见 [runtime.py](../src/rtmw_preview/runtime.py:7)。保留源码相对布局可以减少适配量；改为普通 wheel 安装或冻结为 EXE 时，需要适配根目录、模型路径和 DLL 搜索方式。

摄像头模块位于主 Python 包之外，应用通过项目根目录导入，见 [app.py](../src/rtmw_preview/app.py:44)。打包时需要显式纳入该模块和外部资源。

### 3.2 EXE 封装的定位

如需 EXE，可评估 PyInstaller 目录模式或 Nuitka standalone。封装时仍需正确收集 TensorRT、ONNX Runtime、CUDA、cuDNN、Qt 及其所需依赖。

生成 EXE 主要解决交付形式，不会消除 GPU 驱动和运行库要求。对于当前 Windows 相机桌面应用，目录式发布便于定位 DLL 缺失、维护模型、替换配置和回滚版本。

## 4. 当前缓存的发布方式

代码当前使用以下两组缓存目录：

```text
temp/engine/ort_1.22.0_trt_10.9.0.34/
├─ exclude_topk_nonmaxsuppression_min_50/
│  └─ yolox_m/
└─ exclude__min_1/
   └─ rtmw_x/
```

发布时应复制这两个有效目录的配套文件，保留文件名和目录层级。历史尝试生成的 `exclude_topk`、`exclude_topk_cast` 等目录不属于当前代码选用的配置。

需要遵循以下要求：

- 提前放入两个 ONNX 模型，避免现场触发自动下载。
- 固定模型文件名、布局和构建设置；安装路径变化后，确认缓存仍能命中。
- 当前代码会创建缓存目录，并可能更新缓存，安装时需要保证对应位置具备所需写权限。
- 若后续将引擎迁入正式资产目录，应同步修改加载路径，避免重要发布资产被作为临时文件清理。
- 模型、引擎和依赖应作为同一发布版本保存，避免现场混搭。

模型与缓存被 [当前 .gitignore](../.gitignore:33) 排除，仅克隆仓库不会获得完整发布资产。发布流程需要显式收集这些文件。

### 4.1 已有运行记录

项目已有日志记录了首次构建与后续初始化耗时：

| 模型 | 首次构建对应初始化耗时 | 后续一次初始化耗时 |
| --- | --- | --- |
| YOLOX-M | 约 280.2 秒 | 约 5.8 秒 |
| RTMW-X | 约 244.8 秒 | 约 1.3 秒 |

记录见 [本机运行日志](../log/log_2026-09-21.log:82)。日志属于本地运行产物，其他仓库副本可能不存在此文件。

这些记录支持当前工程已经具备提前编译、后续复用的基础，但不代表另一台设备已经通过部署验收，也不是完整应用启动耗时的承诺。

## 5. 当前实现与工业部署的差距

| 当前情况 | 推荐处理 |
| --- | --- |
| `make install` 使用 `--upgrade` | 发布安装严格使用已验收的锁定版本 |
| `install-local` 仍可能访问网络并刷新依赖 | 准备完整离线依赖，确保断网安装可完成 |
| 缺少模型时启动流程可能触发下载 | 将模型作为必备发布资产，启动前检查完整性 |
| 推理初始化同时承担加载和编译 | 在发布阶段准备引擎，生产启动检查资产和兼容性 |
| 检查缓存目录中存在 `.engine` | 补充实际缓存加载证据，不能以文件存在代替命中确认 |
| 缓存目录记录 ORT/TRT 版本和分区策略 | 发布清单补齐模型校验值、GPU 型号、构建参数等 |
| 摄像头模块与资源位于主包之外 | 显式纳入交付清单 |

安装与启动入口见 [Makefile](../Makefile:5)。生产启动应调用已安装、已锁定的环境，避免启动过程触发依赖同步或升级。

### 5.1 混合执行与异常回退的区别

部分节点由 CUDA 或 CPU 执行，是当前设计中的混合执行策略，可以属于正常状态。TensorRT 没有承担预期计算，则需要通过日志或发布阶段的 profiling 识别。

`disable_fallback()` 不表示所有算子必须由 TensorRT 执行，也不表示禁止重新编译。仅检查 TensorRT 出现在 provider 列表中，同样不能证明所有预期子图都使用了引擎。

### 5.2 启动时间可控的要求

如果现场要求启动时绝不能临时编译数分钟，现有实现还没有提供严格的只加载保证。

建议明确区分发布构建与生产运行两个操作：

- 发布构建负责生成并验收完整模型资产。
- 生产启动负责检查版本、资产完整性和加载结果。
- 资产缺失或环境不匹配时报告部署故障，通过维护流程重建。

这些边界需要后续实现，不能仅依靠开启 `trt_engine_cache_enable` 获得。

## 6. 推荐落地流程

1. **确定基准机。** 固定 Windows、GPU、驱动、Python 和运行库版本，记录环境基线。
2. **准备最终发布布局。** 固定应用、依赖锁文件、ONNX 模型和推理配置。
3. **预编译并覆盖实际输入范围。** 当前 `make TensorRT` 可在不打开相机的情况下生成缓存，但使用零图预热。如果存在动态输入范围，还需覆盖正式业务范围，避免现场遇到新 shape 后重建。
4. **生成完整版本包。** 应用、ONNX、引擎、依赖和发布清单作为同一版本保存。
5. **在另一台同配置设备离线验收。** 确认首次启动命中缓存，无下载或重建，再检查真实画面的识别结果、延迟、显存和持续运行表现。
6. **批量安装并支持整包回滚。** 更新时切换完整版本，独立保留现场配置与日志，避免新代码配旧引擎。

发布清单至少应记录：应用版本、完整依赖版本、目标 GPU 型号、驱动基线、模型与引擎校验值、构建设置、输入范围和验收结果。

## 7. 后续架构选择

### 7.1 保留 ONNX Runtime + TensorRT

这是第一版推荐路线。它可以直接复用现有 rtmlib 前后处理、算子分区和界面，重点补齐离线交付、版本管理和预编译资产管理，改造范围较小。

### 7.2 原生 TensorRT SDK

当出现明确的 C++ 集成、体积或启动时间要求时，可评估原生 TensorRT SDK。该路线需要自行接管输入输出、显存、动态 shape，以及当前交由 CUDA/CPU 执行的计算，属于推理架构改造，不能通过简单替换模型后缀完成。

### 7.3 EPContext / 嵌入式引擎

ONNX Runtime 提供 EPContext / 嵌入式引擎方向，可减少部分初始化工作。其 TensorRT EP 文档列出了整模型需要满足 TensorRT 执行条件等约束，因此不能直接套用到当前主动排除部分 YOLOX 算子的方案。

如果未来选择该路线，需要针对锁定的 ONNX Runtime 版本确认实际支持范围、模型分区限制和资源打包方式。

## 8. 官方参考

- [ONNX Runtime TensorRT EP：版本要求](https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html#requirements)
- [ONNX Runtime TensorRT EP：引擎缓存与失效条件](https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html#trt_engine_cache_enable)
- [ONNX Runtime TensorRT EP：缓存类型](https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html#tensorrt-ep-caches)
- [ONNX Runtime TensorRT EP：EPContext 约束](https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html#more-about-embedded-engine-model--epcontext-model)

官方在线文档可能随版本更新，实施时应以项目锁定版本的实际支持能力为准。
