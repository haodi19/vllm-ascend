# 蓝绿区协同开发说明

本文档用于向 AI 说明当前的开发环境、代码来源和 Git 协作方式。后续协助开发时，
必须遵守这里的职责划分和安全限制。

## 一、当前项目的代码来源

上游仓库和基线分支为：

```text
上游仓库：https://github.com/LookAround0301/vllm-ascend
上游基线分支：moe_offload_v0260rc_k3_0812
```

GitHub 的 Fork 操作针对整个仓库，而不是只 Fork 某一个分支。因此，“从
`moe_offload_v0260rc_k3_0812` 分支 Fork 一份”的实际含义是：

1. 将上游仓库 Fork 到个人 GitHub 账号；
2. 从上游的 `moe_offload_v0260rc_k3_0812` 分支创建个人开发分支；
3. 将个人开发分支推送到个人 Fork 仓库；
4. 蓝区和绿区都使用个人 Fork 仓库，不直接在上游仓库开发。

如果在 GitHub Fork 页面看到“只复制默认分支”一类选项，应取消该选项，以便复制其他
分支。即使个人 Fork 中没有目标分支，也可以在蓝区添加 `upstream` 后从目标分支创建。

## 二、蓝区与绿区的职责

### 蓝区：Vibe Coding 开发环境

蓝区可以访问外网，是与 AI 进行 vibe coding 的主要环境。

蓝区中的工作包括：

- AI 阅读和分析项目代码；
- AI 根据需求修改代码和编写测试脚本；
- AI 进行可在蓝区完成的静态检查；
- 用户检查 AI 给出的修改和 Git diff；
- 在用户确认后创建 commit；
- 将开发分支 push 到个人 Fork 仓库。

蓝区通常没有项目所需的昇腾 NPU 或完整 CANN 运行环境，所以无法真正完成 NPU 构建和
测试。AI 不能因为静态检查通过，就宣称修改已经在目标硬件上验证成功。

AI 必须区分以下状态：

- 已完成代码修改；
- 已完成蓝区静态检查；
- 尚待绿区构建和测试；
- 已由用户在绿区验证通过。

### 绿区：由用户操作的构建和测试环境

绿区的所有命令均由用户本人执行，不是由蓝区 AI 直接操作。AI 的职责是提供清晰的命令、
预期结果和排查方法。

绿区用于：

- 从个人 Fork 仓库拉取蓝区提交的代码；
- 构建项目或自定义算子；
- 在昇腾 NPU 上运行功能、精度、安全性和性能测试；
- 由用户人工检查测试结果；
- 将经过脱敏的关键文本复制回蓝区供 AI 分析。

绿区必须遵守以下信息安全和 Git 限制：

- 绿区绝对不能执行 `git push`；
- 不在绿区创建正式开发 commit；
- 不直接向蓝区传输可能包含敏感信息的完整日志或文件；
- 日志必须由用户人工检查，只复制必要且脱敏后的内容；
- 绿区发现的问题应返回蓝区，由 AI 在蓝区修改，再通过个人 Fork 仓库同步回来。

### 测试交付约定

后续每轮测试都提供一个可一次执行的脚本，用户运行后返回经过检查的控制台输出。
脚本不包含 Git 操作，不把测试结果或日志写入文件；构建所需的正常产物不在此限制内。
需要构建、校验和多轮性能对比时，由脚本按顺序执行，并在任一步失败时停止。

当前迁移测试入口：`python -B script/run_load_migration_tests.py`。

## 三、远端仓库关系

推荐在蓝区使用以下远端命名：

```text
origin   -> <PERSONAL_FORK_URL>
upstream -> https://github.com/LookAround0301/vllm-ascend
```

其中：

- `origin` 是个人 Fork，蓝区从这里拉取并向这里推送；
- `upstream` 是导师指定的代码来源，只用于获取上游代码；
- 绿区通常只需要 `origin`，且只能从 `origin` 拉取，不能推送；
- 不向 `upstream` 直接 push。

## 四、首次准备个人 Fork

### 1. 在 GitHub 上创建 Fork

用个人账号打开：

```text
https://github.com/LookAround0301/vllm-ascend
```

点击 **Fork**，创建个人仓库：

```text
<PERSONAL_FORK_URL>
```

如果页面提供“只复制默认分支”的选项，取消勾选。Fork 完成后，不要直接在导师的上游仓库
中开发。

### 2. 蓝区首次克隆并创建开发分支

```bash
git clone <PERSONAL_FORK_URL> <BLUE_PROJECT_DIRECTORY_NAME>
cd <BLUE_PROJECT_DIRECTORY_NAME>

git remote add upstream https://github.com/LookAround0301/vllm-ascend.git
git fetch upstream moe_offload_v0260rc_k3_0812

git switch -c <DEVELOPMENT_BRANCH> upstream/moe_offload_v0260rc_k3_0812
git push -u origin <DEVELOPMENT_BRANCH>
```

这样可以确保个人开发分支确实以导师指定的上游分支为起点，不依赖 Fork 时是否复制了该
分支。

检查远端和分支关系：

```bash
git remote -v
git branch -vv
git log --oneline --decorate -5
```

### 3. 绿区首次克隆个人开发分支

以下命令由用户在绿区执行：

```bash
git clone -b <DEVELOPMENT_BRANCH> <PERSONAL_FORK_URL> <GREEN_PROJECT_DIRECTORY_NAME>
cd <GREEN_PROJECT_DIRECTORY_NAME>

git branch --show-current
git rev-parse HEAD
git status
```

绿区克隆的是个人 Fork，不是导师的上游仓库。

## 五、日常开发闭环

```text
蓝区：AI 进行 vibe coding
    ↓
蓝区：用户检查修改，创建 commit 并 push 到个人 Fork
    ↓
绿区：用户 pull 同一开发分支
    ↓
绿区：用户构建并在 NPU 上测试
    ↓
用户：复制脱敏后的关键结果回蓝区
    ↓
蓝区：AI 分析结果并进行下一轮修改
```

## 六、蓝区日常 Git 操作

开始一轮开发前：

```bash
git switch <DEVELOPMENT_BRANCH>
git pull --ff-only origin <DEVELOPMENT_BRANCH>
git status
```

AI 修改代码后，先检查本轮差异：

```bash
git status
git diff
git diff --stat
```

只添加本次任务涉及的文件：

```bash
git add <FILE_1> <FILE_2>
git diff --cached
git status
```

不要在没有检查文件范围时直接使用 `git add .`。

按照当前仓库的贡献规范创建提交。例如，如果项目要求 Signed-off-by：

```bash
git commit -s -m "<TYPE>: <COMMIT_MESSAGE>"
```

推送到个人 Fork：

```bash
git push origin <DEVELOPMENT_BRANCH>
```

获取本轮 commit ID，随后将它告知绿区操作者：

```bash
git rev-parse HEAD
```

除非用户明确授权，否则 AI 不应自行执行 `git commit` 或 `git push`。AI 可以修改代码、
完成静态检查并给出命令，由用户检查后执行。

## 七、绿区日常 Git 操作

以下命令全部由用户在绿区执行：

```bash
git switch <DEVELOPMENT_BRANCH>
git status
git fetch origin
git pull --ff-only origin <DEVELOPMENT_BRANCH>
```

测试前确认版本：

```bash
git branch --show-current
git rev-parse HEAD
git status --short
```

需要确认：

- 当前分支是 `<DEVELOPMENT_BRANCH>`；
- HEAD 与蓝区刚推送的 commit ID 一致；
- 工作区没有来源不明的未提交修改。

然后由用户执行：

```bash
<BUILD_COMMAND>
<TEST_COMMAND>
```

绿区禁止执行：

```bash
git push
```

如果绿区发现代码需要修改，不要形成只存在于绿区的正式修复。先执行：

```bash
git status
git diff
```

然后将必要的错误信息或差异以脱敏文本返回蓝区，由 AI 在蓝区修改。

## 八、同步上游基线的方式

只有确实需要吸收导师上游仓库的新提交时，才在蓝区执行上游同步。不要在绿区完成该操作。

蓝区先获取上游分支：

```bash
git fetch upstream moe_offload_v0260rc_k3_0812
git log --oneline --decorate --graph --max-count=20 \
    <DEVELOPMENT_BRANCH> upstream/moe_offload_v0260rc_k3_0812
```

是否 merge 或 rebase 需要根据团队要求决定，不能由 AI 擅自选择。确认策略后，再在蓝区完成
同步、解决冲突、测试可执行的内容并 push 到个人 Fork。绿区随后仍只从个人 Fork 拉取。

## 九、绿区结果返回格式

每次测试建议返回：

```text
项目：<PROJECT_NAME>
分支：<DEVELOPMENT_BRANCH>
Commit ID：<COMMIT_ID>
工作区状态：clean / <未提交文件说明>
测试设备：<DEVICE_INFO>
构建配置：<BUILD_CONFIGURATION>
实际测试命令：<ACTUAL_TEST_COMMAND>
功能或精度结果：<CORRECTNESS_RESULT>
性能结果：<PERFORMANCE_RESULT>
结论：通过 / 失败 / 需要进一步分析
```

性能实验还要记录所有影响可比性的条件，例如：

- batch size；
- sequence length；
- 数据类型；
- warmup 和 iteration 次数；
- 使用的物理设备和逻辑设备；
- 设备负载或空闲情况；
- 优化模式及相关环境变量；
- 其他项目特有参数。

不要只返回一个时延或吞吐量数值，否则无法判断实验是否可比。

## 十、AI 协作规则

1. 修改前先检查仓库实际存在的说明文件、贡献规范、现有实现、构建流程和测试调用链，
   不得假设某个固定名称的说明文件一定存在。
2. AI 在蓝区进行 vibe coding；绿区命令由用户本人执行。
3. 尽量做扩展型、向后兼容的修改，并保留旧模式用于 A/B 对比。
4. 新增优化开关时，说明默认模式、启用方法和恢复旧模式的方法。
5. 不修改与当前任务无关的代码，不顺便修复无关的历史问题。
6. 未提交改动可能属于用户，不得擅自覆盖或删除。
7. 未经明确允许，不执行 `git reset --hard`、`git checkout --`、`git clean -fd`
   或强制推送。
8. 删除测试脚本前，先确认它是一次性排查脚本，还是需要长期保留的正式工具。
9. 绿区使用的脚本应输出简洁、可复制、方便脱敏的摘要。
10. 蓝区不能完成的构建或测试必须明确标记为“待绿区验证”。
11. 给出命令时，必须标明它应在蓝区还是绿区执行。
12. 只有蓝区可以向个人 Fork push，绿区绝对不能 push。

## 十一、异常处理

如果 `git pull --ff-only` 失败，先检查：

```bash
git status
git log --oneline --decorate -10
git branch -vv
```

不要立即强制推送或执行破坏性重置。

如果绿区因未提交修改无法 pull，先执行：

```bash
git status
git diff
```

在不知道修改来源时，不要直接丢弃这些改动。

如果出现：

```text
warning: CRLF will be replaced by LF
```

这通常表示 Windows 工作区使用 CRLF，而 Git 仓库将文件保存为 LF，不一定是代码错误。
Shell、CMake 和 Linux 脚本通常应使用 LF，但仍需检查是否意外改变了无关文件的换行符。

## 十二、核心原则

```text
代码来源：导师指定的上游仓库和基线分支。
开发仓库：用户自己的 GitHub Fork。
蓝区：AI 辅助 vibe coding，用户审核、提交并推送。
绿区：用户拉取、构建和测试，绝不推送。
结果传递：用户人工检查并脱敏后，将关键文本返回蓝区。
```
