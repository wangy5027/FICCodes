# BoA-SQL (2026)

**BoA-SQL: Executable Blueprint-of-Action for Text-to-SQL with reinforcement learning**

- 作者：Yang Wang; Zhilong Xie; Lin Zhang; Lingyun Gu; Qing Li
- 期刊：Engineering Applications of Artificial Intelligence, 166, 113454 (2026)
- 代码提交与维护：[wangy5027](https://github.com/wangy5027)

本目录提供 BoA-SQL 的 Agent 推理与 Spider 实验部分，通过数据库结构筛选、任务规划、逐步 SQL 生成及执行反馈完成 Text-to-SQL。**强化学习训练代码不包含在本次提交中。**

## 开始使用

- [项目说明、环境配置、运行与评测](Spider_Experiment/README.md)
- [Spider 和 test-suite 数据下载及放置说明](Spider_Experiment/DATA_SETUP.md)
- [Agent 运行入口](Spider_Experiment/Run.py)

外层文件夹按团队规范命名为 `YangWang_BoA-SQL_2026_EAAI`。内部以 `Spider_Experiment` 存放 Agent 项目。资源路径根据源码位置定位，该子目录可重命名；请保持其内部相对层级。本文链接使用仓库当前目录名称。

默认模型接口使用 `OPENAI_API_KEY`、`OPENAI_BASE_URL` 和 `OPENAI_MODEL` 环境变量，不包含真实 API 密钥。

本次保留 Agent 源码、test-suite 工具、三份历史预测及配套 gold，以及修正版评测所需的 70 个参考文件。原始数据、模型权重、运行缓存、旧 Git 历史和中间实验结果不随代码上传。

当前版本已完成静态检查、模拟入口检查和路径迁移回归检查，尚未重新执行完整实验；依赖版本、最终预测版本及已知运行限制见项目说明。

贡献流程和修改范围请遵守[仓库主 README](../README.md)。
