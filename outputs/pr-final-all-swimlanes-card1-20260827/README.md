# 最终PR版本泳道

- 设备：Ascend950PR_9579卡1。
- CANN：9.2.0。
- 参数：每个case一轮，`--skip-golden --manual include --enable-chip-swimlane 4`。
- 代码：最终PR源代码树；泳道采集完成后仅执行squash，不改变代码内容。
- 状态：8/8 PASS，详见`status.tsv`。

各子目录的`merged_swimlane.json`为Perfetto可读泳道图，`name_map_*.json`为kernel名称映射。
