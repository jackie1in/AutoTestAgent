# Neo4j 版本化存储灰度清单

## 目标
- 在不影响现网 `runner -> mapping_pipeline/manual_capture -> persistence` 主链路的前提下，逐步从覆盖写切到 `ingestion/revision/release` 三层版本模型。
- 任意阶段可回滚，且不需要删除历史数据。

## 阶段 N+1：双写上线（写侧）
- [ ] 执行迁移到 `005`（含 `IngestionRun / TransitionEntity / TransitionRevision / GraphRelease` 约束与索引）。
- [ ] 确认写侧双写生效：
  - [ ] 新 Session 产生 `(:IngestionRun)`。
  - [ ] `(:Session)-[:GENERATES]->(:IngestionRun)` 存在。
  - [ ] 本轮写入节点存在 `ingest_version_id`。
  - [ ] `(:IngestionRun)-[:EMITS]->(:State|:Transition|:Evidence|:Menu|:Zone)` 存在。
  - [ ] 同步写 `TransitionEntity/TransitionRevision`，并标记 `is_active`。
- [ ] 基础健康检查：
  - [ ] `persist_mapping_result` 正常完成，无写失败重试风暴。
  - [ ] `merge_consistency`、`manual_hybrid`、`models` 测试通过。
  - [ ] 采样检查 revision 链：同 stable_key 多次写入可看到 `SUPERSEDES`。

## 阶段 N+2：读侧灰度（release 优先）
- [ ] 在灰度环境设置 `MAPPING_RELEASE_ID`，启用 release 读取路径。
- [ ] 验证读取策略：
  - [ ] release 命中时，从 `TransitionRevision(is_active=true)-[:IN_RELEASE]->GraphRelease` 读取。
  - [ ] release 不存在或为空时，自动 fallback 到旧 `Transition` 读取。
- [ ] 兼容性检查：
  - [ ] NL resolve / playback 行为与切换前一致（或仅有可解释差异）。
  - [ ] 前端 `/api/playback` 路径构建无异常。
  - [ ] 读侧指标（延迟、错误率）与基线相比在可接受范围。

## 阶段 N+3：收敛与固化
- [ ] 默认启用 release 读取（生产环境）。
- [ ] 保留 fallback 开关一个发布窗口（至少 1 周期）。
- [ ] 完成稳定性复盘：
  - [ ] 冲突决策可解释（revision source/confidence/supersedes 链完整）。
  - [ ] 回滚演练通过（按 release 切回旧版本）。
- [ ] 决策是否逐步弱化旧字段写入（仅在连续稳定后）。

## 回滚策略（按风险等级）

### Level 1：读侧回滚（最快）
- 操作：清空或关闭 `MAPPING_RELEASE_ID`，恢复旧查询路径。
- 预期：不影响写入，业务立即恢复到旧读取结果。

### Level 2：停用 revision 激活逻辑
- 操作：保留双写，但将 active revision 选择逻辑降级为“仅记录，不参与决策”。
- 预期：可继续保留审计数据，同时降低读侧行为变化。

### Level 3：完整回退到旧行为
- 操作：关闭新写逻辑入口，仅保留原 `Transition` 写入与读取。
- 注意：不删除新节点，避免丢失审计轨迹，待故障复盘后再恢复灰度。

## 推荐巡检 Cypher（上线后抽样）

```cypher
MATCH (s:Session)-[:GENERATES]->(run:IngestionRun)
RETURN s.id, run.id, run.mode, run.created_at
ORDER BY run.created_at DESC LIMIT 20;
```

```cypher
MATCH (run:IngestionRun)-[:EMITS]->(t:Transition)
RETURN run.id, count(t) AS transition_count
ORDER BY transition_count DESC LIMIT 20;
```

```cypher
MATCH (ent:TransitionEntity)-[:HAS_REVISION]->(rev:TransitionRevision)
OPTIONAL MATCH (rev)-[:SUPERSEDES]->(old:TransitionRevision)
RETURN ent.stable_key, rev.revision_id, rev.is_active, old.revision_id
LIMIT 50;
```

```cypher
MATCH (rev:TransitionRevision {is_active: true})-[:IN_RELEASE]->(r:GraphRelease {id: $release_id})
RETURN r.id, count(rev) AS active_revision_count;
```
