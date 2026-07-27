# Chronos-AEGIS Benchmark & Ablation Summary Tables

## Scenario Summary (final-query-time metrics)

| Scenario | n | trust_mean | p_success | mdri_dependency_node_count |
|---|---|---|---|---|
| adverse_weather | 10 | 0.5139 [0.4361, 0.5917] | 0.4993 | 0.00 |
| communication_loss | 10 | 0.4844 [0.4020, 0.5668] | 0.4673 | 0.00 |
| compromised_firmware_deployment | 10 | 0.4843 [0.4270, 0.5416] | 0.5202 | 1.00 |
| delayed_firmware_rollout | 10 | 0.5018 [0.4449, 0.5587] | 0.4495 | 1.00 |
| firmware_update_in_flight | 10 | 0.4646 [0.3935, 0.5357] | 0.4942 | 1.00 |
| fuel_shortage | 10 | 0.4815 [0.3986, 0.5644] | 0.4739 | 0.00 |
| gps_degradation | 10 | 0.4646 [0.3854, 0.5439] | 0.4651 | 0.00 |
| maintenance_delay | 10 | 0.4815 [0.3986, 0.5644] | 0.4739 | 0.00 |
| mixed_cyber_operational | 10 | 0.4740 [0.4230, 0.5250] | 0.4709 | 0.00 |
| mixed_firmware_and_communication_degradation | 10 | 0.4819 [0.4244, 0.5394] | 0.4958 | 1.00 |
| multiple_simultaneous_failures | 10 | 0.4921 [0.4300, 0.5541] | 0.4834 | 0.00 |
| normal_mission | 10 | 0.4815 [0.3986, 0.5644] | 0.4739 | 0.00 |
| runway_closure | 10 | 0.4815 [0.3986, 0.5644] | 0.4739 | 0.00 |
| satellite_blackout | 10 | 0.5379 [0.4639, 0.6120] | 0.4122 | 0.00 |
| sensor_spoofing | 10 | 0.4942 [0.4134, 0.5750] | 0.4844 | 0.00 |
| software_exploit_injection | 10 | 0.4695 [0.4212, 0.5178] | 0.4737 | 0.00 |
| staged_patch_rollout | 10 | 0.5012 [0.4110, 0.5915] | 0.4865 | 1.00 |
| supply_chain_delay | 10 | 0.4815 [0.3986, 0.5644] | 0.4739 | 0.00 |

## Retrieval Accuracy (RQ6/RQ7)

| n_episodes | precision@1 | majority-class baseline |
|---|---|---|
| 180 | 0.7722 | 0.5000 |

## Scheduler Calibration Benchmark

| initial w_cyber | initial w_trust | final w_cyber | final w_trust | signal strength |
|---|---|---|---|---|
| 1.0000 | 1.0000 | 1.1455 | 0.8545 | 0.80 |

## Continual-Learning Convergence (Section 16, RQ10)

| mission | scenario | task_loss | ewc_penalty | replay_batch_size |
|---|---|---|---|---|
| 0 | normal_mission | 0.873289 | 0.000000 | 0 |
| 1 | satellite_blackout | 0.532723 | 0.000000 | 1 |
| 2 | gps_degradation | 0.440391 | 0.000000 | 2 |
| 3 | adverse_weather | 0.413547 | 0.000000 | 3 |
| 4 | maintenance_delay | 0.366952 | 0.000000 | 4 |
| 5 | runway_closure | 0.368033 | 0.000000 | 5 |
| 6 | software_exploit_injection | 0.520445 | 0.000000 | 6 |
| 7 | sensor_spoofing | 0.547037 | 0.000000 | 7 |
| 8 | communication_loss | 0.529597 | 0.000000 | 8 |
| 9 | fuel_shortage | 0.556231 | 0.000000 | 8 |
| 10 | supply_chain_delay | 0.593655 | 0.000000 | 8 |
| 11 | multiple_simultaneous_failures | 0.424051 | 0.000000 | 8 |
| 12 | mixed_cyber_operational | 0.554628 | 0.000000 | 8 |
| 13 | firmware_update_in_flight | 0.588463 | 0.000000 | 8 |
| 14 | delayed_firmware_rollout | 0.479703 | 0.000000 | 8 |
| 15 | compromised_firmware_deployment | 0.610230 | 0.000000 | 8 |
| 16 | staged_patch_rollout | 0.480370 | 0.000000 | 8 |
| 17 | mixed_firmware_and_communication_degradation | 0.459174 | 0.000000 | 8 |

## No-EMM Ablation (Section 17.3)

Condition A: full (EpisodicMissionMemory enabled)

Condition B: no_emm (NullEpisodicMemory -- Algorithm 1 lines 3/15 skipped)

| metric | full | no_emm | difference | Cohen's d |
|---|---|---|---|---|
| trust_mean | 0.4873 | 0.4873 | 0.0000 | 0.0000 |
| p_success | 0.4762 | 0.4762 | 0.0000 | 0.0000 |
| expected_delay | 0.9400 | 0.9400 | 0.0000 | 0.0000 |
| cyber_impact | 0.4878 | 0.4878 | 0.0000 | 0.0000 |
| comm_reliability | 0.4467 | 0.4467 | 0.0000 | 0.0000 |
| operational_confidence | 0.4841 | 0.4841 | 0.0000 | 0.0000 |
| mdri_value | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| mdri_dependency_node_count | 0.2778 | 0.2778 | 0.0000 | 0.0000 |
| memory_episodes_retrieved | 4.6389 | 0.0000 | 4.6389 | 6.5145 |
| warning_count | 0.0000 | 1.0000 | -1.0000 | 0.0000 |
