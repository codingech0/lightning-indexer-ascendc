# MhcPre project rules

- Default source is archived BR5. Preserve its timestamps when merely reorganizing the repository.
- Each new candidate must put its actual Asia/Shanghai (UTC+08:00) modification timestamp and shared version identifier on the first line of both code/op_host/mhc_pre.cpp and code/op_kernel/mhc_pre.cpp, even if only one changes logically.
- Compile, correctness-test and benchmark on the remote Ascend NPU via background SSH. Do not use Computer Use or local CPU tests as substitutes.
- Preserve the last platform-passing archive and an independently installed NPU baseline.
- Only offer a new competition submission after real NPU validation. Report correctness failures, performance regressions, and whether platform latency below 3 us was actually achieved.
- Keep credentials, connection details, generated builds, and private environment paths out of Git.
