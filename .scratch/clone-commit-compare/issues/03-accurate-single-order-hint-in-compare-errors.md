# 03: Accurate single-order hint in `compare` errors

**What to build:** A state remembers which orders each step's QA Artifacts cover. When `compare` finds a state without a QA Artifact for the requested step, its error names that state and mentions that a single-order run only produces that order's artifact only when the state's artifacts really came from a single-order run. Otherwise the error just says to re-run the step with QA on. This completes spec story 26.

**Blocked by:** 02

**Status:** ready-for-agent

- [x] The error for a state whose step was run with QA off and never targeted an order does not mention single-order runs
- [x] The error for a state whose step artifacts came only from a single-order run names that order
- [x] The error still identifies which of the two states lacks the artifact
- [x] Order coverage of QA Artifacts is cloned with the state and dropped when a Commit discards those artifacts
- [x] Tests assert on the raised error through `compare`, not on how coverage is stored
