# Engineering norms

## Reuse proven work before writing new code

- **Build on what already exists.** When the thing you're writing resembles code already in the
  repo — a new variant of an existing computation, a sibling path, a parallel module — study the
  existing version first and carry over its hard-won design and optimizations. A new path should
  inherit its sibling's performance engineering, not just reproduce its behavior.

- **Apply known, already-proven performance wins up front, not as an afterthought.** If an
  optimization exists elsewhere in the codebase, or a comment flags one as deferred ("TODO",
  "perf win later", "not yet"), treat it as a debt to pay *before* the code runs at scale — a
  deferred-performance note means the path is not ready for large or expensive runs.

- **Cost-check before anything expensive.** Before launching large jobs, long runs, or cluster
  work, estimate or measure memory and runtime on a representative worst case first. A two-minute
  probe is far cheaper than a failed multi-hour run.

## Communication

- Write in plain, standard language. The reader knows standard programming and CS terminology but
  NOT operations / systems / queueing jargon. Avoid that jargon and unusual figurative terms;
  prefer the ordinary word. E.g. say a job "finished" (not "drained"), the queue is "empty" (not
  "drained"), something is "important" / "essential" / "critical" (not "load-bearing"). When in
  doubt, choose the most common, literal verb.
