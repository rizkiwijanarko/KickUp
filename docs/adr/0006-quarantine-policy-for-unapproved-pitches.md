# Quarantine Policy for Unapproved Pitches

When an idea reaches the maximum reflection revision limit without passing 100% of the Critic's binary rubric checks, treating it as approved misleads downstream deck generation, while discarding it entirely destroys generated synthesis. We decided to establish an explicit Quarantine Policy: pitches that fail critique at max revisions are segregated into `quarantined_pitches` with attached flaw diagnostics and failing check summaries, while fully certified briefs are exposed as `approved_pitches`. Both are exported cleanly in the output artifact.
