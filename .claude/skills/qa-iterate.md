---
name: qa-iterate
description: Run one QA iteration against a Rakuten AI scenario agent and read the handoff block to decide next steps
---

# QA Iteration Skill

Run this skill to evaluate a scenario agent, see the pass rate trend, and get a proposed prompt
improvement from the advisor. The full iterative workflow is documented in `AGENT_SKILLS.md`.

## Steps

1. **Read `AGENT_SKILLS.md`** at the project root for full context on the workflow.

2. **Run one iteration:**
   ```sh
   qa-agent iterate \
     --scenario-id <SCENARIO_ID> \
     --purpose "<what this agent should do>" \
     [--config qa_agent.yaml]
   ```

3. **Read the AGENT HANDOFF block** printed at the end of the run. It tells you:
   - Current pass rate and trend (IMPROVING / STABLE / REGRESSING)
   - Whether the advisor produced a proposed prompt
   - The exact command to run next iteration

4. **If the advisor proposed a new prompt:**
   - Read the run file: `runs/<scenario_id>/run-<N>-*.json`
   - Find the `advice` field and extract the `PROPOSED PROMPT` section
   - Apply it to MongoDB: `db.task_agents.updateOne({_id: ObjectId(...)}, {$set: {"instructions.prompt": "<new prompt>"}})`
   - Then run the next iteration (step 2 again)

5. **Check history at any time:**
   ```sh
   qa-agent history --scenario-id <SCENARIO_ID>
   ```

6. **Open the dashboard to browse all runs visually:**
   ```sh
   qa-agent dashboard
   ```
   Opens `http://localhost:8080` in your browser. Shows all scenarios, pass-rate trends,
   prompt diffs, per-test conversation turns, and advisor output. Press Ctrl+C to stop.

## Notes

- Each iteration is automatically numbered (1, 2, 3…) per scenario
- The advisor sees all previous iterations and won't repeat suggestions that didn't work
- Run files live at `runs/<scenario_id>/run-<N>-<timestamp>.json`
- You decide when to apply proposed prompts — nothing is auto-applied to MongoDB
