/**
 * Project-local Atomic workflow. Manual review gates are intentional.
 * Authored against the published Atomic workflow({...}) API shape; the file
 * must still be reloaded and smoke-tested in an installed Atomic CLI.
 *
 * Prefer Atomic's BUILT-IN `goal` workflow for a stronger autonomous
 * deterministic reviewer/ledger loop once the repository plans are in place.
 */
import { workflow } from "@bastani/workflows";
import { Type } from "typebox";

export default workflow({
  name: "feargreed-delivery",
  description: "Review and execute one FearGreedIndex ExecPlan with human gates.",
  inputs: {
    plan_path: Type.String({ description: "Checked-in .agent/execplans/FGI-*.md path." }),
  },
  outputs: {
    status: Type.String({ description: "cancelled, ready-for-owner-review, or needs-rework." }),
    summary: Type.String({ description: "Implementation and validation handoff." }),
  },
  run: async (ctx) => {
    const planPath = String(ctx.inputs.plan_path);

    const preflight = await ctx.task("inspect-plan-and-branch", {
      reads: [planPath],
      context: "fresh",
      prompt: [
        `Read ${planPath}, AGENTS.md, PRODUCT.md, DESIGN.md, .agent/PLANS.md,`,
        `.agent/ROADMAP.md, .agent/STATUS.md, and the relevant source/tests.`,
        `Confirm the plan is allowed under frozen v2.1 and research-only v3 boundaries.`,
        `Report current Git branch/status, exact implementation scope, test commands,`,
        `open decisions, blockers, and a safe checkpoint strategy.`,
        `Do not edit files yet. Never guess missing approvals.`,
      ].join(" "),
    });

    const startApproved = await ctx.ui.confirm(
      `Review preflight for ${planPath}. Start implementation on a safe feature branch?\n\n${preflight.text}`,
    );
    if (!startApproved) return { status: "cancelled", summary: preflight.text };

    const implementation = await ctx.task("implement-and-checkpoint", {
      reads: [planPath],
      prompt: [
        `Follow AGENTS.md and implement ONLY the approved ExecPlan ${planPath}.`,
        `Honor protected research/data boundaries and stop conditions.`,
        `Update the ExecPlan progress and .agent/STATUS.md with actual results,`,
        `commits, blockers and next exact step as you work; commit meaningful changes.`,
        `No merge, tag, deployment, or publishing. Report the changed files.`,
      ].join(" "),
    });

    const validation = await ctx.task("run-and-report-validation", {
      reads: [planPath],
      context: "fresh",
      prompt: [
        `Independently inspect the diff and ${planPath}. Run applicable exact tests,`,
        `build commands, publication/schema checks, and relevant regression coverage.`,
        `Report commands, exit codes, expected versus actual results, and any unrun tests.`,
        `Do not mark anything passing without evidence. Never alter research evidence.`,
      ].join(" "),
    });

    const review = await ctx.task("independent-boundary-review", {
      reads: [planPath],
      context: "fresh",
      prompt: [
        `Review the implementation independently against AGENTS.md and ${planPath}.`,
        `Check methodology parity, v3 shadow isolation, future-data leakage,`,
        `sensitive data/licenses, security, deployment risks, test gaps, and scope creep.`,
        `Return explicit BLOCKER findings or state NO BLOCKERS with supporting evidence.`,
        `Never approve on missing or unrun mandatory validation.`,
      ].join(" "),
    });

    const summary = [
      "IMPLEMENTATION", implementation.text,
      "VALIDATION", validation.text,
      "INDEPENDENT REVIEW", review.text,
    ].join("\n\n");

    const ownerAccepted = await ctx.ui.confirm(
      `Review the evidence and confirm this work is ready for YOUR review.\n` +
      `This confirmation does NOT merge, deploy, or publish.\n\n${summary}`,
    );

    return {
      status: ownerAccepted ? "ready-for-owner-review" : "needs-rework",
      summary,
    };
  },
});
