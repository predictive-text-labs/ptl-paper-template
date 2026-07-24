export const meta = {
  name: 'rejudge-confident',
  description:
    'Re-judge rejected rewrites with a confident-voice rule: accept dropping reflexive softeners that do not change the claim; keep result-bounding hedges, quantifiers, defined terms, and citations.',
  whenToUse: 'After the judge rejected many rewrites for dropped hedges, when the paper wants a more confident voice.',
  phases: [{ title: 'Re-judge', detail: 'Fable: is the dropped hedge/term reflexive (accept) or load-bearing (keep)?' }],
}

// Input (via args): { batch_dir, batch_count, model? }
// Each batch file is a 1-item array: {id, original_sentence, gemini_raw_response, prior_reject_reason}.
// Returns { accepted:[{id, rewrite}], kept_rejected:[{id, note}], stats }

const RUBRIC = `This is a confident, stance-taking academic paper, and the authors want to shed REFLEXIVE hedging — reflexive softeners weaken the voice without changing what is claimed. A previous judge rejected these shortened rewrites, usually for dropping a hedge/qualifier or loosening a term. Re-judge each with the rule below.

You are given the ORIGINAL sentence, Gemini's shortened REPLY, and the prior reject reason. First pull the single best clean rewrite from the reply (strip options/commentary/fences; preserve all LaTeX markup verbatim — \\citep{...}, \\ref{...}, $...$). Then decide:

ACCEPT (keep=true) if the rewrite's ONLY loss versus the original is a REFLEXIVE SOFTENER or harmless stylistic looseness that does NOT change what is asserted — e.g. dropping "we believe", "arguably", "essentially", "somewhat", or a bare "can"/"may" where the claim already stands on its own; using a plain synonym for a NON-technical word. The paper is allowed to sound confident and direct.

KEEP REJECTED (keep=false) if the change alters the claim's TRUTH CONDITIONS or drops load-bearing content:
 - a hedge that bounds an empirical RESULT's scope, sign, or certainty ("may not equal", "under our assumptions", "suggests", "tends to", "for these questions and forecasters") — dropping it OVERCLAIMS a result;
 - a quantifier that changes the claim ("each", "all", "every", "only", "at least", "both");
 - a precisely DEFINED technical term the paper relies on (e.g. "latency error", "modeling error", "executable path", "information latency", "worked example") swapped for a vaguer word;
 - a dropped/altered citation, reference, or math; or any changed factual claim, sign, or scope.

The rewrite must also still be shorter than the original and a grammatical, declarative drop-in (same speech act — not an imperative).

When genuinely unsure whether a hedge is reflexive or result-bounding, KEEP REJECTED — overclaiming an empirical paper is worse than a little verbosity. Put the clean rewrite in "rewrite" when keep=true. Return ONLY schema-valid JSON, one verdict per id, id echoed exactly.`

function rejudgePrompt(path) {
  return `${RUBRIC}

FIRST, use your Read tool to read the JSON array (one item) at:
  ${path}
The item has: id, original_sentence, gemini_raw_response, prior_reject_reason. Re-judge it and return exactly one verdict.`
}

const SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['verdicts'],
  properties: {
    verdicts: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['id', 'keep'],
        properties: {
          id: { type: 'string' },
          keep: { type: 'boolean' },
          rewrite: { type: ['string', 'null'] },
          note: { type: ['string', 'null'] },
        },
      },
    },
  },
}

const A = typeof args === 'string' ? JSON.parse(args) : args || {}
let batchPaths = A.batch_paths || []
if (!batchPaths.length && A.batch_dir && A.batch_count) {
  batchPaths = Array.from(
    { length: A.batch_count },
    (_, i) => `${A.batch_dir}/rej_${String(i).padStart(3, '0')}.json`
  )
}
const model = A.model || 'fable'

phase('Re-judge')
log(`re-judging ${batchPaths.length} rewrites (confident-voice rule) with ${model} at high effort`)
const judged = await parallel(
  batchPaths.map((p, i) => () =>
    agent(rejudgePrompt(p), {
      schema: SCHEMA,
      label: `rejudge#${i}`,
      phase: 'Re-judge',
      model,
      agentType: 'general-purpose',
      effort: 'high',
    })
  )
)
const verdicts = judged.filter(Boolean).flatMap((r) => r.verdicts || [])

const accepted = []
const kept_rejected = []
const seen = new Set()
for (const v of verdicts) {
  if (seen.has(v.id)) continue
  seen.add(v.id)
  const rw = (v.rewrite || '').trim()
  if (v.keep && rw) accepted.push({ id: v.id, rewrite: rw })
  else kept_rejected.push({ id: v.id, note: v.note || 'load-bearing — not a reflexive softener' })
}
log(`recovered ${accepted.length}, still rejected ${kept_rejected.length}`)
return {
  schema_version: 'rejudge-confident.v1',
  accepted,
  kept_rejected,
  stats: { total: batchPaths.length, recovered: accepted.length, kept_rejected: kept_rejected.length },
}
