export const meta = {
  name: 'repair-terminology',
  description:
    'Repair rejected rewrites by restoring the exact defined terms (and any other flagged detail) from the original, then re-verify faithful + shorter + grammatical drop-in.',
  whenToUse: 'After the judge rejected rewrites for terminology drift and you want to salvage them.',
  phases: [{ title: 'Repair', detail: 'Fable restores dropped terms/details and re-verifies' }],
}

// Input (via args): { batch_dir, batch_count, model? }
// Each batch file is a 1-item array: {id, original_sentence, gemini_raw_response, reject_reason}.
// Returns { repaired:[{id, keep, rewrite}], failed:[{id, note}], stats }

const RUBRIC = `You are repairing a shortened rewrite of one sentence from an academic paper. A previous judge REJECTED the shortened version because it dropped or altered important details — most often a DEFINED TERM swapped for a looser synonym, but sometimes also a dropped hedge, quantifier, anchor, or claim.

You are given the ORIGINAL sentence (the ground truth: its wording, defined terms, hedges, and claims are correct and authoritative), Gemini's shortened REPLY, and the judge's REASON for rejection.

Your job: produce a shortened rewrite that keeps Gemini's concision and fluidity BUT restores EVERY important detail the reason flagged. Put back the exact defined term(s) from the original verbatim, and restore any dropped hedge, quantifier, anchor, or claim so the meaning matches the original exactly. Preserve all LaTeX markup verbatim (\\citep{...}, \\ref{...}, $...$, escaped literals). The rewrite must read as a single declarative sentence (or sentences) that drops in as a verbatim replacement — do NOT turn a statement into a command/imperative, and do NOT change the claim.

Then decide keep:
- keep=true ONLY IF the repaired sentence is (a) still meaningfully shorter than the original, (b) faithful to ALL important details — nothing dropped, added, or changed versus the original, and (c) a grammatical, same-speech-act, drop-in replacement.
- keep=false if restoring every flagged detail leaves it no shorter than the original, or you cannot preserve everything while shortening. Do not force it — a false keep ships a meaning change.

Put the repaired sentence in "rewrite" (even when keep=false, for logging). Return ONLY schema-valid JSON, one verdict per id, id echoed exactly.`

function repairPrompt(path) {
  return `${RUBRIC}

FIRST, use your Read tool to read the JSON array (one item) at:
  ${path}
The item has: id, original_sentence, gemini_raw_response, reject_reason. Repair it and return exactly one verdict.`
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
    (_, i) => `${A.batch_dir}/rep_${String(i).padStart(3, '0')}.json`
  )
}
const model = A.model || 'fable'

phase('Repair')
log(`repairing ${batchPaths.length} terminology-drift rewrites with ${model} at high effort`)
const judged = await parallel(
  batchPaths.map((p, i) => () =>
    agent(repairPrompt(p), {
      schema: SCHEMA,
      label: `repair#${i}`,
      phase: 'Repair',
      model,
      agentType: 'general-purpose',
      effort: 'high',
    })
  )
)
const verdicts = judged.filter(Boolean).flatMap((r) => r.verdicts || [])

const repaired = []
const failed = []
const seen = new Set()
for (const v of verdicts) {
  if (seen.has(v.id)) continue
  seen.add(v.id)
  const rw = (v.rewrite || '').trim()
  if (v.keep && rw) repaired.push({ id: v.id, rewrite: rw })
  else failed.push({ id: v.id, note: v.note || 'could not salvage while staying shorter + faithful' })
}
log(`repaired ${repaired.length}, still-failed ${failed.length}`)
return {
  schema_version: 'repair-terminology.v1',
  repaired,
  failed,
  stats: { total: batchPaths.length, repaired: repaired.length, failed: failed.length },
}
