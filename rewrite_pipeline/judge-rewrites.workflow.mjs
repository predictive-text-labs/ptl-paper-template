export const meta = {
  name: 'judge-rewrites',
  description:
    'Judge Gemini sentence rewrites with one simple Claude Fable check per sentence — does the shortened rewrite keep every important detail of the original? Keep on yes.',
  whenToUse:
    'After the Gemini fan-out has produced rewrites and they have been split into per-batch files.',
  phases: [
    { title: 'Judge', detail: 'Fable reads each batch file: does the rewrite keep all important details?' },
    { title: 'Assemble', detail: 'adopt every kept rewrite' },
  ],
}

// =============================================================================
// Input (via `args`):
//   args.batch_paths       string[] of JSON files, each an array of items
//                          {id, kind, original_sentence, gemini_raw_response}
//   args.total_candidates  int, for coverage reporting (optional)
//   args.judge_model       default 'fable'
//
// Each Fable agent READS one batch file (small — no 100k-token echo) and answers
// one question per sentence: does Gemini's shortened rewrite keep all the
// important details of the original? If yes, adopt. LaTeX-corruption safety
// (a dropped \citep / mangled $...$) is enforced deterministically by the Python
// apply stage (reinsert.validate_rewrite), not here — so this workflow is purely
// the meaning judgement and returns {id, rewrite}.
// Returns { accepted:[{id,rewrite}], rejected:[{id,reason}], stats }
// =============================================================================

const RUBRIC = `You check whether a shortened rewrite of one sentence from an academic paper keeps everything that matters.

For each item you get the ORIGINAL sentence and Gemini's REPLY. Gemini was asked to rewrite the sentence using fewer words while keeping it direct and fluid; its reply is usually the rewrite but often includes chatter, quotes, code fences, word-count notes, or several labelled options.

For each item:
1. CHOOSE THE SINGLE BEST rewrite from the reply — if Gemini gave several options (e.g. "Option 1", "Option 2"), pick the one that reads most directly and fluidly while keeping every important detail. Emit it exactly as it must appear in the LaTeX source: keep all LaTeX markup verbatim (\\citep{...}, \\ref{...}, $...$, escaped literals), and strip everything that is not the sentence itself — option labels, markdown, surrounding quotes, code fences, commentary, word-count notes. The value you put in "rewrite" must be a drop-in replacement that can be spliced into the .tex with no further editing. If Gemini refused, said the sentence was already fine, gave no rewrite, or merely repeated the original, set keep=false.
2. Answer ONE question about the sentence you chose: does it capture ALL the important details of the original? Important details include every claim, number, hedge (e.g. "may", "often"), quantifier (e.g. "all", "each", "at least"), named/defined term, citation, reference, and math. If all of them survive with the same meaning, set keep=true. If anything important is missing, added, or changed, set keep=false.

Always put the single chosen, drop-in-ready sentence in "rewrite" when you found one (even when keep=false, so it can be logged). Return ONLY schema-valid JSON, one verdict per id, and echo the ids exactly.`

function judgePrompt(path) {
  return `${RUBRIC}

FIRST, use your Read tool to read the JSON array of items at:
  ${path}
Each item has: id, kind, original_sentence, gemini_raw_response. The array normally holds a SINGLE sentence pair — give it your full attention. Judge every item in the array and return exactly one verdict per item, with the id echoed exactly.`
}

const JUDGE_SCHEMA = {
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
          reason: { type: ['string', 'null'] },
        },
      },
    },
  },
}

// ---- workflow ----

// args may arrive as a parsed object or as a JSON string depending on the host;
// tolerate both.
const A = typeof args === 'string' ? JSON.parse(args) : args || {}
// Either pass batch_paths explicitly, or a {batch_dir, batch_count} pair and let
// the script build the deterministic batch_NNNN.json names (avoids a huge arg).
let batchPaths = A.batch_paths || []
if (!batchPaths.length && A.batch_dir && A.batch_count) {
  batchPaths = Array.from(
    { length: A.batch_count },
    (_, i) => `${A.batch_dir}/batch_${String(i).padStart(4, '0')}.json`
  )
}
const judgeModel = A.judge_model || 'fable'
const totalCandidates = A.total_candidates || A.batch_count || 0

phase('Judge')
log(`judging ${batchPaths.length} batch files with ${judgeModel} at high effort`)
const judged = await parallel(
  batchPaths.map((p, i) => () =>
    agent(judgePrompt(p), {
      schema: JUDGE_SCHEMA,
      label: `judge#${i}`,
      phase: 'Judge',
      model: judgeModel,
      agentType: 'general-purpose',
      effort: 'high',
    })
  )
)
const verdicts = judged.filter(Boolean).flatMap((r) => r.verdicts || [])
log(`collected ${verdicts.length} verdicts${totalCandidates ? ` of ${totalCandidates} candidates` : ''}`)

// ---- Assemble: adopt every keep with a usable rewrite ----
phase('Assemble')
const accepted = []
const rejected = []
const seen = new Set()
for (const v of verdicts) {
  if (seen.has(v.id)) continue
  seen.add(v.id)
  if (!v.keep) {
    rejected.push({ id: v.id, reason: v.reason || 'missing_detail' })
    continue
  }
  const rw = (v.rewrite || '').trim()
  if (!rw) {
    rejected.push({ id: v.id, reason: 'no_extracted_rewrite' })
    continue
  }
  accepted.push({ id: v.id, rewrite: rw })
}

const stats = {
  batches: batchPaths.length,
  verdicts: verdicts.length,
  total_candidates: totalCandidates,
  no_verdict: totalCandidates ? Math.max(0, totalCandidates - seen.size) : null,
  accepted: accepted.length,
  rejected: rejected.length,
}
log(`accepted ${accepted.length}, rejected ${rejected.length}, no_verdict ${stats.no_verdict}`)

return { schema_version: 'judge-rewrites.v3', accepted, rejected, stats }
