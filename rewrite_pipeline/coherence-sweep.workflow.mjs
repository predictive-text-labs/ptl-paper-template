export const meta = {
  name: 'coherence-sweep',
  description:
    'Pre-apply gate: check every paragraph the accepted rewrites would change for cross-sentence coherence breaks introduced by per-sentence shortening (dangling pro-verbs, orphaned pronouns, broken transitions).',
  whenToUse:
    'After `rewrite pairs` and before `rewrite apply --apply`. Save the full return value verbatim to <workdir>/coherence_fixes.json — apply refuses without a sign-off fingerprinted to the current accepted set.',
  phases: [{ title: 'Sweep', detail: 'one Fable agent per changed old/new paragraph pair' }],
}

// Input (via args, printed ready-to-use by `rewrite pairs`):
//   { pair_dir, pair_count, accepted_sha256, model? }
// Each pair file is { old, new } — one paragraph before/after shortening.
// Returns { schema_version, accepted_sha256, fixes:[{pair, quote, problem,
// replacement}], stats } — the exact shape `rewrite apply` consumes from
// coherence_fixes.json. Fix application safety (quote must match exactly once,
// no newlines, Class-A LaTeX tokens preserved) is enforced deterministically
// by Python (coherence.apply_fixes), not here.

const RUBRIC = `You are proofreading ONE paragraph of an academic paper after its sentences were shortened INDIVIDUALLY (each sentence rewritten in isolation, without seeing its neighbors). Your ONLY job is to find places where that isolation BROKE the paragraph — where the NEW paragraph no longer works as connected English even though each sentence was fine on its own.

You get the OLD paragraph (the original, coherent ground truth) and the NEW paragraph (after shortening). Read both carefully, sentence by sentence, and check the NEW paragraph for exactly these defect classes:

1. DANGLING PRO-FORMS: a pro-verb or ellipsis whose antecedent was reworded away. Example actually found in this paper: OLD said "sampling frequency does not alter the result. However, X, Y, and Z still do." — the rewrite changed the first sentence to "makes sampling frequency irrelevant", leaving "still do" with no verb to stand in for. Watch for "do/does/did (too/so/not)", "so is/are", "the same", "neither/nor", "still do".
2. ORPHANED REFERENCES: "this", "these", "that", "such", "it", "the former/latter", "both", or a definite noun phrase ("the pattern", "this risk") whose antecedent in the previous sentence was dropped or reworded so the reference no longer resolves — including singular/plural mismatches ("Traders ... The trader") and enumeration labels a later sentence still cites ("failed one of (i)--(iv)") after the labels were dropped.
3. BROKEN TRANSITIONS: "However", "Instead", "By contrast", "Therefore", "Thus" where the rewrite removed the thing being contrasted with or concluded from, making the connective illogical.
4. GRAMMAR/FLOW BREAKS AT SENTENCE SEAMS: accidental fragments, subject drops, a sentence rewritten into an imperative amid declaratives, tense clashes between adjacent rewritten sentences, or the same distinctive word now repeated awkwardly in back-to-back sentences.

Do NOT flag: the shortening itself, dropped hedges, tone, style preferences, or anything that reads fine — those decisions are final. Do NOT flag issues that already exist verbatim in the OLD paragraph. Only report defects the rewrite INTRODUCED that make the NEW text read as broken or ambiguous English. Be precise and skeptical: most paragraphs are fine; an empty issues list is the expected answer.

For each real issue, give: the exact quote from the NEW paragraph (verbatim substring), what broke, and a MINIMAL replacement — the smallest edit that repairs coherence while keeping the shortened style and preserving all LaTeX markup (\\citep{...}, \\ref{...}, $...$, footnotes) byte-for-byte. The replacement must be a verbatim drop-in for the quoted substring, and the quote should be long enough to be unique within the paper. The paragraph may contain hard line breaks; keep the quote and replacement within a single source line (no newlines in either).`

function sweepPrompt(path) {
  return `${RUBRIC}

FIRST, use your Read tool to read the JSON object at:
  ${path}
It has fields "old" and "new" (one paragraph each). Check the NEW paragraph as instructed and return your verdict.`
}

const SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['issues'],
  properties: {
    issues: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['quote', 'problem', 'replacement'],
        properties: {
          quote: { type: 'string', description: 'verbatim substring of the NEW paragraph that is broken' },
          problem: { type: 'string' },
          replacement: { type: 'string', description: 'verbatim drop-in replacement for the quoted substring' },
        },
      },
    },
  },
}

const A = typeof args === 'string' ? JSON.parse(args) : args || {}
const model = A.model || 'fable'
const paths = Array.from(
  { length: A.pair_count },
  (_, i) => `${A.pair_dir}/pair_${String(i).padStart(3, '0')}.json`
)

phase('Sweep')
log(`sweeping ${paths.length} rewritten paragraphs for cross-sentence coherence breaks`)
const results = await parallel(
  paths.map((p, i) => () =>
    agent(sweepPrompt(p), {
      schema: SCHEMA,
      label: `pair#${String(i).padStart(3, '0')}`,
      phase: 'Sweep',
      model,
      agentType: 'general-purpose',
      effort: 'high',
    }).then((r) => ({ pair: i, issues: (r && r.issues) || [] }))
  )
)
const fixes = results
  .filter(Boolean)
  .flatMap((r) => r.issues.map((it) => ({ pair: r.pair, ...it })))
log(`${fixes.length} coherence fixes across ${paths.length} paragraphs`)
return {
  schema_version: 'coherence-sweep.v1',
  accepted_sha256: A.accepted_sha256 || null,
  fixes,
  stats: { pairs: paths.length, fixes: fixes.length },
}
