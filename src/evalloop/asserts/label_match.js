// promptfoo custom javascript assertion for answer_type=label.
//
// Referenced from promptfooconfig.yaml as:
//   - type: javascript
//     value: file://../src/evalloop/asserts/label_match.js
//
// Signature per https://www.promptfoo.dev/docs/configuration/expected-outputs/javascript/ :
//   module.exports = (output, context) => GradingResult
// context.vars must contain `expected` (string) and `labels` (string[]),
// both injected by evalloop build.py (labels via defaultTest.vars, expected per-test).

// Must stay in lockstep with _normalize_label() in optimize.py (GEPA's
// in-process training metric). tests/test_label_normalization.py pins both
// implementations to the same fixture table -- extend that fixture when
// changing either side.
function normalizeLabel(value) {
  if (typeof value !== 'string') return '';
  let s = value.trim();
  // full-width alphanumeric/punctuation -> half-width (labels are Japanese but models
  // sometimes emit full-width punctuation/quotes around an otherwise correct label)
  s = s.replace(/[！-～]/g, (ch) => String.fromCharCode(ch.charCodeAt(0) - 0xfee0));
  // strip wrapping quotes/brackets a model might add
  s = s.replace(/^["'「『\[]+/, '').replace(/["'」』\]]+$/, '');
  // strip trailing sentence punctuation (both full-width and half-width forms)
  s = s.replace(/[。.、,]+$/g, '');
  return s.trim();
}

module.exports = (output, context) => {
  const vars = context.vars || {};
  const expectedRaw = vars.expected;
  const labelsRaw = vars.labels || [];

  if (typeof expectedRaw !== 'string') {
    return {
      pass: false,
      score: 0,
      reason: `context.vars.expected must be a string for answer_type=label, got: ${JSON.stringify(expectedRaw)}`,
    };
  }

  const expected = normalizeLabel(expectedRaw);
  const normOutput = normalizeLabel(output);

  if (normOutput === expected) {
    return {
      pass: true,
      score: 1,
      reason: `normalized output "${normOutput}" exactly matches expected "${expected}"`,
    };
  }

  // Fallback: the model wrapped the label in extra text. Accept it only if
  // exactly one label from the configured label set appears in the output,
  // to avoid rewarding an output that lists multiple/all candidate labels.
  const labels = Array.isArray(labelsRaw) ? labelsRaw : [];
  const normLabels = labels.map(normalizeLabel);
  const containedLabels = normLabels.filter((l) => l.length > 0 && normOutput.includes(l));
  const uniqueContained = Array.from(new Set(containedLabels));

  if (uniqueContained.length === 1) {
    const onlyMatch = uniqueContained[0];
    if (onlyMatch === expected) {
      return {
        pass: true,
        score: 1,
        reason: `output contains exactly one known label "${onlyMatch}", matches expected "${expected}"`,
      };
    }
    return {
      pass: false,
      score: 0,
      reason: `output contains exactly one known label "${onlyMatch}", but expected "${expected}"`,
    };
  }

  return {
    pass: false,
    score: 0,
    reason:
      `normalized output "${normOutput}" does not match expected "${expected}" ` +
      `(ambiguous or no known label found; labels=${JSON.stringify(labels)})`,
  };
};

// exposed for tests/test_label_normalization.py only; promptfoo calls the
// default export above
module.exports.normalizeLabel = normalizeLabel;
