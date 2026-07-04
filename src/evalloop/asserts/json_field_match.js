// promptfoo custom javascript assertion for answer_type=json.
// Runs *in addition to* an `is-json` assertion (which validates syntactic
// JSON + optional schema). This assertion checks the parsed JSON is deeply
// equal to context.vars.expected (a JSON-compatible object from golden.jsonl).
//
// Referenced from promptfooconfig.yaml as:
//   - type: javascript
//     value: file://../src/evalloop/asserts/json_field_match.js

function deepEqual(a, b) {
  if (a === b) return true;
  if (typeof a !== typeof b) return false;
  if (a === null || b === null) return a === b;
  if (typeof a !== 'object') return a === b;
  if (Array.isArray(a) !== Array.isArray(b)) return false;
  if (Array.isArray(a)) {
    if (a.length !== b.length) return false;
    return a.every((v, i) => deepEqual(v, b[i]));
  }
  const aKeys = Object.keys(a).sort();
  const bKeys = Object.keys(b).sort();
  if (aKeys.length !== bKeys.length || aKeys.some((k, i) => k !== bKeys[i])) return false;
  return aKeys.every((k) => deepEqual(a[k], b[k]));
}

module.exports = (output, context) => {
  const expected = (context.vars || {}).expected;

  let parsed;
  try {
    parsed = typeof output === 'string' ? JSON.parse(output) : output;
  } catch (e) {
    // is-json (run alongside this assert) already reports the syntax error;
    // this assert just needs to fail cleanly rather than throw.
    return { pass: false, score: 0, reason: `output is not valid JSON: ${e.message}` };
  }

  if (deepEqual(parsed, expected)) {
    return { pass: true, score: 1, reason: 'parsed JSON deep-equals expected' };
  }

  return {
    pass: false,
    score: 0,
    reason: `parsed JSON does not match expected. got=${JSON.stringify(parsed)} expected=${JSON.stringify(expected)}`,
  };
};
