// Invoked by tests/test_label_normalization.py: exercises the FULL
// label_match.js assertion with `labels` passed as a JSON-encoded string --
// the contract build.py relies on (a real array would be expanded into a
// promptfoo test matrix, multiplying every case by len(labels)).
//
//   node js_label_match_verdict_runner.js <label_match.js>
const path = require('path');

const [assertPath] = process.argv.slice(2);
const assertion = require(path.resolve(assertPath));

const labelsJson = JSON.stringify(['契約照会', '障害報告', '機能要望', 'その他']);
const run = (output, expected) => assertion(output, { vars: { expected, labels: labelsJson } }).pass;

const results = {
  'exact-match': run('契約照会', '契約照会'),
  'contained-single-label': run('回答: 障害報告 です', '障害報告'),
  'wrong-label': run('その他', '契約照会'),
  'ambiguous-two-labels': run('契約照会か障害報告', '契約照会'),
};
process.stdout.write(JSON.stringify(results));
