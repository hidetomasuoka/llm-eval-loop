// Invoked by tests/test_json_field_match.py: runs every case in the shared
// fixture through asserts/json_field_match.js (the full promptfoo assertion)
// and prints a {name: pass} JSON object on stdout.
//
//   node js_json_field_match_runner.js <fixture.json> <json_field_match.js>
const fs = require('fs');
const path = require('path');

const [fixturePath, assertPath] = process.argv.slice(2);
const { cases } = JSON.parse(fs.readFileSync(fixturePath, 'utf8'));
const assertion = require(path.resolve(assertPath));

const results = {};
for (const c of cases) {
  const verdict = assertion(c.output, { vars: { expected: c.expected } });
  results[c.name] = verdict.pass;
}
process.stdout.write(JSON.stringify(results));
