// Invoked by tests/test_label_normalization.py: runs every case in the shared
// fixture through asserts/label_match.js's normalizeLabel and prints a
// {name: normalized} JSON object on stdout.
//
//   node js_normalize_runner.js <fixture.json> <label_match.js>
const fs = require('fs');
const path = require('path');

const [fixturePath, labelMatchPath] = process.argv.slice(2);
const { cases } = JSON.parse(fs.readFileSync(fixturePath, 'utf8'));
const { normalizeLabel } = require(path.resolve(labelMatchPath));

const results = {};
for (const c of cases) {
  results[c.name] = normalizeLabel(c.input);
}
process.stdout.write(JSON.stringify(results));
