"""Small synthetic checks; never use held-out Karel rollouts for reducer development."""
import json
import tempfile
import unittest
from pathlib import Path

from analyze_karel_unaligned import CENTROID, DOMAINS, METHODS, RETRIEVAL, SIMPLE, analyze, load_rows


class PairedReducerTest(unittest.TestCase):
    def make_row(self, domain, record, target):
        code = 'aa' if record in ('x', 'y') else 'bb'
        row = dict(domain=domain, code_id=code, program_id=record, target_id=target,
                   reference_action_count=int(target > 0), support_unique_traces=1 + int(record == 'x'), methods={})
        for name in METHODS:
            wins = (name in RETRIEVAL and record == 'x') or (name == RETRIEVAL[1] and record == 'y')
            wins |= (name == SIMPLE[0] and code == 'aa') or (name == SIMPLE[1] and code == 'bb')
            row['methods'][name] = dict(success=wins, termination='stop' if wins else 'crash')
        return row

    def test_record_code_weighting_seed_pairing_and_bootstrap_reselection(self):
        rows = [self.make_row(d, p, t) for d in DOMAINS for p in ('x', 'y', 'z') for t in range(6)]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'synthetic.jsonl'
            path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
            result = analyze(path)
            external = result['domains']['external']
            all_rows = external['all']
            self.assertEqual(all_rows['counts'], dict(targets=18, program_records=3, distinct_codes=2))
            self.assertEqual(all_rows['contrasts']['highest_scoring_simple_methods'], list(SIMPLE[:2]))
            self.assertAlmostEqual(all_rows['estimates']['retrieval_seed_average']['endpoint_success']['mean'], 1/3)
            self.assertAlmostEqual(all_rows['estimates']['retrieval_seed_average']['crash'], 2/3)
            d = all_rows['contrasts']['retrieval_minus_strongest_simple']
            self.assertAlmostEqual(d['mean'], -1/6)
            # Drawing either code twice makes a simple comparator perfect; failing
            # to reselect would put a positive delta in the upper tail.
            self.assertAlmostEqual(d['percentile_95'][1], -1/6)
            self.assertEqual(external['variable_support']['counts']['distinct_codes'], 1)
            self.assertEqual(external['zero_reference_actions']['counts']['targets'], 3)
            for method in (*SIMPLE, 'centroid_seed_average', 'retrieval_scrambled_17'):
                expected_delta = 1/3 - all_rows['estimates'][method]['endpoint_success']['mean']
                if method == 'retrieval_scrambled_17':
                    expected_delta = all_rows['estimates'][RETRIEVAL[0]]['endpoint_success']['mean']
                rates = all_rows['paired_discordance'][method]
                self.assertAlmostEqual(rates['retrieval_only_success'] - rates['comparator_only_success'], expected_delta)
            self.assertFalse(result['gate']['practical_advantage'])
            path.write_text(''.join(json.dumps(row) + '\n' for row in rows[:-1]))
            with self.assertRaisesRegex(ValueError, 'incomplete targets'):
                load_rows(path)
            bad = self.make_row(DOMAINS[0], 'x', 0)
            del bad['methods'][CENTROID[0]]
            path.write_text(json.dumps(bad) + '\n')
            with self.assertRaisesRegex(ValueError, 'wrong method set'):
                load_rows(path)


if __name__ == '__main__':
    unittest.main()
