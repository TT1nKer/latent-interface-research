"""Pure text construction. The source prefix never depends on the later permutation."""
LETTERS = 'ABCD'


def make_case(row, shift):
    assert len(row['choices']) == 4 and 0 <= shift < 4
    order = [(i + shift) % 4 for i in range(4)]
    original = '\n'.join(f'{LETTERS[i]}. {x}' for i, x in enumerate(row['choices']))
    updated = '\n'.join(f'{LETTERS[i]}. {row["choices"][j]}' for i, j in enumerate(order))
    prefix = ('<|im_start|>user\nAnswer the following multiple-choice question.\n'
              + row['question'] + '\n' + original + '\n')
    ending = '\nReply with only the letter of the correct choice.<|im_end|>\n<|im_start|>assistant\nThe correct answer is'
    suffix = ('The choices below replace the earlier letter assignments. Use ONLY these updated letters.\n'
              + updated + ending)
    return dict(prefix=prefix, suffix=suffix, original=prefix + ending,
                order=order, answer_index=order.index(row['answer_index']),
                old_answer_index=row['answer_index'])


def validate(rows):
    assert len({r['id'] for r in rows}) == len(rows)
    for row in rows:
        cases = [make_case(row, k) for k in range(4)]
        assert len({c['prefix'] for c in cases}) == 1
        assert sorted(c['answer_index'] for c in cases) == list(range(4))
        for c in cases:
            assert row['choices'][c['order'][c['answer_index']]] == row['choices'][row['answer_index']]

