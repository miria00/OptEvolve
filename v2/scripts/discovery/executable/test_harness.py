import unittest
from pathlib import Path
import numpy as np
from judge import certify, aggregate
from space import validate_source, source_id, HOOKS, apply_function_patch

class HarnessTests(unittest.TestCase):
    def test_judge_uses_full_batch_and_feasibility(self):
        D = np.eye(3)
        Y = np.eye(3)
        self.assertTrue(certify(D, Y, Y)['accepted'])
        bad = Y.copy(); bad[:, -1] = 1 / 3
        self.assertFalse(certify(D, Y, bad)['accepted'])
        self.assertFalse(certify(D, Y, Y + 1)['accepted'])
        self.assertFalse(certify(D, Y, np.full_like(Y, np.nan))['accepted'])
        self.assertFalse(certify(D, Y, Y[:, :1])['accepted'])

    def test_failure_does_not_win_by_returning_early(self):
        result = aggregate([{'accepted':False, 'elapsed_s':.0001},
                            {'accepted':True, 'elapsed_s':2}], 30)
        self.assertEqual(result['times'], [30, 2])

    def test_eight_executable_hooks_and_source_guard(self):
        seed = Path(__file__).with_name('seed.py').read_text()
        validate_source(seed)
        self.assertEqual(len(HOOKS), 8)
        self.assertEqual(source_id(seed), source_id(seed + '\n# whitespace only\n'))
        for bad in ['import os\n', 'import scipy.optimize\n',
                    'from scipy import optimize\n', 'np.sum = lambda x: 0\n', 'open("test")\n']:
            with self.assertRaises(ValueError):
                validate_source(bad + seed)

    def test_function_patch_preserves_other_hooks(self):
        seed = Path(__file__).with_name('seed.py').read_text()
        changed = apply_function_patch(seed, {'width':"def width(meta):\n return {'chunk':32,'unroll':2}"})
        from space import layer_ids
        before, after = layer_ids(seed), layer_ids(changed)
        self.assertNotEqual(before['8'], after['8'])
        for layer in range(1,8):
            self.assertEqual(before[str(layer)], after[str(layer)])
        with self.assertRaises(ValueError):
            apply_function_patch(seed, {'width': 'def admit(meta): return "gpu"'})

    def test_executable_teaching_examples_reach_independent_certificate(self):
        from standard_reference_examples import textbook_admm,textbook_drs
        D=np.diag([1.,2.,3.]);Y=np.array([[.5,-.2],[.3,1.2],[.8,.1]])
        H=D.T@D;b=D.T@Y;z=np.full((3,2),1/3)
        for x in [textbook_admm(H,b,z,1.,200),textbook_drs(H,b,z,1.,1.5,200)]:
            self.assertTrue(certify(D,Y,x)['accepted'])

if __name__ == '__main__':
    unittest.main()
