import random
import unittest
import numpy as np
from bandit_policy import LinUCB,context,mutate,reward
from recovery_space import DEFAULT,check

class PolicyTests(unittest.TestCase):
    def test_updates_prefer_a_successful_action_in_fixed_context(self):
        p=LinUCB(seed=1,alpha=.1);x=np.ones(9)
        for _ in range(8):p.update(x,'model',1.)
        self.assertEqual(p.choose(x)[0],'model')
        q=LinUCB.restore(p.dump(),1)
        self.assertEqual(q.choose(x)[0],'model')

    def test_generic_mutations_remain_legal(self):
        r=random.Random(3)
        for _ in range(100):
            s,gene=mutate(DEFAULT,r)
            check(s)
            self.assertNotEqual(s,DEFAULT)

    def test_reward_requires_full_certificate_for_high_value(self):
        good=[{'accepted':True,'elapsed_s':2},{'accepted':True,'elapsed_s':4}]
        bad=[{'accepted':True,'elapsed_s':2},{'accepted':False,'fraction_certified':.999}]
        self.assertGreater(reward(good,120),reward(bad,120))

if __name__=='__main__':unittest.main()
