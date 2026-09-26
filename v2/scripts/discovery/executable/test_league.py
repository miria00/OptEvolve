import unittest
from league import compare,StopRule


class LeagueTests(unittest.TestCase):
    def rows(self,time,accepted=True):
        return [{'case':c,'repetition':r,'accepted':accepted,'elapsed_s':time}
                for c in ['easy','hard'] for r in range(3)]

    def test_promotion_and_noisy_tie(self):
        self.assertTrue(compare(self.rows(4),self.rows(8),cap_s=30)['promote'])
        self.assertFalse(compare(self.rows(7.9),self.rows(8),cap_s=30)['promote'])

    def test_one_failed_column_case_prevents_promotion(self):
        rows=self.rows(1);rows[0]['accepted']=False
        self.assertFalse(compare(rows,self.rows(8),cap_s=30)['promote'])

    def test_average_win_cannot_hide_large_case_regression(self):
        rows=self.rows(1)
        for r in rows:
            if r['case']=='hard':r['elapsed_s']=15
        result=compare(rows,self.rows(8),cap_s=30)
        self.assertGreater(result['speedup_geomean'],1)
        self.assertFalse(result['promote'])

    def test_pairing_and_stopping(self):
        with self.assertRaises(ValueError):compare(self.rows(1)[:-1],self.rows(8),cap_s=30)
        rule=StopRule(20,3600,8)
        self.assertIsNone(rule.reason(evaluations=3,elapsed_s=15,since_promotion=2))
        self.assertIn('patience',rule.reason(evaluations=10,elapsed_s=15,since_promotion=8))

    def test_controller_promotes_successive_champions_and_stops(self):
        import contextlib,io,json,sys,tempfile
        from pathlib import Path
        from unittest.mock import patch
        import league_campaign as runner
        seed=(runner.C.HERE/'seed.py').read_text()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data=root/'data';data.mkdir()
            (data/'manifest.json').write_text('{}')
            baseline=root/'baseline.py';baseline.write_text(seed.replace('128','256'))
            calls=[]
            def propose(prompt,endpoint,rngseed,out,parent):
                calls.append(prompt)
                return {'code':seed.replace('128',str(64//len(calls))),'rationale':'test'},{}
            def evaluate(slot,source,cases,out,cap):
                # Baseline, initial program, then two successive improvements.
                name=source.name
                t=8 if name=='baseline.py' else 20 if name=='proposer_seed.py' else 4 if source.parent.name=='p001' else 2
                return [{'case':c,'accepted':True,'elapsed_s':t,'worst_gap':1e-6} for c in cases]
            argv=['league_campaign.py','--baselines','baseline='+str(baseline),
                  '--data',str(data),'--out',str(root/'run'),'--remote-root','/unused','--slot','0',
                  '--evaluations','2','--patience','8']
            previous=runner.C.REMOTE
            try:
                with patch.object(sys,'argv',argv),patch.object(runner.C,'deploy'),patch.object(runner.C,'llm',side_effect=propose),patch.object(runner.C,'evaluate_on_slot',side_effect=evaluate),contextlib.redirect_stdout(io.StringIO()):
                    runner.main()
            finally:runner.C.REMOTE=previous
            state=json.loads((root/'run/status.json').read_text())
            self.assertTrue(state['complete'])
            self.assertEqual(len(state['promotions']),2)
            self.assertEqual(state['evaluations'],2)
            self.assertEqual(calls[0],calls[1])
            self.assertIn('budget',state['termination'])


if __name__=='__main__':unittest.main()
