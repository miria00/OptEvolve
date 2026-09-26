import unittest
import numpy as np
import jax
jax.config.update('jax_enable_x64',True)
from typed_space import DEFAULT,compile_spec,canonical_spec,check

class TypedSpaceTests(unittest.TestCase):
    def test_bad_composition_and_relaxation_rejected(self):
        for change in ({'formulation':'resolvent'}, {'relaxation':2.0}, {'state':'inverse'}):
            with self.assertRaises(ValueError):check({**DEFAULT,**change})

    def test_inactive_parameters_deduplicate(self):
        self.assertEqual(canonical_spec(DEFAULT),canonical_spec({**DEFAULT,'resolvent_step':100.,'relaxation':1.9}))

    def test_factor_policies_preserve_drs_operator(self):
        rng=np.random.default_rng(42)
        D=rng.normal(size=(4,6));Y=rng.normal(size=(4,3))
        Q=np.eye(6)+D.T@D;B=D.T@Y;z=np.full((6,3),1/6)
        def proj(x):
            u=np.sort(x,axis=0)[::-1];cs=np.cumsum(u,axis=0)-1
            r=np.sum(u-cs/np.arange(1,7)[:,None]>0,axis=0)-1
            return np.maximum(x-cs[r,np.arange(3)]/(r+1),0)
        for i in range(7):
            x=proj(z);z=z+1.5*(np.linalg.solve(Q,2*x-z+B)-x)
        expected=proj(z)
        for factor in ['inverse','cholesky','recompute','host_inverse']:
            s={**DEFAULT,'algorithm':'drs','formulation':'resolvent','state':factor,'relaxation':1.5}
            code,certificate=compile_spec(s);ns={};exec(code,ns)
            options={'dtype':'float64','tol':1e-4,'unroll':1}
            p=ns['formulate'](D,Y,options);state=ns['initialize'](p,options)
            state=ns['advance'](p,state,7,options);got=ns['solution'](p,state)
            np.testing.assert_allclose(got,expected,atol=2e-13,rtol=2e-13)

    def test_handoff_is_finite_and_changes_executed_engine(self):
        s={**DEFAULT,'handoff':1,'state':'inverse'};code,_=compile_spec(s);ns={};exec(code,ns)
        options={'dtype':'float64','tol':1e-4,'unroll':1}
        p=ns['formulate'](np.eye(3),np.eye(3),options);state=ns['initialize'](p,options)
        self.assertEqual(p['phase'],'fbs')
        state=ns['advance'](p,state,1,options)
        p,state=ns['handoff'](p,state,{},options)
        self.assertEqual(p['phase'],'drs');self.assertIsNotNone(p['factor'])
        state=ns['advance'](p,state,4,options)
        self.assertTrue(np.isfinite(ns['solution'](p,state)).all())

    def test_search_progress_cannot_outrank_certified_success(self):
        from typed_campaign import parent_rank
        def row(gap,t,solved=1):
            return {'score':{'solved':solved,'total':2,'penalized_geomean_s':t},
                    'rows':[{'accepted':False,'worst_gap':gap,'feasibility':1e-14}]}
        self.assertGreater(parent_rank(row(1.01e-4,10),True),parent_rank(row(.1,1),True))
        self.assertGreater(parent_rank(row(0,29,2),True),parent_rank(row(1.01e-4,1),True))

    def test_retirement_preserves_checked_float64_output(self):
        import jax.numpy as jnp
        s={**DEFAULT,'contract':'float32','restrict':'retire'}
        code,_=compile_spec(s);ns={};exec(code,ns)
        options={'dtype':'float32','tol':1e-4,'unroll':1}
        Y=np.array([[.2,.2,1.],[.3,.3,0.],[.5,.5,0.]])
        p=ns['formulate'](np.eye(3),Y,options)
        raw=jnp.array([[.2,.2,.33333334],[.3,.3,.33333334],[.5,.5,.33333334]],dtype=jnp.float32)
        state={'x':raw,'z':raw,'chunks':1}
        checked=ns['solution'](p,state).copy()
        p,state=ns['restrict'](p,state,options)
        self.assertEqual(p['alive'].tolist(),[2])
        np.testing.assert_array_equal(ns['solution'](p,state)[:,:2],checked[:,:2])

if __name__=='__main__':unittest.main()
