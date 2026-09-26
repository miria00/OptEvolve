import unittest
import numpy as np
import jax
jax.config.update('jax_enable_x64',True)
from recovery_space import DEFAULT,compile_spec,audit

class RecoveryTests(unittest.TestCase):
    def test_admm_drs_state_conjugacy(self):
        rng=np.random.default_rng(6621)
        D=rng.normal(size=(5,8));Y=rng.normal(size=(5,4))
        options={'dtype':'float64','tol':1e-4,'unroll':1}
        for eta in (.1,1.,10.):
            for relaxation in (.5,1.,1.6,1.9):
                runs=[]
                for algorithm in ('admm','drs'):
                    source,_=compile_spec({**DEFAULT,'algorithm':algorithm,
                        'formulation':'resolvent','state':'inverse',
                        'resolvent_step':eta,'relaxation':relaxation})
                    ns={};exec(source,ns)
                    p=ns['formulate'](D,Y,options)
                    runs.append((ns,p,ns['initialize'](p,options)))
                for _ in range(25):
                    runs=[(ns,p,ns['advance'](p,v,1,options)) for ns,p,v in runs]
                    a,b=runs[0][2],runs[1][2]
                    np.testing.assert_allclose(a['x'],b['x'],atol=1e-12,rtol=1e-12)
                    np.testing.assert_allclose(a['x']+a['u'],b['z'],atol=1e-12,rtol=1e-12)

    def test_admm_matches_independent_numpy_steps(self):
        rng=np.random.default_rng(621);D=rng.normal(size=(4,6));Y=rng.normal(size=(4,3))
        from standard_reference_examples import simplex_projection as simplex
        for eta in (.1,1.,10.):
            z=np.full((6,3),1/6);u=np.zeros_like(z);H=D.T@D;B=D.T@Y
            for _ in range(7):
                x=np.linalg.solve(np.eye(6)+eta*H,z-u+eta*B)
                xa=1.6*x-.6*z;zn=simplex(xa+u);u+=xa-zn;z=zn
            for factor in ('inverse','cholesky','recompute','host_inverse'):
                s={**DEFAULT,'algorithm':'admm','formulation':'resolvent','state':factor,
                   'resolvent_step':eta,'relaxation':1.6}
                source,_=compile_spec(s);ns={};exec(source,ns)
                opts={'dtype':'float64','tol':1e-4,'unroll':1}
                p=ns['formulate'](D,Y,opts);v=ns['initialize'](p,opts)
                v=ns['advance'](p,v,7,opts)
                np.testing.assert_allclose(ns['solution'](p,v),z,atol=1e-12,rtol=1e-12)
                self.assertEqual(v['actual_iterations'],7)

    def test_internal_retirement_saves_unseen_columns(self):
        import jax.numpy as jnp
        from judge import certify
        Y=np.array([[.2,.2,1.],[.3,.3,0.],[.5,.5,0.]])
        s={**DEFAULT,'algorithm':'admm','formulation':'resolvent','state':'inverse',
           'restrict':'retire','certificate':'device_screened'}
        source,_=compile_spec(s);ns={};exec(source,ns)
        opts={'dtype':'float64','tol':1e-4,'unroll':1}
        p=ns['formulate'](np.eye(3),Y,opts);v=ns['initialize'](p,opts)
        v['x']=jnp.asarray(np.column_stack((Y[:,:2],np.full(3,1/3))))
        v['z']=v['x']
        v=ns['retire'](p,v,np.array([0,0,1]),opts)
        self.assertEqual(p['alive'].tolist(),[2])
        v=ns['advance'](p,v,128,opts)
        full=ns['solution'](p,v)
        np.testing.assert_array_equal(full[:,:2],Y[:,:2])
        self.assertTrue(certify(np.eye(3),Y,full,1e-4)['accepted'])

    def test_configuration_labels_cannot_satisfy_execution_audit(self):
        s={**DEFAULT,'algorithm':'admm','formulation':'resolvent','state':'inverse',
           'restrict':'retire','certificate':'device_screened'}
        source,_=compile_spec(s)
        self.assertFalse(audit(s,source,[{'accepted':True}])['method_recovered'])
        self.assertFalse(audit(s,source,[])['method_recovered'])

if __name__=='__main__':unittest.main()
