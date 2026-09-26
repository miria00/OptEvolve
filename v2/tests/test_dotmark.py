"""DOTmark loader and the reduction form of the transport operator."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("m,n", [(3, 4), (7, 5), (16, 16)])
def test_reduction_operator_equals_incidence(m, n):
    from atlas.bench.dotmark import TransportLinOp
    from atlas.bench.lp_cell import IncidenceLinOp
    from atlas.bench.ot import bipartite_arcs
    tail, head, nn = bipartite_arcs(m, n)
    inc = IncidenceLinOp(tail=jnp.asarray(tail), head=jnp.asarray(head), n_nodes=nn, norm_bound=1.0)
    red = TransportLinOp(m=m, n=n, norm_bound=1.0)
    rng = np.random.default_rng(m * 100 + n)
    x, y = jnp.asarray(rng.normal(size=m * n)), jnp.asarray(rng.normal(size=m + n))
    assert np.allclose(red.forward(x), inc.forward(x), atol=1e-12, rtol=1e-12)
    assert np.allclose(red.adjoint(y), inc.adjoint(y), atol=1e-12, rtol=1e-12)
    assert abs(float(red.forward(x) @ y - x @ red.adjoint(y))) < 1e-10


def test_dotmark_composite_both_operators():
    from atlas.bench import dotmark
    if not (dotmark.ROOT / "Data" / "Shapes" / "data32_1001.csv").exists():
        pytest.skip("DOTmark not downloaded")
    pi = dotmark.composite("Shapes", 32, 1001, 1002, operator="incidence")
    pr = dotmark.composite("Shapes", 32, 1001, 1002, operator="reduction")
    assert pi.data["simplex_mass"] == 1.0 and pi.shape == (1024 * 1024,)
    x = jnp.asarray(np.random.default_rng(0).random(1024 * 1024))
    assert np.allclose(pi.L.forward(x), pr.L.forward(x), rtol=1e-12, atol=1e-9)
    assert float(pi.L.norm_bound) == pytest.approx(1.02 * np.sqrt(2048), rel=1e-12)
