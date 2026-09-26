"""Dispatch bounded native implementations after mathematical construction."""
from atlas.typing_ import TypeCheckError


def check_compatible(problem, genome):
    name = getattr(genome.backend, "kernel", "jax")
    if name == "jax":
        return
    if name in ("tv_fused128", "tv_fused256", "tv_recip256"):
        from atlas.backend.cuda_tv import check_compatible as check_tv
        return check_tv(problem, genome)
    if name in ("csr_thread128", "csr_warp128", "csr_warp256"):
        if genome.scheme != "pdhg" or problem.name != "lp_standard" or genome.backend.precision != "f64":
            raise TypeCheckError("Generated CSR kernels require standard LP PDHG and f64")
        return
    raise TypeCheckError(f"Unknown native kernel: {name}")


def install_kernel(problem, genome, built):
    check_compatible(problem, genome)
    name = getattr(genome.backend, "kernel", "jax")
    if name == "jax":
        return built
    if name.startswith("tv_"):
        from atlas.backend.cuda_tv import install_kernel as install
    else:
        from atlas.backend.cuda_lp import install_kernel as install
    return install(problem, genome, built)
