// Single-pass FP64 TV stencil. Reads only the old state, with no cross-block
// synchronization or input/output aliasing. Recompute neighboring primal
// values to remove the primal-to-dual intermediate and fuse relaxation.
#include <cuda_runtime.h>
#include <climits>
#include "xla/ffi/api/ffi.h"
namespace ffi = xla::ffi;

template<bool Proximal, bool Reciprocal>
__device__ double Primal(int p, int h, int w, const double* x,
                        const double* u, const double* y, double t, double inv) {
  int i = p / w, j = p % w, n = h * w;
  double v0 = (i > 0 ? u[p-w] : 0.) - (i+1 < h ? u[p] : 0.);
  double v1 = (j > 0 ? u[n+p-1] : 0.) - (j+1 < w ? u[n+p] : 0.);
  double adj = v0 + v1;
  if constexpr (Proximal) {
    double numerator = (x[p] - t*adj) + t*y[p];
    if constexpr (Reciprocal) return numerator * inv;
    return numerator / (1.+t);
  }
  return (x[p] - t*(x[p]-y[p])) - t*adj;
}

__device__ double Clip(double v, double lam) {
  return isnan(v) ? v : fmin(fmax(v, -lam), lam);
}

template<int Threads, bool Proximal, bool Reciprocal>
__global__ void TVUpdate(int h, int w, const double* x, const double* u,
                         const double* y, const double* t_, const double* s_,
                         const double* lam_, const double* inv_, double rho, double* xp, double* up) {
  int p = blockIdx.x * Threads + threadIdx.x, n = h*w;
  if (p >= n) return;
  int i = p/w, j = p%w;
  double t = t_[0], s = s_[0], lam = lam_[0], inv = inv_[0];
  double primal = Primal<Proximal,Reciprocal>(p,h,w,x,u,y,t,inv);
  double bar = 2.*primal-x[p];
  double d0 = i+1 < h ? (2.*Primal<Proximal,Reciprocal>(p+w,h,w,x,u,y,t,inv)-x[p+w])-bar : 0.;
  double d1 = j+1 < w ? (2.*Primal<Proximal,Reciprocal>(p+1,h,w,x,u,y,t,inv)-x[p+1])-bar : 0.;
  double a = Clip(u[p]+s*d0, lam), b = Clip(u[n+p]+s*d1, lam);
  xp[p] = rho == 1. ? primal : (1.-rho)*x[p]+rho*primal;
  up[p] = rho == 1. ? a : (1.-rho)*u[p]+rho*a;
  up[n+p] = rho == 1. ? b : (1.-rho)*u[n+p]+rho*b;
}

template<int Threads, bool Reciprocal>
ffi::Error LaunchTV(cudaStream_t stream, int64_t proximal, double rho,
                    ffi::Buffer<ffi::F64> x, ffi::Buffer<ffi::F64> u,
                    ffi::Buffer<ffi::F64> y, ffi::Buffer<ffi::F64> t,
                    ffi::Buffer<ffi::F64> s, ffi::Buffer<ffi::F64> lam, ffi::Buffer<ffi::F64> inv,
                    ffi::ResultBuffer<ffi::F64> xp, ffi::ResultBuffer<ffi::F64> up) {
  auto xd = x.dimensions(), ud = u.dimensions();
  if (xd.size() != 2 || ud.size() != 3 || xd[0] < 1 || xd[1] < 1 ||
      xd[0] > INT_MAX || xd[1] > INT_MAX || x.element_count() > INT_MAX/2 ||
      ud[0] != 2 || ud[1] != xd[0] || ud[2] != xd[1] ||
      y.element_count() != x.element_count() ||
      xp->element_count() != x.element_count() || up->element_count() != u.element_count() ||
      t.element_count() != 1 || s.element_count() != 1 || lam.element_count() != 1 || inv.element_count() != 1 ||
      (proximal != 0 && proximal != 1) || !isfinite(rho))
    return ffi::Error::InvalidArgument("Invalid TV stencil dimensions or attributes");
  int h = xd[0], w = xd[1], blocks = (h*w+Threads-1)/Threads;
  if (proximal)
    TVUpdate<Threads,true,Reciprocal><<<blocks,Threads,0,stream>>>(h,w,x.typed_data(),u.typed_data(),
        y.typed_data(),t.typed_data(),s.typed_data(),lam.typed_data(),inv.typed_data(),rho,xp->typed_data(),up->typed_data());
  else
    TVUpdate<Threads,false,Reciprocal><<<blocks,Threads,0,stream>>>(h,w,x.typed_data(),u.typed_data(),
        y.typed_data(),t.typed_data(),s.typed_data(),lam.typed_data(),inv.typed_data(),rho,xp->typed_data(),up->typed_data());
  auto error = cudaPeekAtLastError();
  return error == cudaSuccess ? ffi::Error::Success() : ffi::Error::Internal(cudaGetErrorString(error));
}

// HANDLERS_GENERATED_HERE
