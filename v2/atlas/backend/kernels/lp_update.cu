// Bounded code-generation template. No fast math or inter-block dependencies.
// Each specialization fuses one CSR matvec with its PDHG vector update.
#include <cuda_runtime.h>
#include "xla/ffi/api/ffi.h"
namespace ffi = xla::ffi;

template<int Lanes, int Threads, bool Dual>
__global__ void Update(int rows, const double* a, const int* col,
                       const int* ptr, const double* v, const double* old,
                       const double* base, const double* cost,
                       const double* step, double* out) {
  int tid = blockIdx.x * Threads + threadIdx.x;
  int row = tid / Lanes;
  int lane = threadIdx.x % Lanes;
  if (row >= rows) return;
  double sum = 0.;
  for (int j = ptr[row] + lane; j < ptr[row + 1]; j += Lanes) {
    int c = col[j];
    double val = Dual ? 2. * v[c] - old[c] : v[c];
    sum += a[j] * val;
  }
  if constexpr (Lanes == 32) {
    for (int offset = 16; offset; offset /= 2)
      sum += __shfl_down_sync(0xffffffff, sum, offset);
  }
  if (lane == 0) {
    // Preserve reference operation ordering, including separate affine tilt.
    double value = Dual ? (base[row] + step[0] * sum) - step[0] * cost[row]
                        : (base[row] - step[0] * sum) - step[0] * cost[row];
    out[row] = Dual ? value : (value < 0. ? 0. : value);
  }
}

template<int Lanes, int Threads>
ffi::Error Launch(cudaStream_t stream, int64_t dual,
                  ffi::Buffer<ffi::F64> a, ffi::Buffer<ffi::S32> col,
                  ffi::Buffer<ffi::S32> ptr, ffi::Buffer<ffi::F64> v,
                  ffi::Buffer<ffi::F64> old, ffi::Buffer<ffi::F64> base,
                  ffi::Buffer<ffi::F64> cost, ffi::Buffer<ffi::F64> step,
                  ffi::ResultBuffer<ffi::F64> out) {
  int64_t rows = base.element_count();
  if (rows > INT_MAX || ptr.element_count() != rows + 1 ||
      cost.element_count() != rows || out->element_count() != rows ||
      a.element_count() != col.element_count() ||
      old.element_count() != v.element_count() || step.element_count() != 1 ||
      (dual != 0 && dual != 1))
    return ffi::Error::InvalidArgument("Invalid LP CSR update buffer dimensions");
  if (!rows) return ffi::Error::Success();
  int blocks = (rows + Threads / Lanes - 1) / (Threads / Lanes);
  if (dual)
    Update<Lanes, Threads, true><<<blocks, Threads, 0, stream>>>(
        rows, a.typed_data(), col.typed_data(), ptr.typed_data(), v.typed_data(),
        old.typed_data(), base.typed_data(), cost.typed_data(), step.typed_data(),
        out->typed_data());
  else
    Update<Lanes, Threads, false><<<blocks, Threads, 0, stream>>>(
        rows, a.typed_data(), col.typed_data(), ptr.typed_data(), v.typed_data(),
        old.typed_data(), base.typed_data(), cost.typed_data(), step.typed_data(),
        out->typed_data());
  auto error = cudaPeekAtLastError();
  if (error != cudaSuccess) return ffi::Error::Internal(cudaGetErrorString(error));
  return ffi::Error::Success();
}

// HANDLERS_GENERATED_HERE
