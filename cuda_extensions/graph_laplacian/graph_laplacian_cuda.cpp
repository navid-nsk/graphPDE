/*
graph_laplacian_cuda.cpp

C++ wrapper for Graph Laplacian CUDA kernels.
Provides PyTorch interface with autograd support.
*/

#include <torch/extension.h>
#include <vector>

// Forward declarations of CUDA functions
torch::Tensor graph_laplacian_diffusion_cuda_forward(
    torch::Tensor population,
    torch::Tensor diffusion_coef,
    torch::Tensor node_to_city,
    torch::Tensor row_ptr,
    torch::Tensor col_idx,
    torch::Tensor laplacian_values
);

std::vector<torch::Tensor> graph_laplacian_diffusion_cuda_backward(
    torch::Tensor grad_output,
    torch::Tensor population,
    torch::Tensor diffusion_coef,
    torch::Tensor node_to_city,
    torch::Tensor row_ptr,
    torch::Tensor col_idx,
    torch::Tensor laplacian_values
);

// C++ interface
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor graph_laplacian_diffusion_forward(
    torch::Tensor population,
    torch::Tensor diffusion_coef,
    torch::Tensor node_to_city,
    torch::Tensor row_ptr,
    torch::Tensor col_idx,
    torch::Tensor laplacian_values
) {
    CHECK_INPUT(population);
    CHECK_INPUT(diffusion_coef);
    CHECK_INPUT(node_to_city);
    CHECK_INPUT(row_ptr);
    CHECK_INPUT(col_idx);
    CHECK_INPUT(laplacian_values);
    
    return graph_laplacian_diffusion_cuda_forward(
        population, diffusion_coef, node_to_city,
        row_ptr, col_idx, laplacian_values
    );
}

std::vector<torch::Tensor> graph_laplacian_diffusion_backward(
    torch::Tensor grad_output,
    torch::Tensor population,
    torch::Tensor diffusion_coef,
    torch::Tensor node_to_city,
    torch::Tensor row_ptr,
    torch::Tensor col_idx,
    torch::Tensor laplacian_values
) {
    CHECK_INPUT(grad_output);
    CHECK_INPUT(population);
    CHECK_INPUT(diffusion_coef);
    CHECK_INPUT(node_to_city);
    CHECK_INPUT(row_ptr);
    CHECK_INPUT(col_idx);
    CHECK_INPUT(laplacian_values);
    
    return graph_laplacian_diffusion_cuda_backward(
        grad_output, population, diffusion_coef, node_to_city,
        row_ptr, col_idx, laplacian_values
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &graph_laplacian_diffusion_forward, "Graph Laplacian diffusion forward (CUDA)");
    m.def("backward", &graph_laplacian_diffusion_backward, "Graph Laplacian diffusion backward (CUDA)");
}
