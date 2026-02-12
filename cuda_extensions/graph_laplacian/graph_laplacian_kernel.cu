/*
graph_laplacian_kernel.cu

CUDA kernel for efficient Graph Laplacian computation for reaction-diffusion.

Computes: D_i * sum_{u in N(v)} L_{vu} * rho_{i,u}

This is optimized for sparse graphs with CSR (Compressed Sparse Row) format.
*/

#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#define BLOCK_SIZE 256
#define WARP_SIZE 32

// Kernel for computing graph Laplacian diffusion term
// Uses CSR sparse matrix format for efficiency
__global__ void graph_laplacian_diffusion_kernel(
    const float* __restrict__ population,      // (n_nodes, n_ethnicities)
    const float* __restrict__ diffusion_coef,  // (n_cities, n_ethnicities)
    const int* __restrict__ node_to_city,      // (n_nodes,)
    const int* __restrict__ row_ptr,           // (n_nodes + 1,) CSR row pointers
    const int* __restrict__ col_idx,           // (n_edges,) CSR column indices
    const float* __restrict__ laplacian_values, // (n_edges,) Laplacian values
    float* __restrict__ output,                // (n_nodes, n_ethnicities)
    const int n_nodes,
    const int n_ethnicities
) {
    // Thread index
    int node_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (node_idx >= n_nodes) return;
    
    // Get city for this node
    int city_idx = node_to_city[node_idx];
    
    // Get row boundaries in CSR format
    int row_start = row_ptr[node_idx];
    int row_end = row_ptr[node_idx + 1];
    
    // Process each ethnicity
    for (int eth = 0; eth < n_ethnicities; eth++) {
        // Get diffusion coefficient for this city and ethnicity
        float D_i = diffusion_coef[city_idx * n_ethnicities + eth];
        
        // Safety check: clamp D_i to reasonable range
        D_i = fminf(fmaxf(D_i, 0.0f), 10.0f);
        
        // Compute Laplacian term: sum_{u in N(v)} L_{vu} * rho_{i,u}
        float laplacian_sum = 0.0f;
        
        for (int edge_idx = row_start; edge_idx < row_end; edge_idx++) {
            int neighbor_idx = col_idx[edge_idx];
            float L_vu = laplacian_values[edge_idx];
            float rho_neighbor = population[neighbor_idx * n_ethnicities + eth];
            
            // Safety checks
            if (isnan(L_vu) || isinf(L_vu)) continue;
            if (isnan(rho_neighbor) || isinf(rho_neighbor)) continue;
            
            // Clamp population to reasonable range
            rho_neighbor = fminf(fmaxf(rho_neighbor, 0.0f), 1e6f);
            
            laplacian_sum += L_vu * rho_neighbor;
        }
        
        // Clamp laplacian_sum to prevent explosion
        laplacian_sum = fminf(fmaxf(laplacian_sum, -1e6f), 1e6f);
        
        // Apply diffusion coefficient
        float result = D_i * laplacian_sum;
        
        // Final safety clamp
        result = fminf(fmaxf(result, -1e6f), 1e6f);
        
        output[node_idx * n_ethnicities + eth] = result;
    }
}

// Backward pass for graph Laplacian diffusion
__global__ void graph_laplacian_diffusion_backward_kernel(
    const float* __restrict__ grad_output,     // (n_nodes, n_ethnicities)
    const float* __restrict__ population,      // (n_nodes, n_ethnicities)
    const float* __restrict__ diffusion_coef,  // (n_cities, n_ethnicities)
    const int* __restrict__ node_to_city,      // (n_nodes,)
    const int* __restrict__ row_ptr,           // (n_nodes + 1,)
    const int* __restrict__ col_idx,           // (n_edges,)
    const float* __restrict__ laplacian_values, // (n_edges,)
    float* __restrict__ grad_population,       // (n_nodes, n_ethnicities)
    float* __restrict__ grad_diffusion_coef,   // (n_cities, n_ethnicities)
    const int n_nodes,
    const int n_ethnicities,
    const int n_cities
) {
    int node_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (node_idx >= n_nodes) return;
    
    int city_idx = node_to_city[node_idx];
    int row_start = row_ptr[node_idx];
    int row_end = row_ptr[node_idx + 1];
    
    for (int eth = 0; eth < n_ethnicities; eth++) {
        float D_i = diffusion_coef[city_idx * n_ethnicities + eth];
        float grad_out = grad_output[node_idx * n_ethnicities + eth];
        
        // Safety check: skip if grad_out is NaN or Inf
        if (isnan(grad_out) || isinf(grad_out)) {
            continue;
        }
        
        // Safety check: skip if D_i is NaN or Inf
        if (isnan(D_i) || isinf(D_i)) {
            continue;
        }
        
        // Gradient w.r.t. diffusion coefficient
        float laplacian_sum = 0.0f;
        for (int edge_idx = row_start; edge_idx < row_end; edge_idx++) {
            int neighbor_idx = col_idx[edge_idx];
            float L_vu = laplacian_values[edge_idx];
            float rho_neighbor = population[neighbor_idx * n_ethnicities + eth];
            
            // Safety checks
            if (isnan(L_vu) || isinf(L_vu) || isnan(rho_neighbor) || isinf(rho_neighbor)) {
                continue;
            }
            
            laplacian_sum += L_vu * rho_neighbor;
        }
        
        // Safety check on laplacian_sum
        if (isnan(laplacian_sum) || isinf(laplacian_sum)) {
            continue;
        }
        
        // Compute gradient contribution (with clamping to prevent explosion)
        float grad_diffusion = grad_out * laplacian_sum;
        
        // Clamp gradient to reasonable range
        grad_diffusion = fminf(fmaxf(grad_diffusion, -1e6f), 1e6f);
        
        // Accumulate gradient for diffusion coefficient (SAFE)
        if (!isnan(grad_diffusion) && !isinf(grad_diffusion)) {
            atomicAdd(&grad_diffusion_coef[city_idx * n_ethnicities + eth], grad_diffusion);
        }
        
        // Gradient w.r.t. population (transpose of Laplacian operation)
        for (int edge_idx = row_start; edge_idx < row_end; edge_idx++) {
            int neighbor_idx = col_idx[edge_idx];
            float L_vu = laplacian_values[edge_idx];
            
            // Safety checks
            if (isnan(L_vu) || isinf(L_vu)) {
                continue;
            }
            
            float grad_pop = grad_out * D_i * L_vu;
            
            // Clamp gradient
            grad_pop = fminf(fmaxf(grad_pop, -1e6f), 1e6f);
            
            // Accumulate gradient (SAFE)
            if (!isnan(grad_pop) && !isinf(grad_pop)) {
                atomicAdd(&grad_population[neighbor_idx * n_ethnicities + eth], grad_pop);
            }
        }
    }
}

// Host function to launch forward kernel
torch::Tensor graph_laplacian_diffusion_cuda_forward(
    torch::Tensor population,
    torch::Tensor diffusion_coef,
    torch::Tensor node_to_city,
    torch::Tensor row_ptr,
    torch::Tensor col_idx,
    torch::Tensor laplacian_values
) {
    const int n_nodes = population.size(0);
    const int n_ethnicities = population.size(1);
    
    auto output = torch::zeros_like(population);
    
    const int threads = BLOCK_SIZE;
    const int blocks = (n_nodes + threads - 1) / threads;
    
    graph_laplacian_diffusion_kernel<<<blocks, threads>>>(
        population.data_ptr<float>(),
        diffusion_coef.data_ptr<float>(),
        node_to_city.data_ptr<int>(),
        row_ptr.data_ptr<int>(),
        col_idx.data_ptr<int>(),
        laplacian_values.data_ptr<float>(),
        output.data_ptr<float>(),
        n_nodes,
        n_ethnicities
    );
    
    return output;
}

// Host function to launch backward kernel
std::vector<torch::Tensor> graph_laplacian_diffusion_cuda_backward(
    torch::Tensor grad_output,
    torch::Tensor population,
    torch::Tensor diffusion_coef,
    torch::Tensor node_to_city,
    torch::Tensor row_ptr,
    torch::Tensor col_idx,
    torch::Tensor laplacian_values
) {
    const int n_nodes = population.size(0);
    const int n_ethnicities = population.size(1);
    const int n_cities = diffusion_coef.size(0);
    
    auto grad_population = torch::zeros_like(population);
    auto grad_diffusion_coef = torch::zeros_like(diffusion_coef);
    
    const int threads = BLOCK_SIZE;
    const int blocks = (n_nodes + threads - 1) / threads;
    
    graph_laplacian_diffusion_backward_kernel<<<blocks, threads>>>(
        grad_output.data_ptr<float>(),
        population.data_ptr<float>(),
        diffusion_coef.data_ptr<float>(),
        node_to_city.data_ptr<int>(),
        row_ptr.data_ptr<int>(),
        col_idx.data_ptr<int>(),
        laplacian_values.data_ptr<float>(),
        grad_population.data_ptr<float>(),
        grad_diffusion_coef.data_ptr<float>(),
        n_nodes,
        n_ethnicities,
        n_cities
    );
    
    return {grad_population, grad_diffusion_coef};
}
