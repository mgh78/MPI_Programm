#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>

void swap(double** a, double** b) {
    double *temp = *b;
    *b = *a;
    *a = temp;
}

int main(int argc, char** args) {
    int rank;
    int num_procs;
    int block_size;
    
    MPI_Init(&argc, &args);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Status status;
    
    double delta_t = 0.02;
    // Reduced grid size for testing
    int grid_size = 1024;  // Much smaller for testing
    int num_time_steps = 100;  // Reduced time steps for testing
    double conductivity = 0.1;
    
    // Ensure grid_size is divisible by num_procs
    if (grid_size % num_procs != 0) {
        if (rank == 0) {
            printf("Error: grid_size (%d) must be divisible by number of processes (%d)\n", 
                   grid_size, num_procs);
        }
        MPI_Finalize();
        return 1;
    }
    
    // Calculate block size and allocate memory
    block_size = grid_size / num_procs;
    double* T_k = (double*)malloc(sizeof(double) * (block_size + 2));
    double* T_kn = (double*)malloc(sizeof(double) * (block_size + 2));
    
    if (T_k == NULL || T_kn == NULL) {
        printf("Process %d: Memory allocation failed!\n", rank);
        MPI_Finalize();
        return 1;
    }
    
    // Initialize local block (including ghost points)
    for (int i = 1; i <= block_size; i++) {
        T_k[i] = i + (block_size * rank);
    }
    
    // Debug print initial values
    printf("Process %d: Initial values [%f, %f, %f]\n", 
           rank, T_k[1], T_k[block_size/2], T_k[block_size]);
    
    for (int k = 0; k < num_time_steps; k++) {
        // Exchange ghost points
        if (rank > 0) {
            MPI_Sendrecv(&T_k[1], 1, MPI_DOUBLE, rank - 1, 0,
                        &T_k[0], 1, MPI_DOUBLE, rank - 1, 0,
                        MPI_COMM_WORLD, &status);
        }
        
        if (rank < num_procs - 1) {
            MPI_Sendrecv(&T_k[block_size], 1, MPI_DOUBLE, rank + 1, 0,
                        &T_k[block_size + 1], 1, MPI_DOUBLE, rank + 1, 0,
                        MPI_COMM_WORLD, &status);
        }
        
        // Handle boundary conditions
        if (rank == 0) {
            T_k[0] = T_k[1];
        }
        if (rank == num_procs - 1) {
            T_k[block_size + 1] = T_k[block_size];
        }
        
        // Compute new temperatures
        for (int i = 1; i <= block_size; i++) {
            double dTdt_i = conductivity * (-2 * T_k[i] + T_k[i-1] + T_k[i+1]);
            T_kn[i] = T_k[i] + delta_t * dTdt_i;
        }
        
        swap(&T_k, &T_kn);
        
        // Optional: Add debug output for specific timesteps
        if (k == 0 || k == num_time_steps-1) {
            printf("Process %d: Step %d, Values [%f, %f, %f]\n", 
                   rank, k, T_k[1], T_k[block_size/2], T_k[block_size]);
        }
    }
    
    // Calculate local average
    double local_sum = 0.0;
    for (int i = 1; i <= block_size; i++) {
        local_sum += T_k[i];
    }
    double local_avg = local_sum / block_size;
    
    // Reduce to get global average
    double global_avg;
    MPI_Reduce(&local_avg, &global_avg, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    global_avg /= num_procs;
    
    if (rank == 0) {
        printf("Global average temperature: %f\n", global_avg);
    }
    
    free(T_k);
    free(T_kn);
    MPI_Finalize();
    return 0;
}