#include <stdio.h>
#include <string.h>
#include "mpi.h"

int main(int argc, char *argv[]) {
    int my_rank, p, tag = 0; 
    char msg[20];
    MPI_Status status;

    // Initialize MPI
    MPI_Init(&argc, &argv);

    // Get the rank of the process
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    // Get the total number of processes
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    if (my_rank == 0) {
        // Process 0 sends a message
        strcpy(msg, "Hello");
        MPI_Send(msg, strlen(msg) + 1, MPI_CHAR, 1, tag, MPI_COMM_WORLD);
    }

    if (my_rank == 1) {
        // Process 1 receives the message
        MPI_Recv(msg, 20, MPI_CHAR, 0, tag, MPI_COMM_WORLD, &status);
        printf("Process 1 received message: %s\n", msg);
    }

    // Finalize MPI
    MPI_Finalize();
    return 0;
}
