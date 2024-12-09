#include <stdio.h>
#include<stdlib.h>
#include "mpi.h"


int main(int argc, char *argv[]){


     int my_rank, number_process, tag = 0; 
     int number_bins=0;
     int number_beads=0;
     int share_beads;
     

     MPI_Status status;

    // Initialize MPI
    MPI_Init(&argc, &argv);

    // Get the rank of the process
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    // Get the total number of processes
    MPI_Comm_size(MPI_COMM_WORLD, &number_process);

   if (my_rank == 0) {
        printf("Enter number of bins:\n");
        scanf("%d", &number_bins);

        printf("Enter number of beads:\n");
        scanf("%d", &number_beads);

        // Broadcast the parameters to all processes
        MPI_Bcast(&number_bins, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&number_beads, 1, MPI_INT, 0, MPI_COMM_WORLD);
    } else {
        // Non-root processes receive the broadcasted values
        MPI_Bcast(&number_bins, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&number_beads, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    share_beads= number_beads / number_process;
    int *histogram_global= calloc(number_bins,sizeof(int));



    

    int *histogram_each= calloc(number_bins,sizeof(int));

    for(int bead_id=0;bead_id< share_beads;++bead_id){

        double pos=number_bins/2.-0.5;  // Start in the middle bin

        for(int height=0;height<number_bins;height++){

            pos += rand() %2 -0.5;  // Randomly move left or right
        }
        histogram_each[(int)pos]++;
    }

    MPI_Reduce(histogram_each, histogram_global, number_bins, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);


    // Root process prints the result
    if (my_rank == 0) {
        printf("Final histogram:\n");
        for (int i = 0; i < number_bins; ++i) {
            printf("Bin %d: %d beads\n", i, histogram_global[i]);
        }
        free(histogram_global); // Free global histogram
    }

    // Free local histogram
    free(histogram_each);


MPI_Finalize();
return 0;



}