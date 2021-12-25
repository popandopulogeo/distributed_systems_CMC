#include <cstdio>
#include <cstdlib>
#include <mpi.h>
#include <cstdbool>
#include <unistd.h>
#include <queue>
#include <string>
#include <sstream>

const int ROOT_PROCESS = 0;

enum ProcessState 
{
    PARENT_WAITING = 0,
    PARENT_COMPLETED = 1,
    LEFT_WAITING = 2,
    LEFT_COMPLETED = 3,
    RIGHT_WAITING = 4,
    RIGHT_COMPLETED = 5
};

enum Tag 
{
    REQUEST_TAG = 0,
    TOKEN_TAG = 1
};


int do_smth(int process_num) 
{
    const char *file_name = "file.txt";

    if (access(file_name, F_OK) != -1) 
    {
        fflush(stdout);
        return 1;
    } 
    else 
    {
        FILE *file = fopen(file_name, "w");
        int temp = rand() % 1000000;
        usleep(temp);
        fclose(file);
        remove(file_name);
        fflush(stdout);
        return 0;
    }
}

bool give_token_to(int process, int receiver, int *sys_state, int process_count) 
{
    int probe_output;
    MPI_Status status;
    fflush(stdout);
    MPI_Iprobe(receiver, REQUEST_TAG, MPI_COMM_WORLD, &probe_output, &status);

    if (probe_output) 
    {
        fflush(stdout);
        MPI_Send(sys_state, process_count, MPI_INT, receiver, TOKEN_TAG, MPI_COMM_WORLD);
        fflush(stdout);
        return true;
    }

    return false;
}

int critical_section(int (*do_smth)(int)) 
{

    int process, process_count;
    MPI_Status status;

    MPI_Comm_rank(MPI_COMM_WORLD, &process);
    MPI_Comm_size(MPI_COMM_WORLD, &process_count);

    // Init system_state
    int system_state[process_count];

    int dummy = 0;

    int parent = (process + 1) / 2 - 1;
    if (process != ROOT_PROCESS) 
    {
        MPI_Send(&dummy, 1, MPI_INT, parent, REQUEST_TAG, MPI_COMM_WORLD);
        fflush(stdout);

        // Receive token from parent
        MPI_Recv(&system_state, process_count, MPI_INT, parent, TOKEN_TAG, MPI_COMM_WORLD, &status);
        fflush(stdout);
    }

    //Critical section
    fflush(stdout);
    if (do_smth(process) != 0)
        return 1;
    fflush(stdout);

    system_state[process_count] |= 1;

    //Give token to left son
    int left_son = (process + 1) * 2 - 1;
    int right_son = (process + 1) * 2;
    bool left_son_ready = false;
    bool right_son_ready = false;
    
    while (!(left_son_ready && right_son_ready)) 
    {

        if (left_son >= process_count) 
        {
            left_son_ready = true;
        }

        if (!left_son_ready) 
        {
            
            left_son_ready = give_token_to(process, left_son, system_state, process_count);
            if (left_son_ready) 
            {
                system_state[process_count] = ProcessState::LEFT_COMPLETED;

                // Ask token back
                MPI_Send(&dummy, 1, MPI_INT, left_son, REQUEST_TAG, MPI_COMM_WORLD);
                fflush(stdout);
                MPI_Recv(&system_state, process_count, MPI_INT, left_son, TOKEN_TAG, MPI_COMM_WORLD, &status);
                fflush(stdout);
            }
        }

        if (right_son >= process_count) 
        {
            right_son_ready = true;
        }

        if (!right_son_ready) 
        {
            right_son_ready = give_token_to(process, right_son, system_state, process_count);
            if (right_son_ready) 
            {
                system_state[process_count] = ProcessState::RIGHT_COMPLETED;
                
                // Ask token back
                MPI_Send(&dummy, 1, MPI_INT, right_son, REQUEST_TAG, MPI_COMM_WORLD);
                fflush(stdout);
                MPI_Recv(&system_state, process_count, MPI_INT, right_son, TOKEN_TAG, MPI_COMM_WORLD, &status);
                fflush(stdout);
            }
        }
    }

    // Give token to parent
    if (process != ROOT_PROCESS) 
    {
        int parent_received = false;
        while (!parent_received) 
        {
            parent_received = give_token_to(process, parent, system_state, process_count);
        }
    }

    fflush(stdout);

    return 0;
}


int main(int argc, char *argv[]) 
{
    MPI_Init(&argc, &argv);
    int process;
    MPI_Comm_rank(MPI_COMM_WORLD, &process);

    double start_time = MPI_Wtime();

    if (critical_section(do_smth) != 0) 
    {
        MPI_Abort(MPI_COMM_WORLD, 1);
        exit(1);
    }
    double end_time = MPI_Wtime();
    MPI_Barrier(MPI_COMM_WORLD);

    printf("Process[%d] took %f s\n", process, end_time - start_time);
    MPI_Finalize();
    return 0;
}