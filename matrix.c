#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <mpi.h>
#include <time.h>
#include <sys/time.h>
#include <signal.h>
#include <mpi-ext.h>

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

void term(int condition, const char *error_msg) {
    if (condition) {
        if (error_msg != NULL) {
            printf("%s\n", error_msg);
        }
        exit(1);
    }
}

void* alloc(size_t bytes) {
    void *p = calloc(1, bytes);
    term(p == NULL, "Cannot allocate memory.\n");
    return p;
}

void print_arri(int *arr, size_t n) {
    for (int i = 0; i < n; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");
}

void print_mat(double *arr, size_t n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%f ", arr[i * n + j]);
        }
        printf("\n");
    }
    printf("\n");
}

void print_mat_rect(double *arr, size_t n, size_t m) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            printf("%f ", arr[i * m + j]);
        }
        printf("\n");
    }
    printf("\n");
}

void random_init(double *arr, size_t n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            arr[i * n + j] = (double)random() / RAND_MAX;
        }
    }
}

void expand_mat(double *A, size_t n, double *B, size_t m) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            B[i * m + j] = A[i * n + j];
        }
    }
}

void transpose(double *A, size_t n) {
    int i, j;
    for (i = 1; i < n; i++) {
        for (j = 0; j < i; j++) {
            double tmp = A[i * n + j];
            A[i * n + j] = A[j * n + i];
            A[j * n + i] = tmp;
        }
    }
}

void read_matrix(double *A, size_t n, FILE *file_in) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            int ret = fscanf(file_in, "%lf", &A[i * n + j]);
        }
    }
}

void write_matrix(double *A, size_t n, FILE *file_out) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            int ret = fprintf(file_out, "%lf ", A[i * n + j]);
        }
        int ret = fprintf(file_out, "\n");
    }
}

MPI_Comm proc_comm = MPI_COMM_WORLD;

int task_size = 0; // initial number of processes
int procs_killed = 0;
int initial_rank = 0;
int *alive_indices = NULL; // tmp var
int *rank_task = NULL;
int *task_rank = NULL;
int uniq_rank = 0;

int spawned = 0; // flag for newly spawned process
const char *matrix_inputs = "_mat_in";
int verbose_output = 0;

int gargc;
char **gargv;

int original_n = 0;
int n_mat = 0;
int part_size = 0;
int part_elems = 0;
double *bufA = NULL;
double *bufB = NULL;
double *bufC = NULL;
int permute_step = 0;

double *shuffle_buf = NULL;
int entered_multiply = 0;

double *original_a = NULL;
double *original_b = NULL;
double *original_c = NULL;

int mpi_multiplicate(size_t n);
void on_error(int exec);

void allocate_all() {
    /* part of mpi_multiplicate */
    int proc_rank;
    MPI_Comm_rank(proc_comm, &proc_rank);
    
    part_size = n_mat / task_size;
    part_elems = part_size * n_mat;
    bufA = alloc(part_elems * sizeof(*bufA));
    bufB = alloc(part_elems * sizeof(*bufB));
    bufC = alloc(part_elems * sizeof(*bufC));
}

static void verbose_errhandler(MPI_Comm* comm, int* err, ...) {
    on_error(0);
}

void on_error(int exec) {
    if (!exec) { // disable automatic execution of handler, wait until operation completes
        return;
    }
    
    int rank, size;

    if (!spawned) {
        MPI_Comm old_comm = proc_comm;

        /* exclude communication with killed processes */
        MPIX_Comm_revoke(old_comm);
        MPIX_Comm_shrink(old_comm, &proc_comm);
        
        /* renumerate processes */
        MPI_Comm_rank(proc_comm, &rank);
        MPI_Comm_size(proc_comm, &size); // number of alive processes!
        
        
        /* immediate actions after failure */
        
        /* which ranks did the killed processes have before renumeration? */
        MPI_Allgather(&initial_rank, 1, MPI_INT, alive_indices, 1, MPI_INT, proc_comm);        
        procs_killed = task_size - size;
        
        if (procs_killed == 0) return;
    } else {
        procs_killed = 1; // spawned child. The while loop will execute at least once
    }
    
    /* respawn all killed procs ((((alive1, alivek), spawned1), spawned2), ...)  */
    MPI_Comm parentcomm;
    MPI_Comm childcomm;
    MPI_Comm intracomm;
    MPI_Comm_get_parent(&parentcomm);
    
    int procs_spawned = 0;
    while (procs_spawned < procs_killed) {
        if (!spawned) {
            // already running process
            MPI_Comm_spawn(gargv[0], MPI_ARGV_NULL, 1, MPI_INFO_NULL, 
                0, proc_comm, &childcomm, MPI_ERRCODES_IGNORE);
            if (rank == 0) {
                // send context to child
                MPI_Send(&procs_spawned, 1, MPI_INT, 0,
                    0, childcomm);
                MPI_Send(&procs_killed, 1, MPI_INT, 0,
                    0, childcomm);
            }
            
            MPI_Intercomm_merge(childcomm, 0, &intracomm); // 0 means before
            MPI_Comm_free(&childcomm);
            MPI_Comm_free(&proc_comm);
            proc_comm = intracomm;
            
            int rk;
            MPI_Comm_rank(proc_comm, &rk);
            rank = rk;
            
        } else {
            // freshly created process
            MPI_Comm_rank(proc_comm, &rank);
            // receive context
            MPI_Status status;
            MPI_Recv(&procs_spawned, 1, MPI_INT, 0,
                0, parentcomm, &status);
            MPI_Recv(&procs_killed, 1, MPI_INT, 0,
                0, parentcomm, &status);
            
            MPI_Intercomm_merge(parentcomm, 1, &intracomm); // 1 means after
            MPI_Comm_free(&parentcomm);
            proc_comm = intracomm;
            
            spawned = 0; // the process is ready to participate!
            MPI_Comm_rank(proc_comm, &rank);
        }
        if (rank == 0) {
            printf("1 proc spawned!\n\n");
        }
        if (rank == 0) {
            printf("\n", rank);
        }
        MPI_Barrier(proc_comm);
        procs_spawned++;
    }
    MPI_Barrier(proc_comm);
    
    /* re-set error handler */
    MPI_Errhandler errh;
    MPI_Comm_create_errhandler(verbose_errhandler, &errh);
    MPI_Comm_set_errhandler(proc_comm, errh);
    
    MPI_Bcast(&size, 1, MPI_INT, 0, proc_comm); // not necessary for alive procs!
    MPI_Bcast(&task_size, 1, MPI_INT, 0, proc_comm);
    int procs_alive = size;
    
    MPI_Bcast(&n_mat, 1, MPI_INT, 0, proc_comm);
    MPI_Bcast(&original_n, 1, MPI_INT, 0, proc_comm);
    MPI_Bcast(&permute_step, 1, MPI_INT, 0, proc_comm);
    
    MPI_Bcast(&verbose_output, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    if (rank == 0 && verbose_output) {
        printf("Task table: \n");
        for (int i = 0; i < size; i++) {
            printf("%d-%d\n", i, alive_indices[i]);
        }
    }
    
    
    if (size <= rank) { // allocate resources in spawned procs
        initial_rank = -1;
        rank_task = alloc(task_size * sizeof(*rank_task));
        task_rank = alloc(task_size * sizeof(*task_rank));
        allocate_all();
    }
    MPI_Allgather(&initial_rank, 1, MPI_INT, rank_task, 1, MPI_INT, proc_comm);

    int max_uniq;
    MPI_Allreduce(&uniq_rank, &max_uniq, 1, MPI_INT, MPI_MAX, proc_comm);
    max_uniq++;
    
    /* assign new tasks in each process separately */
    int spawn_idx = procs_alive;
    for (int task = 0; task < task_size; task++) {
        task_rank[task] = -1;
    }
    for (int r = 0; r < procs_alive; r++) {
        task_rank[rank_task[r]] = r;
    }
    for (int task = 0; task < task_size; task++) {
        if (task_rank[task] == -1) {
            rank_task[spawn_idx] = task; // assign task to spawned proc
            task_rank[task] = spawn_idx;
            if (spawn_idx == rank) {
                uniq_rank = max_uniq;
                if (verbose_output) {
                    printf("Chose uniq %d for rank %d\n", uniq_rank, spawn_idx);
                }
            }
            spawn_idx++;
            max_uniq++;
        }
    }
    
    /* initialize all global variables inside spawned procs and launch multiplicate! */
    if (procs_alive <= rank) {
        initial_rank = rank_task[rank];
        alive_indices = alloc(task_size * sizeof(*alive_indices)); // allocate for later use
        
        while (mpi_multiplicate(n_mat)) {}
        MPI_Finalize();
        exit(0);
    }
    
    /* non-spawned procs fall through here out of the function */
}



void shuffle_order(double *A, int use_permute) {
    /* move around pieces according to task table */
    // shuffle using shuffle_buf and rank_task and permute_step
    for (int rk = 0; rk < task_size; rk++) {
        /* which task does this rank execute? */
        int task = rank_task[rk];
        if (!use_permute) {
            memcpy((void*)&(shuffle_buf[part_elems*rk]),
            (void*)&(A[part_elems*task]), part_elems*sizeof(*shuffle_buf));
        } else {
            task = (task - permute_step + task_size) % task_size;
            memcpy((void*)&(shuffle_buf[part_elems*rk]),
            (void*)&(A[part_elems*task]), part_elems*sizeof(*shuffle_buf));
        }
    }
}

void unshuffle_order(double *A) {
    /* after calculations were made, unshuffle */
    // shuffle using shuffle_buf and rank_task and permute_step
    for (int rk = 0; rk < task_size; rk++) {
        /* which task does this rank execute? */
        int task = rank_task[rk];
        memcpy((void*)&(shuffle_buf[part_elems*task]),
        (void*)&(A[part_elems*rk]), part_elems*sizeof(*shuffle_buf));
        
    }
}

int mpi_multiplicate(size_t n) {
    int i, j, k;
    MPI_Status status;
    
    int proc_num;
    MPI_Comm_size(proc_comm, &proc_num);
    int proc_rank;
    MPI_Comm_rank(proc_comm, &proc_rank);
    
    // ! both regular and spawned procs enter here !
    
    if (!entered_multiply) {
        n_mat = n;
        allocate_all();
        entered_multiply = 1;
    }
    
    double *A, *B, *C;
    if (proc_rank == 0) {
        if (original_a == NULL) {
            original_a = alloc(n*n*sizeof(*original_a));
            original_b = alloc(n*n*sizeof(*original_b));
            original_c = alloc(n*n*sizeof(*original_c));
            shuffle_buf = alloc(n_mat * n_mat * sizeof(*shuffle_buf));
        }
        A = original_a;
        B = original_b;
        C = original_c;
        FILE *save_checkpoint = fopen(matrix_inputs, "r");
        for (int i = 0; i < n_mat; i++) {
            for (int j = 0; j < n_mat; j++) {
                fscanf(save_checkpoint, "%lf", &A[i * n + j]);
            }
        }
        for (int i = 0; i < n_mat; i++) {
            for (int j = 0; j < n_mat; j++) {
                fscanf(save_checkpoint, "%lf", &B[i * n + j]);
            }
        }
        fclose(save_checkpoint);
    }
    
    if (proc_rank == 0) {
        shuffle_order(A, 0);
    }
    MPI_Scatter(shuffle_buf, part_elems, MPI_DOUBLE, bufA, part_elems, MPI_DOUBLE, 0, proc_comm);
    
    if (proc_rank == 0) {
        transpose(B, n);
        shuffle_order(B, 1);
    }
    MPI_Scatter(shuffle_buf, part_elems, MPI_DOUBLE, bufB, part_elems, MPI_DOUBLE, 0, proc_comm); 

    
    
    char filename[PATH_MAX];
    sprintf(filename, "_%03d.save", initial_rank);
    
    /* read checkpoint */
    FILE *save_checkpoint = fopen(filename, "r");
    int lasti = 0, lastj = 0;
    char read_char;
    if (save_checkpoint != NULL && 1) {
        double val;
        char end;
        /* we are at permute step, no more than so many calculations were written down */
        for (int cnt = 0; cnt < permute_step * part_size * part_size; cnt++) {
            fscanf(save_checkpoint, "%lf %c\n", &val, &end);
        }
        int cnt = 0;
        while (cnt < part_size * part_size) {
            int ret = fscanf(save_checkpoint, "%lf %c\n", &val, &end);
            if (ret != 2 || end != 'e') {
                break;
            }
            cnt++;
        }
        lasti = cnt / part_size;
        lastj = cnt % part_size;
        fclose(save_checkpoint);
    }
    
    if (verbose_output) {
        if (proc_num == 0) {
            printf("Start from permutation %d\n", permute_step);
        }
        printf("Task %d continues from: i=%d j=%d\n", initial_rank, lasti, lastj);
    }
    
    
    save_checkpoint = fopen(filename, "a"); // reopen for writing
    
    /* toss random for random failures */
    int wd = 0;
    if (proc_rank == 0) {
        if ( rand() % 3 == 0) {
            wd = 1;
        }
    }
    /* main matrix multiplication */
    while (permute_step < task_size) {
        for (i = lasti; i < part_size; i++) {
            for (j = lastj; j < part_size; j++) {
                double tmp = 0;
                for (k = 0; k < n; k++) {
                    tmp += bufA[i * n + k] * bufB[j * n + k];
                }
                int strip_idx = (initial_rank - permute_step + task_size) % task_size;
                bufC[i * n + j + strip_idx * part_size] = tmp;
                fprintf(save_checkpoint, "%lf e\n", tmp); // c_ij #marker
                if (uniq_rank == 0 || uniq_rank == 1 || uniq_rank == 3 || uniq_rank == 7) {
                    printf("Kill id %d\n", uniq_rank);
                    raise(SIGKILL);
                }
                fflush(save_checkpoint);
            }
            lastj = 0;
        }
        lasti = 0;
        
        
        int next_proc = task_rank[(initial_rank + 1) % task_size];
        int prev_proc = task_rank[(initial_rank - 1 + task_size) % task_size];
        int rc = MPI_Sendrecv_replace(bufB, part_elems, MPI_DOUBLE, next_proc, 0,
                prev_proc, 0, proc_comm, &status);
        if (rc == MPI_SUCCESS) {
            rc = MPI_Barrier(proc_comm);
        }
        if (rc != MPI_SUCCESS) {
            printf("Proc with task %d detected error!\n", initial_rank);
            fclose(save_checkpoint);
            on_error(1);
            return 1;
        }
        
        permute_step++;
    }
    fclose(save_checkpoint);

    /* load results of computation to assemble */
    {
        save_checkpoint = fopen(filename, "r");
        if (save_checkpoint == NULL) {
            printf("CHECKPOINT FOR %d NOT FOUND", initial_rank);
        }
        double val;
        char end;
        for (int permute = 0; permute < task_size; permute++) {
            for (i = 0; i < part_size; i++) {
                for (j = 0; j < part_size; j++) {
                    fscanf(save_checkpoint, "%lf %c\n", &val, &end);
                    
                    int strip_idx = (initial_rank - permute + task_size) % task_size;
                    bufC[i * n_mat + j + strip_idx * part_size] = val;
                }
            }
        }
        fclose(save_checkpoint);
    }
    
    MPI_Gather(bufC, part_elems, MPI_DOUBLE, C, part_elems, MPI_DOUBLE, 0, proc_comm);
    if (proc_rank == 0) {
        unshuffle_order(C);
        memcpy((void*)C, (void*)shuffle_buf, n_mat*n_mat*sizeof(*C));
        
        if (verbose_output) {
            printf("Matrix C:\n");
            for (int i = 0; i < original_n; i++) {
                for (int j = 0; j < original_n; j++) {
                    printf("%f ", C[i * n_mat + j]);
                }
                printf("\n");
            }
            printf("\n");
        }
        
        transpose(B, n_mat);
        int n1 = original_n;
        for (i = 0; i < n1; i++) {
            for (j = 0; j < n1; j++) {
                double tmp = 0;
                for (k = 0; k < n1; k++) {
                    tmp += A[i * n_mat + k] * B[k * n_mat + j];
                }
                C[i * n_mat + j] = tmp;
            }
        }
        
        if (verbose_output) {
            printf("True matrix C:\n");
            for (int i = 0; i < original_n; i++) {
                for (int j = 0; j < original_n; j++) {
                    printf("%f ", C[i * n_mat + j]);
                }
                printf("\n");
            }
            printf("\n");
        }
    }
    
    free(bufA);
    free(bufB);
    free(bufC);
    
    MPI_Barrier(proc_comm);
    printf("Process finished: %d\n", initial_rank);
    MPI_Barrier(proc_comm);
    return 0;
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    gargc = argc;
    gargv = argv;
    
    int proc_num;
    MPI_Comm_size(proc_comm, &proc_num);
    int proc_rank;
    MPI_Comm_rank(proc_comm, &proc_rank);
 
    /* were we spawned? */
    MPI_Comm parentcomm;
    MPI_Comm_get_parent(&parentcomm);
    spawned = parentcomm != MPI_COMM_NULL;
    if (spawned) { // go sync with spawn operation
        MPI_Errhandler errh;
        MPI_Comm_create_errhandler(verbose_errhandler, &errh);
        MPI_Comm_set_errhandler(proc_comm, errh);
        
        on_error(1);
        MPI_Finalize();
        return 0;
    }
    
    /* variables for failure tolerance */
    initial_rank = proc_rank;
    task_size = proc_num;
    alive_indices = alloc(task_size * sizeof(*alive_indices)); // allocate for later use
    rank_task = alloc(task_size * sizeof(*rank_task));
    task_rank = alloc(task_size * sizeof(*task_rank));
    uniq_rank = proc_rank;
    
    for (int rk = 0; rk < task_size; rk++) {
        rank_task[rk] = rk;
        task_rank[ rank_task[rk] ] = rk;
    }
    initial_rank = rank_task[proc_rank];
    permute_step = 0;

    /* read command line args */
    if (proc_rank == 0) {
        int n = 3; // matrix width
        int verbose = 0;
        double *a, *b, *c; // matrices
        char *file_in = NULL; // optional input file
        
        int arg = 1;
        const char *usage = ("Usage: \"./main <matrix_width> [i mult_in.txt]\"\n"
                 "i mult_in.txt - read matrices A and B from mult_in.txt");
        
        /* start option parsing */
        term(argc == arg, usage);
        
        n = strtol(argv[arg++], NULL, 10);

        while (argc != arg) {
            if (strcmp(argv[arg], "i") == 0) {
                arg++;
                term(argc == arg, "Option \"i\" must be followed by file name.");
                file_in = argv[arg];
            } else if (strcmp(argv[arg], "v") == 0) {
                verbose = 1;
                verbose_output = 1;
            } else {
                term(1, "Option not recognized.");
            }
            arg++;
        }
        /* end option parsing */
        
        a = alloc(n * n * sizeof(*a));
        b = alloc(n * n * sizeof(*b));
        c = alloc(n * n * sizeof(*c));
    
        if (file_in == NULL) {
            random_init(a, n);
            random_init(b, n);
        } else {
            FILE *f = fopen(file_in, "r");
            term(f == NULL, "Input file not found.");
            for (int i = 0; i < n * n; i++) {
                int ret = fscanf(f, "%lf", &a[i]);
                term(ret < 1, "Not enough numbers in file.");
            }
            for (int i = 0; i < n * n; i++) {
                int ret = fscanf(f, "%lf", &b[i]);
                term(ret < 1, "Not enough numbers in file.");
            }
            fclose(f);
        }
        
        if (verbose) {
            printf("Matrix A:\n");
            print_mat(a, n);
            printf("Matrix B:\n");
            print_mat(b, n);
        }
    
        original_n = n;
        n_mat = (n + proc_num - 1) / proc_num * proc_num;
        MPI_Bcast(&n_mat, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&original_n, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&verbose_output, 1, MPI_INT, 0, MPI_COMM_WORLD);

        int i, j, k;
        size_t m = (n + proc_num - 1) / proc_num * proc_num; // make matrix size divisible by proc_num
        double *a_expand = alloc(m * m * sizeof(*a_expand));
        double *b_expand = alloc(m * m * sizeof(*b_expand));
        double *c_expand = alloc(m * m * sizeof(*c_expand));
        expand_mat(a, n, a_expand, m);
        expand_mat(b, n, b_expand, m);
        
        /* save input matrices for failure cases */
        FILE *input_save = fopen(matrix_inputs, "w");
        write_matrix(a_expand, m, input_save);
        fprintf(input_save, "\n");
        write_matrix(b_expand, m, input_save);
        fclose(input_save);
        
        MPI_Errhandler errh;
        MPI_Comm_create_errhandler(verbose_errhandler, &errh);
        MPI_Comm_set_errhandler(proc_comm, errh);

        while (mpi_multiplicate(n_mat)) {}
        
    } else {
        /* non-root process */
        MPI_Bcast(&n_mat, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&original_n, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&verbose_output, 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        MPI_Errhandler errh;
        MPI_Comm_create_errhandler(verbose_errhandler, &errh);
        MPI_Comm_set_errhandler(proc_comm, errh);
        
        while (mpi_multiplicate(n_mat)) {}
    }

    MPI_Finalize();
    return 0;
}