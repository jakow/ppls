/**
 * Adaptive quadrature with Bag of Tasks on MPI.
 *
 * Principles of operation:
 * The BoT farmer creates a task bag using a stack data structure, to make the
 * the order of execution roughly equivalent to those of recursive calls.
 * As described by the problem, the farmer maintains a bag with tasks that are
 * assigned to workers as they become available.
 *
 *
 * The workers:
 * When a worker is spawned, it sends a 0 result (line 162) to the farmer.
 * This is effectively a handshake that says "I'm available". This removes
 * the need to add code that handles the state of each worker at startup.
 * Instead, sending the zero result does nothing to the final result but the
 * farmer then knows that there is a worker with a given ID that is available.
 * Thus the same code can be reused later on, when the workers return with real
 * results.
 *
 * The worker uses a synchronous MPI_Recv, which blocks until a message
 * is received. This is because the worker cannot continue if it has not
 * received any new data.
 * The worker sends new tasks and results to the farmer using the standard,
 * blocking MPI_Send. If the message buffer is full, the worker must wait. This
 * is important because the worker sends and receives using the same buffer, and
 * the buffer should not be modified before a message is enqueued.
 * If an immediate mode send was used, and the message buffer is full,
 * then there is a chance that a new incoming task would override the data
 * buffer before the message is send.
 *
 * The farmer:
 * The farmer keeps track of the state of the workers in an array
 * (idle_list[n_workers]). At the start of execution, all workers are considered
 * busy. Then the farmer enters a loop, whose condition is that there are some
 * busy workers and/or some tasks left in the bag. The loop thus only terminates
 * when all tasks have finished.
 *
 * On entering the loop may only proceed if a message is received.
 * Thus a standard mode synchronous receive MPI_Recv is used, which
 * blocks until a message is received. The loop starts by receiving a message
 * using wildcards from ANY_SOURCE and with ANY_TAG.
 * Receiving a TASK_TAGged message means that there is one more task in
 * the queue. They are both placed in the bag of tasks. If a message is
 * RESULT_TAGged, it is accumulated by the farmer.
 *
 * Subsequently, the farmer assigns new tasks as long as there are tasks present
 * and there are idle workers which can accept them.
 *
 * Compile with:
 * mpicc = /usr/lib64/openmpi/bin/mpicc
 * mpirun = /usr/lib64/openmpi/bin/mpirun
 * mpicc -o stack.o -c stack.c
 * mpicc -o aquadSolution.o -c aquadSolution.c
 * mpicc aquadSolution.o stack.o -o aquadSolution -lm
 * Run with:
 * mpirun -c NPROC aquadSolution
 * Where NPROC is the number of parallel processes to run
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <unistd.h>
#include "stack.h"
#define EPSILON 1e-3
#define F(arg)  cosh(arg)*cosh(arg)*cosh(arg)*cosh(arg)
#define A 0.0
#define B 5.0


#define TASK_TAG 42
#define RESULT_TAG 404
#define DONE_TAG 1337

#define FALSE 0
#define TRUE 1
#define SLEEPTIME 1
#define FARMER 0

int *tasks_per_process;

double farmer(int);

void worker(int);

/**
 * Send an adaptive quadrature task to a given process. Uses blocking MPI_Send
 * @param process ID of the worker to receive the task
 * @param data array of 2 doubles [left, right] for adaptive quadrature
 */
void send_task(int, double*);

/**
 * Receive an adaptive quadrature task from any process.
 * @param recv_data Pointer to an array where the task data is stored
 */
void recv_task(double* recv_data);

/**
 * Send the result of adaptive quadrature computation to the farmer.
 * @param data Buffer with one double value
 */
void send_result(double* data) ;

int main(int argc, char **argv ) {
    int i, myid, numprocs;
    double area, a, b;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD,&myid);

    if(numprocs < 2) {
        fprintf(stderr, "ERROR: Must have at least 2 processes to run\n");
        MPI_Finalize();
        exit(1);
    }

    if (myid == 0) { // Farmer
        // init counters
        tasks_per_process = (int *) malloc(sizeof(int)*(numprocs));
        for (i=0; i<numprocs; i++) {
            tasks_per_process[i]=0;
        }
    }

    if (myid == 0) { // Farmer
        area = farmer(numprocs);
    } else { //Workers
        worker(myid);
    }

    if(myid == 0) {
        fprintf(stdout, "Area=%lf\n", area);
        fprintf(stdout, "\nTasks Per Process\n");
        for (i=0; i<numprocs; i++) {
            fprintf(stdout, "%d\t", i);
        }
        fprintf(stdout, "\n");
        for (i=0; i<numprocs; i++) {
            fprintf(stdout, "%d\t", tasks_per_process[i]);
        }
        fprintf(stdout, "\n");
        free(tasks_per_process);
    }
    MPI_Finalize();
    return 0;
}

double farmer(int numprocs) {
    /* Setup */
    MPI_Status status;
    double result = 0;
    double* send_data;
    double recv_data[2];
    int worker;
    int n_workers = numprocs - 1;
    // Idle list of booleans. Char to occupy less storage than int.
    // If C99 used, that would be simply a bool array
    char * idle_list = (char *) malloc(sizeof(char)*n_workers);
    for (worker = 0; worker < n_workers; worker++) {
        // set all workers to busy
        idle_list[worker] = FALSE;
    }
    int n_idle = 0;
    stack* bag = new_stack();
    // push the initial data to the bag
    recv_data[0] = A;
    recv_data[1] = B;
    push(recv_data, bag);
    // the loop will continue to run as long as there are some tasks in the bag
    while(!is_empty(bag) || n_idle < n_workers) {
        // receive any message, it can be a task or a result
        MPI_Recv(recv_data, 2, MPI_DOUBLE, MPI_ANY_SOURCE,
                 MPI_ANY_TAG, MPI_COMM_WORLD, &status);

        if (status.MPI_TAG == TASK_TAG) {
            push(recv_data, bag);
            // and get another task as well
            MPI_Recv(recv_data, 2, MPI_DOUBLE, status.MPI_SOURCE,
                     TASK_TAG, MPI_COMM_WORLD, &status);
            push(recv_data, bag);
        } else if (status.MPI_TAG == RESULT_TAG) {
            result += recv_data[0];
        }
        // the sending worker is now known to be idle
        idle_list[status.MPI_SOURCE-1] = 1;
        n_idle++;
        // find all idle workers and give them a task
        for (worker = 0; worker < n_workers && !is_empty(bag) && n_idle > 0; ++worker) {
            if (idle_list[worker]) {
                send_data = pop(bag);
                // worker i has process id i+1
                send_task(worker+1, send_data);
                free(send_data);
                idle_list[worker] = FALSE;
                n_idle--;
                tasks_per_process[worker+1]++;
            }
        }
    }
    for (worker = 1; worker < numprocs; ++worker) {
        // send an empty message to all workers with a DONE_TAG to indicate that
        // computation has finished
        MPI_Send(recv_data, 0, MPI_DOUBLE, worker, DONE_TAG, MPI_COMM_WORLD);
    }
    free(idle_list);
    return result;
}

void worker(int mypid) {
    // local data
    MPI_Status status;
    double data[2] = {0, 0};
    double left, right, mid, fmid, fleft, fright, larea, rarea, lrarea;
    double result = 0;
    // start by saying hi to the farmer by sending an empty result
    send_result(&result);
    while(1) {
        // try and get a message from the farmer
        MPI_Probe(FARMER, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        if (status.MPI_TAG == DONE_TAG) {
            break;
        }
        // otherwise a task tag
        recv_task(data);
        // compute the area approximations using adaptive quadrature
        left = data[0];
        right = data[1];
        usleep(SLEEPTIME); // sleep for some time to simulate heavy processing
        mid = (left + right) / 2;
        fleft = F(left);
        fright = F(right);
        fmid = F(mid);
        larea = (fleft + fmid) * (mid - left) / 2;
        rarea = (fmid + fright) * (right - mid) / 2;
        lrarea = (fleft + fright) * (right - left) / 2;
        // is the approximation good enough?
        if (fabs(lrarea - (larea + rarea)) < EPSILON) {
            // send the result
            result = (larea + rarea);
            send_result(&result);
        } else {
            // add two new task to the bag
            data[0] = left;
            data[1] = mid;
//            send_task(FARMER, data);
            send_task(FARMER, data);
            data[0] = mid;
            data[1] = right;
            send_task(FARMER, data);
        }
    }
    // acknowledge done and quit
    MPI_Recv(data, 0, MPI_DOUBLE, FARMER, DONE_TAG, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
}


void send_task(int process, double* data) {
    MPI_Send(data, 2, MPI_DOUBLE, process, TASK_TAG, MPI_COMM_WORLD);
}


void recv_task(double* recv_data) {
    MPI_Recv(recv_data, 2, MPI_DOUBLE, MPI_ANY_SOURCE, TASK_TAG, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);

}


void send_result(double* data) {
    MPI_Send(data, 1, MPI_DOUBLE, FARMER, RESULT_TAG, MPI_COMM_WORLD);
}