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
#define READY_TAG 80085

#define SLEEPTIME 1

#define FARMER 0

int *tasks_per_process;

double farmer(int);

void worker(int);

void send_task(int, double*);

void ack_ready(int);

void recv_task(double* recv_data);

void send_result(double* data);

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
    double area = 0;
    double* send_data;
    double recv_data[2];
    int worker, ready, tag;
    unsigned int tasks_created, tasks_dealt, result_count;
    stack* stack = new_stack();
    // push the initial data to the stack
    recv_data[0] = A;
    recv_data[1] = B;
    push(recv_data, stack);
    tasks_created = 1;
    result_count = 0;
    while(tasks_created > result_count * 2) {
        // immediate probe for new tasks and enqueue them. Ready flag will be
        // true if there is some tag to accumulate
        MPI_Iprobe(MPI_ANY_SOURCE, TASK_TAG, MPI_COMM_WORLD, &ready, &status);
        if (ready) {
            recv_task(recv_data);
            push(recv_data, stack);
            recv_task(recv_data);
            push(recv_data, stack);
            tasks_created += 2;
        }
        // immediate probe for result and accumulate it
        MPI_Iprobe(MPI_ANY_SOURCE, RESULT_TAG, MPI_COMM_WORLD, &ready, &status);
        if (ready) {
            worker = status.MPI_SOURCE;
            MPI_Recv(recv_data, 1, MPI_DOUBLE, worker, RESULT_TAG,
                     MPI_COMM_WORLD, &status);
            area += recv_data[0];
            result_count++;

        }
        // now assign tasks to idle workers
        int count = 0;
        while(!is_empty(stack)) {
            MPI_Iprobe(MPI_ANY_SOURCE, READY_TAG, MPI_COMM_WORLD, &ready,
                       &status);
            if (ready) {
                worker = status.MPI_SOURCE;
                // acknowledge the message
                ack_ready(worker);
                // and send a new task to the acknowledged worker
                send_data = pop(stack);
                send_task(worker, send_data);
                tasks_per_process[worker]++;
                free(send_data);
                count++;
            } else {
                // otherwise keep the worker in the queue
                break;
            }
        }
    }

    for (int i = 1; i < numprocs; ++i) {
        MPI_Send(recv_data, 0, MPI_DOUBLE, i, DONE_TAG, MPI_COMM_WORLD);
    }
    return area;
}

void worker(int mypid) {
    // local data
    MPI_Status status;
    double data[2];
    double left, right, mid, fmid, fleft, fright, larea, rarea, lrarea;
    double result = 0;
    // start by saying hi to the farmer
    MPI_Send(&mypid, 1, MPI_INT, FARMER, READY_TAG, MPI_COMM_WORLD);
    while(1) {
        // try and get a message from the farmer
        MPI_Probe(FARMER, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        if (status.MPI_TAG == DONE_TAG) {
            break;
        }
        // otherwise a task tag
        MPI_Recv(data, 2, MPI_DOUBLE, FARMER, TASK_TAG,
                 MPI_COMM_WORLD, &status);
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
            send_task(FARMER, data);
            data[0] = mid;
            data[1] = right;
            send_task(FARMER, data);
        }
        MPI_Send(&mypid, 1, MPI_INT, FARMER, READY_TAG, MPI_COMM_WORLD);
    }
    // acknowledge done
    MPI_Recv(data, 0, MPI_DOUBLE, FARMER, DONE_TAG, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
}

/**
 * Send a task to a given worker
 * @param worker ID of the worker to receive the task
 * @param data array of 2 doubles [left, right] for adaptive quadrature
 */
void send_task(int worker, double* data) {
    MPI_Send(data, 2, MPI_DOUBLE, worker, TASK_TAG, MPI_COMM_WORLD);
}

void send_result(double* data) {
    MPI_Send(data, 1, MPI_DOUBLE, FARMER, RESULT_TAG, MPI_COMM_WORLD);
}
/**
 *
 * @param worker
 */

void ack_ready(int worker) {
    int dummy;
    MPI_Recv(&dummy, 1, MPI_INT, worker, READY_TAG, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
}

void recv_task(double* recv_data) {
    MPI_Recv(recv_data, 2, MPI_DOUBLE, MPI_ANY_SOURCE, TASK_TAG, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);

}