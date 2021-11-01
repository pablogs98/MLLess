import time
import os
from multiprocessing import Process
from math import sqrt

import torch
import torch.distributed as dist

from benchmarks.pytorch.sampler import Sampler


class PytorchExecution:
    def __init__(self, model, optimizer, loss_func, threshold, epochs, execution_name, learning_rate, num_minibatches,
                 dataset_name, dataset_remote_bucket, seed, write_results=True,
                 adaptive_lr=False, max_time=None, n_threads=1, n_iters=None):
        self.model = model
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.threshold = threshold
        self.epochs = epochs
        self.execution_name = execution_name
        self.learning_rate = learning_rate
        self.num_minibatches = num_minibatches
        self.dataset_name = dataset_name
        self.rank = 0
        self.dataset_remote_bucket = dataset_remote_bucket
        self.seed = seed
        self.write_results = write_results
        self.adaptive_lr = adaptive_lr
        self.max_time = max_time
        self.n_threads = n_threads
        self.n_iters = n_iters

        self.sample = False
        self.step_time = []

    def run(self):
        ti = time.time()
        step = 0
        under_threshold = 0
        epoch = 0
        n_workers = dist.get_world_size()
        torch.set_num_threads(self.n_threads)

        dataset = Sampler(self.dataset_remote_bucket, self.dataset_name, self.rank, self.num_minibatches, self.seed,
                          n_workers)

        if self.rank == 0:
            if self.write_results:
                f_iter = open('pytorch_iter_{}.csv'.format(self.execution_name), 'w')
                f_iter.write('step;time;loss\n')
            print("-----------------------------------\n"
                  "Running {}\n"
                  "-----------------------------------\n"
                  "\tNumber of workers: {}\n"
                  "\tDataset: {}\n"
                  "\tLearning rate: {}\n"
                  "\tConvergence threshold: {}\n"
                  "\tMax epochs: {}\n"
                  "-----------------------------------".format(self.execution_name, n_workers, self.dataset_name,
                                                               self.learning_rate, self.threshold, self.epochs))

        t0 = time.time()

        iter_per_epoch = int(self.num_minibatches / n_workers)

        for minibatch in dataset:
            t_step = time.time()
            # Step
            samples, labels = self.get_samples_labels(minibatch)
            self.optimizer.zero_grad()
            prediction = self.model(samples)
            loss = self.loss_func(prediction, labels)
            loss.backward()
            self.average_gradients()
            self.optimizer.step()
            self.average_loss(loss)

            self.step_time.append(time.time() - t_step)

            # Check finalisation (10 iterations under convergence threshold)
            if loss.item() <= self.threshold:
                under_threshold += 1
            else:
                under_threshold = 0

            if under_threshold == 10 and self.rank == 0:
                break
            t_iter_1 = time.time()

            if self.rank == 0:
                if self.write_results:
                    f_iter.write('{};{};{}\n'.format(step, t_iter_1 - t0, loss.item()))

                if step % 15 == 0:
                    print("[Worker {}]: step {} is over! Loss={}".format(self.rank, step, loss.item()))

            # Check finalisation (max number of epochs)
            if step % iter_per_epoch == 0 and step != 0:
                if self.rank == 0:
                    print("Epoch {} is over!".format(epoch))
                # Scale learning rate
                if self.adaptive_lr:
                    for g in self.optimizer.param_groups:
                        g['lr'] = self.learning_rate / sqrt(epoch + 2)
                if epoch == (self.epochs - 1):
                    break
                epoch += 1

            if self.max_time is not None and (time.time() - ti) >= self.max_time:
                break

            if self.n_iters is not None and step >= self.n_iters:
                break

            step += 1

        t1 = time.time()
        if self.rank == 0:
            print(f"Finished training. Loss = {loss.item()}. Total time: {t1 - ti}. Training time: {t1 - t0}")
            print(f"Execution cost ($) = {6 * (0.2 / 3600) * (t1 - ti)}")

            if self.write_results:
                f_iter.close()

    def average_gradients(self):
        for param in self.model.parameters():
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data /= dist.get_world_size()

    def average_loss(self, loss):
        dist.reduce(loss, 0)
        if self.rank == 0:
            loss /= float(dist.get_world_size())

    def get_samples_labels(self, minibatch):
        raise NotImplementedError


def run_distributed(n_workers, execution, init_rank, end_rank, address='127.0.0.1', port='29500', backend='gloo'):
    processes = []
    for rank in range(init_rank, end_rank):
        p = Process(target=run_execution,
                    args=(n_workers, rank, address, port, backend, execution),
                    daemon=True)
        processes.append(p)
        p.start()
    processes[0].join()


def run_execution(n_workers, rank, address, port, backend, execution):
    print(f"Running worker {rank}! Trying to connect to master @ {address}:{port}...")
    os.environ['MASTER_ADDR'] = address
    os.environ['MASTER_PORT'] = port
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    dist.init_process_group(backend, rank=rank, world_size=n_workers)
    execution.rank = rank
    execution.run()
