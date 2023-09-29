"""jobshop class."""
import collections
from ortools.sat.python import cp_model
import wandb
from stable_baselines3.common.base_class import BaseAlgorithm
#class Task:
#    def __init__(self, machine, duration):
#        self.machine = machine
#        self.duration = duration

class JobShop:
    def __init__(self):
        self.jobs_data = [ [] ]

    def InsertJobs(self, job, machine_id, processing_time):
        while len(self.jobs_data) < job+1:
            self.jobs_data.append([])
        self.jobs_data[job].append([machine_id, processing_time])
        #print (self.jobs_data)

    def BuildModel(self):
        self.machines_count = 1 + max(task[0] for job in self.jobs_data for task in job)
        self.all_machines = range(self.machines_count)
        # Computes horizon dynamically as the sum of all durations.
        self.horizon = sum(task[1] for job in self.jobs_data for task in job)

        # Create the model.
        self.model = cp_model.CpModel()

        # Named tuple to store information about created variables.
        self.task_type = collections.namedtuple('task_type', 'start end interval')
        # Named tuple to manipulate solution information.
        self.assigned_task_type = collections.namedtuple('assigned_task_type', 'start job index duration')

        # Creates job intervals and add to the corresponding machine lists.
        self.all_tasks = {}
        self.machine_to_intervals = collections.defaultdict(list)

        for job_id, job in enumerate(self.jobs_data):
            for task_id, task in enumerate(job):
                machine = task[0]
                duration = task[1]
                suffix = '_%i_%i' % (job_id, task_id)
                start_var = self.model.NewIntVar(0, self.horizon, 'start' + suffix)
                end_var = self.model.NewIntVar(0, self.horizon, 'end' + suffix)
                interval_var = self.model.NewIntervalVar(start_var, duration, end_var, 'interval' + suffix)
                self.all_tasks[job_id, task_id] = self.task_type(start=start_var, end=end_var, interval=interval_var)
                self.machine_to_intervals[machine].append(interval_var)

        # Create and add disjunctive constraints.
        for machine in self.all_machines:
            self.model.AddNoOverlap(self.machine_to_intervals[machine])

        # Precedences inside a job.
        for job_id, job in enumerate(self.jobs_data):
            for task_id in range(len(job) - 1):
                self.model.Add(self.all_tasks[job_id, task_id + 1].start >= self.all_tasks[job_id, task_id].end)

        # Makespan objective.
        self.obj_var = self.model.NewIntVar(0, self.horizon, 'makespan')
        self.endVars = []
        for job_id, job in enumerate(self.jobs_data):
            self.job_length = len(job)
            if self.job_length > 0:
                self.endVars.append(self.all_tasks[job_id,self.job_length - 1].end)

        self.model.AddMaxEquality(self.obj_var, self.endVars)

        self.model.Minimize(self.obj_var)
        # print('>>>',job_id, len(job) - 1)
        # self.model.AddMaxEquality(obj_var, [
        #     self.all_tasks[job_id, len(job) - 1].end
        #     for job_id, job in enumerate(self.jobs_data)
        # ])
        # self.model.Minimize(obj_var)

    def Solve(self):
        # Creates the solver and solve.
        self.solver = cp_model.CpSolver()
        self.status = self.solver.Solve(self.model)

    def Output(self):
        if self.status == cp_model.OPTIMAL or self.status == cp_model.FEASIBLE:
            print('Solution:')
            # Create one list of assigned tasks per machine.
            self.assigned_jobs = collections.defaultdict(list)
            for job_id, job in enumerate(self.jobs_data):
                for task_id, task in enumerate(job):
                    machine = task[0]
                    self.assigned_jobs[machine].append(
                        self.assigned_task_type(start=self.solver.Value(
                            self.all_tasks[job_id, task_id].start),
                                           job=job_id,
                                           index=task_id,
                                           duration=task[1]))

            # Create per machine output lines.
            self.output = ''
            i = 0
            for machine in self.all_machines:
                # Sort by starting time.
                self.assigned_jobs[machine].sort()
                self.sol_line_tasks = 'Machine ' + str(machine) + ': '
                self.sol_line = '           '

                for assigned_task in self.assigned_jobs[machine]:
                    name = 'job_%i_task_%i' % (assigned_task.job,
                                               assigned_task.index)
                    # Add spaces to output to align columns.
                    self.sol_line_tasks += '%-15s' % name

                    start = assigned_task.start
                    duration = assigned_task.duration
                    sol_tmp = '[%i,%i]' % (start, start + duration)
                    # Add spaces to output to align columns.
                    self.sol_line += '%-15s' % sol_tmp

                self.sol_line += '\n'
                self.sol_line_tasks += '\n'
                self.output += self.sol_line_tasks
                self.output += self.sol_line

            # Finally print the solution found.
            print(f'Optimal Schedule Length: {self.solver.ObjectiveValue()}')
            print(self.output)
            
            wandb.log({'Objective Value': self.solver.ObjectiveValue(), 'steps': i})
            
        else:
            print('No solution found.')

        i += 1
        # Statistics.
        print('\nStatistics')
        print('  - conflicts: %i' % self.solver.NumConflicts())
        print('  - branches : %i' % self.solver.NumBranches())
        print('  - wall time: %f s' % self.solver.WallTime())

# def main():
#     instance1 = JobShop()
#     instance1.InsertJobs(0, 0, 3)
#     instance1.InsertJobs(0, 1, 2)
#     instance1.InsertJobs(0, 2, 2)
#     instance1.InsertJobs(1, 0, 2)
#     instance1.InsertJobs(1, 2, 1)
#     instance1.InsertJobs(1, 1, 4)
#     instance1.InsertJobs(2, 1, 4)
#     instance1.InsertJobs(2, 2, 3)
#     instance1.BuildModel()
#     instance1.Solve()
#     instance1.Output()

#     instance2 = JobShop()
#     instance2.InsertJobs(0, 0, 5)
#     instance2.InsertJobs(0, 1, 1)
#     instance2.InsertJobs(0, 2, 7)
#     instance2.InsertJobs(1, 0, 4)
#     instance2.InsertJobs(1, 2, 2)
#     instance2.InsertJobs(1, 1, 8)
#     instance2.InsertJobs(2, 1, 3)
#     instance2.InsertJobs(2, 2, 3)
#     instance2.BuildModel()
#     instance2.Solve()
#     instance2.Output()

# if __name__ == '__main__':
#     main()