import torch
import pandas as pd

class Logger(object):

    def __init__(self, runs, metrics):
        self.valid_results = [[] for _ in range(runs)]
        self.test_results = [[] for _ in range(runs)]
        self.metrics_name = metrics

    def add_result(self, run, valid_metrics, test_metrics):
        assert len(valid_metrics) == len(test_metrics)
        assert len(valid_metrics) == len(self.metrics_name)
        self.valid_results[run].append(valid_metrics)
        self.test_results[run].append(test_metrics)

    def print_statistics(self, run=None, record=False, out_path=None):
        if run is not None:
            valid_results = torch.tensor(self.valid_results[run]) # [num epoch, num metric]
            test_results = torch.tensor(self.test_results[run])
            argmin = valid_results[:, 0].argmin().item() # epoch with lowest mse
            print(f'Run {run + 1:02d}:')
            print(f'Chosen epoch: {argmin + 1}')
            print(f'Best Valid {self.metrics_name[0]}: {valid_results[argmin, 0]:.3f}, '
                  f'{self.metrics_name[1]}: {valid_results[argmin, 1]:.3f}, '
                  f'{self.metrics_name[2]}: {valid_results[argmin, 2]:.3f}, '
                  f'{self.metrics_name[3]}: {valid_results[argmin, 3]:.3f}')
            print(f'Final Test {self.metrics_name[0]}: {test_results[argmin, 0]:.3f}, '
                  f'{self.metrics_name[1]}: {test_results[argmin, 1]:.3f}, '
                  f'{self.metrics_name[2]}: {test_results[argmin, 2]:.3f}, '
                  f'{self.metrics_name[3]}: {test_results[argmin, 3]:.3f}')
        else:
            valid_results = torch.tensor(self.valid_results)  # [num run, num epoch, num metric]
            test_results = torch.tensor(self.test_results)
            valid_results_, test_results_ = [], []
            for r1, r2 in zip(valid_results, test_results): # [num epoch, num metric]
                argmin = r1[:, 0].argmin().item()
                valid_results_.append(r1[argmin])
                test_results_.append(r2[argmin])

            valid_results = torch.stack(valid_results_, dim=0) # [num run, num metric]
            test_results = torch.stack(test_results_, dim=1)

            print(f'All runs:')
            print(f'Final Valid {self.metrics_name[0]}: {valid_results[:, 0].mean():.3f} ± {valid_results[:, 0].str():.3f}, '
                  f'{self.metrics_name[1]}: {valid_results[:, 1].mean():.3f} ± {valid_results[:, 1].str():.3f}, '
                  f'{self.metrics_name[2]}: {valid_results[:, 2].mean():.3f} ± {valid_results[:, 2].str():.3f}, '
                  f'{self.metrics_name[3]}: {valid_results[:, 3].mean():.3f} ± {valid_results[:, 3].str():.3f}')
            print(f'Final Test {self.metrics_name[0]}: {test_results[:, 0].mean():.3f} ± {test_results[:, 0].str():.3f}, '
                f'{self.metrics_name[1]}: {test_results[:, 1].mean():.3f} ± {test_results[:, 1].str():.3f}, '
                f'{self.metrics_name[2]}: {test_results[:, 2].mean():.3f} ± {test_results[:, 2].str():.3f}, '
                f'{self.metrics_name[3]}: {test_results[:, 3].mean():.3f} ± {test_results[:, 3].str():.3f}')

            if record:
                results = {f'valid {self.metrics_name[i]}': valid_results[:, i].tolist() for i in range(len(self.metrics_name))}
                results += {f'test {self.metrics_name[i]}': test_results[:, i].tolist() for i in range(len(self.metrics_name))}
                results = pd.DataFrame(results)
                results.to_csv(out_path, index=False)