# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# Ref: https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/evaluation.py

import os
from .hook import Hook


class EvaluationHook(Hook):
    """
    Evaluation Hook for validation during training
    """
    
    def after_train_step(self, algorithm):
        if self.every_n_iters(algorithm, algorithm.num_eval_iter) or self.is_last_iter(algorithm):
            algorithm.print_fn("validating...")
            eval_dict = algorithm.evaluate('eval')
            algorithm.log_dict.update(eval_dict)

            # update best metrics
            if algorithm.log_dict['eval/top-1-acc'] > algorithm.best_eval_acc:
                algorithm.best_eval_acc = algorithm.log_dict['eval/top-1-acc']
                algorithm.best_it_acc = algorithm.it
            if algorithm.log_dict['eval/mae_all'] < algorithm.best_eval_mae_all:
                algorithm.best_eval_mae_all = algorithm.log_dict['eval/mae_all']
                algorithm.best_it_mae_all = algorithm.it
            if algorithm.log_dict['eval/mae_3_4'] < algorithm.best_eval_mae_3_4:
                algorithm.best_eval_mae_3_4 = algorithm.log_dict['eval/mae_3_4']
                algorithm.best_it_mae_3_4 = algorithm.it
            if algorithm.log_dict['eval/mae_3_4_reg'] < algorithm.best_eval_mae_3_4_reg:
                algorithm.best_eval_mae_3_4_reg = algorithm.log_dict['eval/mae_3_4_reg']
                algorithm.best_it_mae_3_4_reg = algorithm.it
            if algorithm.log_dict['eval/mae_all_reg'] < algorithm.best_eval_mae_all_reg:
                algorithm.best_eval_mae_all_reg = algorithm.log_dict['eval/mae_all_reg']
                algorithm.best_it_mae_all_reg = algorithm.it
            print('DEBUG: ', algorithm.log_dict)
    def after_run(self, algorithm):
        
        if not algorithm.args.multiprocessing_distributed or (algorithm.args.multiprocessing_distributed and algorithm.args.rank % algorithm.ngpus_per_node == 0):
            save_path = os.path.join(algorithm.save_dir, algorithm.save_name)
            algorithm.save_model('latest_model.pth', save_path)

        results_dict = {'eval/best_acc': algorithm.best_eval_acc, 'eval/best_it_acc': algorithm.best_it_acc}
        results_dict = {'eval/best_mae_3_4': algorithm.best_eval_mae_3_4, 'eval/best_it_mae_3_4': algorithm.best_it_mae_3_4}
        results_dict = {'eval/best_mae_3_4_reg': algorithm.best_eval_mae_3_4_reg, 'eval/best_it_mae_3_4_reg': algorithm.best_it_mae_3_4_reg}
        results_dict = {'eval/best_mae_all': algorithm.best_eval_mae_all, 'eval/best_it_mae_all': algorithm.best_it_mae_all}
        results_dict = {'eval/best_mae_all_reg': algorithm.best_eval_mae_all_reg, 'eval/best_it_mae_all_reg': algorithm.best_it_mae_all_reg}
        if 'test' in algorithm.loader_dict:
            # load the best model and evaluate on test dataset
            best_model_path = os.path.join(algorithm.args.save_dir, algorithm.args.save_name, 'model_best.pth')
            algorithm.load_model(best_model_path)
            test_dict = algorithm.evaluate('test')
            results_dict['test/best_acc'] = test_dict['test/top-1-acc']
        algorithm.results_dict = results_dict
        