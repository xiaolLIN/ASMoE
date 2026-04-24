import argparse
import time
from recbole.utils.utils import ensure_dir
from recbole.utils.utils import dict2str
from recbole.quick_start import run_recbole
from recbole.utils.utils import get_local_time


def accu_per(over_metr, split_metr, degree, split_test_statis):
    metrics = ['recall@3', 'recall@5', 'recall@10', 'recall@20', 'ndcg@3', 'ndcg@5', 'ndcg@10', 'ndcg@20']
    split_user_data = split_test_statis
    contr_per_dict = { }
    all_user_num = 0
    for part in split_test_statis:
        all_user_num += split_test_statis[part]
    for metric in metrics:
        over_metr_sum = over_metr[metric] * all_user_num
        split_metr_sum = split_metr[metric] * split_user_data['new_degree-{}'.format(str(degree))]
        contr_percent = round(split_metr_sum * 100 / over_metr_sum, 3)
        contr_per_dict[metric] = contr_percent
    import numpy as np
    avg_contr = np.mean(list(contr_per_dict.values()))
    key_metr_avg_contr = np.mean([contr_per_dict['recall@10'], contr_per_dict['recall@20'], contr_per_dict['ndcg@10'],
                                  contr_per_dict['ndcg@20']])
    return dict2str(contr_per_dict), avg_contr, key_metr_avg_contr


def save_result_txt(run_result, result_path_prefix, desc, args):
    # desc: some exp setup of current model description
    result_file_path = result_path_prefix + '/' + str(args.model)
    with open(result_file_path + '.txt', 'a+') as f:
        f.write('stage: ' + str(args.two_stage + 1) + '\n')
        f.write('start_train_time: ' + run_result['start_train_time'] + '\n')  # help to check the log file.
        f.write('current time: ' + str(get_local_time()) + '\n')
        f.write('desc: ' + desc + '\n')
        f.write('stop_output: ' + str(run_result['stop_output']) + '\n')
        f.write('valid result:' + str(run_result['best_valid_result']) + '\n\n')

        over_metr = run_result['test_result']
        split_test_statis = run_result['split_test_statis']
        split_part_num = len(split_test_statis)
        all_user_num = sum(split_test_statis.values())

        for i in range(split_part_num):
            partition_no = split_part_num - i
            res = run_result['partition-{}_res'.format(str(partition_no))]
            if res is not None:
                f.write('      partition-{} res: {} \n'.format(str(partition_no), res))
                contr_per_str, avg_contr, key_metr_avg_contr = accu_per(over_metr, res, partition_no, split_test_statis)

                f.write('=> #user:{} \n')
                f.write('   contribution ratio(%): {}\n'.format(split_test_statis['new_degree-{}'.format(str(i))], contr_per_str))
                f.write('            avg ratio(%): {}, key metric avg ratio: {}\n\n'.format(str(avg_contr), str(key_metr_avg_contr)))

        f.write('#all user:{}, overall result: {}\n'.format(str(all_user_num), str(run_result['test_result'])))
        f.write('\n')


if __name__ == '__main__':
    begin = time.time()
    parameter_dict = {
        'neg_sampling': None
    }
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='SASRec', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='Amazon_Beauty', help='name of datasets')
    parser.add_argument('--config_files', type=str, default='configs/Amazon_Beauty.yaml', help='config files')
    parser.add_argument('--rm_dup_inter', type=str, default=None)

    parser.add_argument('--split_eval', type=int, default=0)  # eval on test subsets

    parser.add_argument('--desc', default='none', type=str, help='some exp setup of current model description')
    parser.add_argument('--tau_s1', type=float, default=0.5)  # the temperature of gumbel softmax in stage 1
    parser.add_argument('--eval_split_num', type=int, default=3)

    # asmoe s2
    parser.add_argument('--two_stage', type=int, default=0)
    parser.add_argument('--tau_s2', type=float, default=0.5)  # the temperature of gumbel softmax in stage 2
    parser.add_argument('--stage1_model_desc', type=str,
                        default='stage1_model_desc')  # the desc of stage 1 model
    parser.add_argument("--ft_strategy", type=str, default='all')
    parser.add_argument('--reg_lmd', type=float, default=0.1)
    parser.add_argument('--lora_rank', type=int, default=32)

    # empirical study
    parser.add_argument('--cp_part', type=int, default=0)
    parser.add_argument('--de_part', type=int, default=0)
    parser.add_argument('--aug_part', type=int, default=0)
    parser.add_argument('--retain_part', type=int, default=0)
    parser.add_argument("--retain_type", type=str, default='random')

    parser.add_argument('--def_hl', type=int, default=0)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--n_heads', type=int, default=8)

    parser.add_argument('--train_batch_size', type=int, default=256)
    parser.add_argument('--learning_rate', type=float, default=0.0001)

    args, _ = parser.parse_known_args()

    result_path_prefix = "run_results/" + str(args.dataset) + "/" + str(args.model)
    ensure_dir(result_path_prefix)

    config_file_list = args.config_files.strip().split(' ') if args.config_files else None
    run_result = run_recbole(model=args.model, dataset=args.dataset, config_file_list=config_file_list, config_dict=parameter_dict)
    end=time.time()
    print(end-begin)

    save_result_txt(run_result=run_result, result_path_prefix="run_results/" + str(args.dataset), desc=args.desc, args=args)
