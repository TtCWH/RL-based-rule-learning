# author = 'Han Wang'
# coding = utf-8

from argparse import ArgumentParser

def get_args():
    parser = ArgumentParser(description='rl based rule learning model')

    parser.add_argument('--special_token', type=str, default='', help='train with special setting')

    parser.add_argument('--saved_model_path', type=str, default='./saved_model', help='trained model saving path')
    parser.add_argument('--result_path', type=str, default='./result', help='model output path')
    parser.add_argument('--local_data_path', type=str, default='./data', help='local path of data')
    parser.add_argument('--folder_id', type=str, default='', help='')

    parser.add_argument('--max_len', type=int, default=60)
    parser.add_argument('--lhs_len', type=int, default=5)

    parser.add_argument('--seed', type=int, default=10086, help='the answer to life, the universe and everything.')

    parser.add_argument('--generate_rules', type=int, default=1, help='generate rules or train models')

    parser.add_argument('--fasttext_epochs', type=int, default=50)
    parser.add_argument('--agent_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--dropout_rate', type=float, default=0.5)

    parser.add_argument('--num_rollouts', type=int, default=3)
    parser.add_argument('--p_embedding_size', type=int, default=200)
    parser.add_argument('--v_embedding_size', type=int, default=100)
    parser.add_argument('--s_embedding_size', type=int, default=600)
    parser.add_argument('--hidden_size', type=int, default=600)

    # hyper
    parser.add_argument('--entropy_prop', type=float, default=3e-2)
    parser.add_argument('--entropy_min', type=float, default=1e-30)
    parser.add_argument('--hc_threshold', type=float, default=0.2)
    parser.add_argument('--conf_threshold', type=float, default=0.8)

    args = parser.parse_args()
    return args
