# from utils import evaluate_synset, get_dataset
from utils import get_loops, get_dataset, get_network, get_eval_pool, evaluate_synset, get_daparam, get_time, TensorDataset, epoch, DiffAugment, ParamDiffAug, get_attention
from networks import ResNet18
import argparse
import torch
import torch.distributed as dist
import torch.cuda.comm

def main():
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--dataset', type=str, default='MHIST', help='dataset')
    parser.add_argument('--model', type=str, default='ConvNetD7', help='model')
    parser.add_argument('--ipc', type=int, default=50, help='image(s) per class')
    parser.add_argument('--eval_mode', type=str, default='SS', help='eval_mode')
    parser.add_argument('--num_exp', type=int, default=1, help='the number of experiments')
    parser.add_argument('--num_eval', type=int, default=200, help='the number of evaluating randomly initialized models')
    parser.add_argument('--epoch_eval_train', type=int, default=1000, help='epochs to train a model with synthetic data')
    parser.add_argument('--Iteration', type=int, default=10, help='training iterations')
    parser.add_argument('--lr_img', type=float, default=0.1, help='learning rate for updating synthetic images, 1 for low IPCs 10 for >= 100')
    parser.add_argument('--lr_net', type=float, default=0.01, help='learning rate for updating network parameters')
    parser.add_argument('--batch_real', type=int, default=128, help='batch size for real data')
    parser.add_argument('--batch_train', type=int, default=128, help='batch size for training networks')
    parser.add_argument('--init', type=str, default='real', help='noise/real/smart: initialize synthetic images from random noise or randomly sampled real images.')
    parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate', help='differentiable Siamese augmentation strategy')
    parser.add_argument('--data_path', type=str, default='C:/Users/samsa/Downloads/ECE1512_2024F_ProjectA_submission_files/submission_files/mhist_dataset/', help='dataset path')
    parser.add_argument('--zca', type=bool, default=False, help='Zca Whitening')
    parser.add_argument('--save_path', type=str, default='./data_h', help='path to save results')
    parser.add_argument('--dd', type=str, default='./data_h', help='dataset distillation path')
    parser.add_argument('--task_balance', type=float, default=0.01, help='balance attention with output')
    args = parser.parse_args()
    args.method = 'DataDAM'
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.dsa_param = ParamDiffAug()
    args.dsa = False if args.dsa_strategy in ['none', 'None'] else True

    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader, zca = get_dataset(args.dataset, args.data_path, args)
    net = ResNet18(channel=channel, num_classes=num_classes)
    tmp = torch.load(args.dd)
    image_syn_eval = tmp['data'][-1][0]
    label_syn_eval = tmp['data'][-1][1]
    for i in range(3):
        mini_net, acc_train, acc_test = evaluate_synset(i, net, image_syn_eval, 
                                                        label_syn_eval, testloader, args)

        print(mini_net, acc_train, acc_test)
if __name__ == '__main__':
    main()


