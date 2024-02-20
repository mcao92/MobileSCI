import argparse 
import torch 

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",type=int,default=100)
    parser.add_argument("--batch_size",type=int,default=4)
    parser.add_argument("--num_workers",type=int,default=4)
    parser.add_argument("--lr",type=float,default=0.0001)
    parser.add_argument("--log_dir",type=str,default="log")
    parser.add_argument("--save_model_step",type=int,default=1)
    parser.add_argument("--save_dir",type=str,default="results")
    parser.add_argument("--save_image_step",type=int,default=100)
    parser.add_argument("--device",type=str,default="cuda")
    parser.add_argument("--show_step",type=int,default=60)
    parser.add_argument("--test_flag",type=bool,default=True)
    parser.add_argument("--color_ch",type=int,default=1)
    parser.add_argument("--pretrained_teacher_model",type=str,default="checkpoints/Efficient_small_15.pth")
    # parser.add_argument("--pre_train_model",type=str,default="baseline_depth_6_width_64-dist-depth_24_width_64_4gpus_200epochs/epoch_199.pth")
    parser.add_argument("--pre_train_model",type=str,default=None)
    parser.add_argument("--weight_path",type=str,default="baseline_depth6_width64_efficient_dist_1gpu_random")
    parser.add_argument("--train_data_path",type=str,default="/home/caomiao/datasets/DAVIS/DAVIS-480/JPEGImages/480p")
    parser.add_argument("--test_data_path",type=str,default="simulation")
    parser.add_argument("--distributed",type=bool,default=False)
    parser.add_argument("--local_rank",default=-1)

    args = parser.parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    local_rank = int(args.local_rank) 
    if args.distributed:
        args.device = torch.device("cuda",local_rank)
    return args

