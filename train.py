import sys
sys.path.append('core')

import argparse
import numpy as np
import datetime
from config.parser import parse_args

import torch
import torch.optim as optim
from colorama import Style, Fore, Back
from raft import RAFT
from gcsraft import GCSRAFT
from gcsraft_certainty import GCSRAFT_certainty
from raft_certainty import RAFT_certainty
from hraft import HomoRAFT
# from swinraft import SwinRAFT
from datasets import fetch_dataloader
from utils.utils import load_ckpt
from loss import *
from ddp_utils import *
from tqdm import tqdm
# TODO 适配其他数据集
from evaluate import *
import datasets
import torch.utils.data as data


from tensorboardX import SummaryWriter



def save_top_k(model, optimizer, scheduler, top_k_state, k, total_steps, save_dir, epe, distributed):
    flag = False
    popped_state = {}
    model_path = os.path.join(save_dir, 'total_steps_{}_EPE_{:.3f}'.format(total_steps, epe))

    if len(top_k_state) < k or epe < top_k_state[-1]['epe']:
	
        if len(top_k_state) >= k:
            popped_state = top_k_state.pop()
            os.remove(os.path.join(save_dir, 'total_steps_{}_EPE_{:.3f}'.format(popped_state['total_steps'], popped_state['epe'])))
        flag = True
        top_k_state.append({'total_steps': total_steps, 'epe': epe})
        if distributed == True:
            torch.save({'total_steps': total_steps, 'state_dict': model.module.state_dict(), 
            'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict()}, model_path)
        else:
            torch.save({'total_steps': total_steps, 'state_dict': model.state_dict(), 
            'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict()}, model_path)              
          
          

    top_k_state.sort(key = lambda s: s['epe'], reverse = True)
    if flag:
        if popped_state == {}:
            print(Back.RED + 'EPE: {:.3f} , length of buffer < {}'.format(epe, k))
        else:
            print(Back.RED + 'EPE: {:.3f}  <  last EPE: {:.3f}'.format(epe, popped_state['epe']))
        print('Save the better model!!!' + Style.RESET_ALL)
        print(Fore.RED + "---------------------------------------------------------------" + Style.RESET_ALL)
    else:
        print(Back.GREEN + 'EPE: {:.3f}  >=  rank-3 EPE: {:.3f}'.format(epe, top_k_state[-1]['epe']))
        print('Do not save this model, QQQ' + Style.RESET_ALL)
        print(Fore.RED + "---------------------------------------------------------------" + Style.RESET_ALL)

    return top_k_state

def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps + 100,
        pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')

    return optimizer, scheduler

def train(args, rank=0, world_size=1, use_ddp=False, current_time='test'):
    """ Full training loop """
    ckpt_path = os.path.join('checkpoints', args.name, current_time)
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path,exist_ok=True)    
    save_config_path = os.path.join('checkpoints', args.name, current_time, 'config.json')
    args_dict = {k: getattr(args, k) for k in vars(args)}
    import json
    with open(save_config_path, 'w') as f:
        json.dump(args_dict, f)
    device_id = rank
    useloss = sequence_loss
    if args.model == 'HomoRAFT':
        model = HomoRAFT(args).to(device_id)
        # load_ckpt(model.raft, args, distributed=False)
        # state = torch.load('models/MCNet_mscoco.pth', map_location='cpu')
        # model.coarsenet.load_state_dict(state['homo_model'])
    elif args.model == 'GCSRAFT_certainty':
        model = GCSRAFT_certainty(args).to(device_id)
    else:
        raise NotImplementedError
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model) # there might not be any, actually
    if use_ddp == True:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank], static_graph=True)

    model.train()

    # # 根据参数层的 name 来进行冻结
    # unfreeze_layers = ["update_block.certainty_head"] # 用列表
    # # 设置冻结参数：
    # for name, param in model.named_parameters():
    #     # print(name, param.shape)
    #     # 错误判定：
    #     # if name.split(".")[0] in unfreeze_layers: # 不要用in来判定，因为"id"也在"text_id"的in中。
    #     # 正确判定：
    #     for unfreeze_layer in unfreeze_layers:
    #         if unfreeze_layer in name:
    #             print(name, param.requires_grad)
    #         else:
    #             param.requires_grad = False
    #             print(name, param.requires_grad)


    train_loader = fetch_dataloader(args, rank=rank, world_size=world_size, use_ddp=use_ddp)
    optimizer, scheduler = fetch_optimizer(args, model)
    # if args.restore_ckpt is not None:
    #     if args.ckpt_load_mode == 'load_model_only':
    #         load_ckpt(model, args, use_ddp)
    #         print(f"restore ckpt from {args.restore_ckpt}")
    #     elif args.ckpt_load_mode == 'load_model_and_optimizer/scheduler':
    #         load_ckpt(model, args, use_ddp, optimizer, scheduler)
    #     else:
    #         raise NotImplementedError
    eval_rank = 0
    if use_ddp:
        eval_rank = 0
    total_steps = 0
    epoch = 0
    should_keep_training = True
    writer_path = os.path.join('checkpoints', args.name, current_time, 'log')
    # if not os.path.exists(writer_path) and rank == 0:
    #     os.makedirs(writer_path, exist_ok=True)    
    if rank == eval_rank:
        writer = SummaryWriter(writer_path)
    # if use_ddp == True:
    #     torch.autograd.set_detect_anomaly(True) # 这句话千万不能加，否则两卡训练会很慢，原因未知
    top_k_state = []
    while should_keep_training:
        if use_ddp == True:
            # shuffle sampler
            train_loader.sampler.set_epoch(epoch)
        epoch += 1
        loop = tqdm((train_loader),desc=f"Epoch: [{epoch}],Steps:[{total_steps}/{args.num_steps}]", total=len(train_loader))
        for i_batch, data_blob in enumerate(loop):
            model.train()
            optimizer.zero_grad()
            image1, image2, flow, valid = [x.cuda(non_blocking=True) for x in data_blob]
            output = model(image1, image2, flow_gt=flow, iters=args.iters)
            loss = useloss(output, flow, valid, args.ce_weight)
            loss['loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip) 
            optimizer.step()
            scheduler.step()
            if rank == eval_rank:
                writer.add_scalar("loss", loss['loss'], total_steps)
                if "flow_loss" in loss:
                    writer.add_scalar("flow_loss", loss['flow_loss'], total_steps)
                if 'certainty_loss' in loss:
                    writer.add_scalar("certainty_loss", loss['certainty_loss'], total_steps)
                if 'mace' in loss:
                    writer.add_scalar("mace", loss['mace'], total_steps)
            if total_steps % args.val_freq == args.val_freq - 1 and rank == eval_rank:
                print(eval_rank)
                print(rank)
                # eval(args) 最终改成这个样子
                with torch.no_grad():
                    model.eval()
                    save_topk = True
                    if args.dataset == 'dunhuang':
                        result = validate_dunhuang(args, model, dstype_list=['val_homo_nocolorchange512','val_homo+pertu_colorchange512'])
                        for dstype, value in result.items():
                            writer.add_scalar(dstype+"EPE", value['EPE'], total_steps)
                            writer.add_scalar(dstype+"1px", value['1px'], total_steps)
                            writer.add_scalar(dstype+"3px", value['3px'], total_steps)
                            writer.add_scalar(dstype+"5px", value['5px'], total_steps)
                            with open(os.path.join('checkpoints', args.name, current_time, dstype+'EPErecords.txt'), 'w') as file:
                                i = 0
                                for imgepe in value['epe_list']:
                                    file.write(str(i) + '\t' + str(imgepe) + "\n")
                        epe = result['val_homo+pertu_colorchange512']['EPE'] # 暂时用val
                    else:
                        save_topk = False
                    model.train()
                if save_topk == True:
                    top_k_state = save_top_k(model, optimizer, scheduler, top_k_state, 1, total_steps, 
                                            os.path.join('checkpoints', args.name, current_time), epe, use_ddp)
                PATH = os.path.join('checkpoints', args.name, current_time, 'latest_model')
                if use_ddp == True:
                    torch.save({'total_steps': total_steps, 'state_dict': model.module.state_dict(), 
					'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict()}, PATH)
                else:
                    torch.save({'total_steps': total_steps, 'state_dict': model.state_dict(), 
					'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict()}, PATH)            
            if total_steps > args.num_steps:
                should_keep_training = False
                break
            total_steps += 1



def main(rank, world_size, args, use_ddp, time):
    if use_ddp:
        print(f"Using DDP [{rank=} {world_size=}]")
        setup_ddp(rank, world_size)
        os.system("export KMP_INIT_AT_FORK=FALSE")
    train(args, rank=rank, world_size=world_size, use_ddp=use_ddp, current_time=time)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)
    args = parse_args(parser)
    # args.name += f"_exp{str(np.random.randint(100))}"
    # 调试
    # args.gpus = [0]
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1, 2, 3"
    smp, world_size = init_ddp()
    # world_size = 3
    # 获取对象的所有属性和方法
    attributes = vars(args)
    time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # 打印所有属性和方法
    for name, value in attributes.items():
        print(f"{name}: {value}")
    if world_size > 1:
        spwn_ctx = mp.spawn(main, nprocs=world_size, args=(world_size, args, True, time), join=False)
        spwn_ctx.join()
    else:
        main(0, 1, args, False, time)
    print("Done!")