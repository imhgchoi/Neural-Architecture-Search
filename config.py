import argparse

def get_args():
    argp = argparse.ArgumentParser(description='Neural Architecture Search', 
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ## General
    argp.add_argument('--controller', type=str, default="enas", choices=["enas", "enas_light"])
    argp.add_argument('--mode', type=str, default="fix", choices=["train", "fix", "test"])
    argp.add_argument('--fixed-arch-dir', type=str, default='./save/states/architecture.tar')
    argp.add_argument('--task-type', type=str, default="vision", choices=["vision","text"])
    argp.add_argument('--dataset', type=str, default="cifar10", choices=["mnist","cifar10","imagenet"])
    argp.add_argument('--light-mode', action='store_true', default=False)
    argp.add_argument('--print-step', type=int, default=5)
    argp.add_argument('--debug', action='store_true', default=False)
    argp.add_argument('--seed', type=int, default=1111)
    argp.add_argument('--restart', action='store_true', default=False)
    argp.add_argument('--restart-fix', action='store_true', default=False)


    ## Optimization
    argp.add_argument('--batch-size', type=int, default=128)
    argp.add_argument('--algo', type=str, default="pg", choices=["pg","ppo"])
    argp.add_argument('--epochs', type=int, default=310)
    argp.add_argument('--fixed-epochs', type=int, default=700)
    argp.add_argument('--controller-max-steps', type=int, default=2000)
    argp.add_argument('--dropout', type=float, default=0.1)
    argp.add_argument('--controller-lr', type=float, default=0.00035) # .00035 -> .001
    argp.add_argument('--worker-lr-scheduler', type=str, default="cosine", choices=["lambda", "cosine"])
    argp.add_argument('--worker-min-lr', type=float, default=0.0005) # 0.001 -> 0.0005
    argp.add_argument('--worker-max-lr', type=float, default=0.05)
    argp.add_argument('--worker-l2-weight-decay', type=float, default=0.0001) # 0.0005 -> 0.00025
    argp.add_argument('--worker-lr-T', type=int, default=10)
    argp.add_argument('--ppo-eps', type=float, default=0.2)
    argp.add_argument('--ppo-K', type=float, default=10)
    

    ## Model General
    argp.add_argument('--worker-type', type=str, default='macro', choices=['macro','micro'])
    argp.add_argument('--macro-num-layers', '--L', type=int, default=12)
    argp.add_argument('--sample-pool-size', type=int, default=10)

    ## Controller
    argp.add_argument('--lstm-size', type=int, default=50) ## 100->50
    argp.add_argument('--lstm-num-layers', type=int, default=1)
    argp.add_argument('--skip-target', type=float, default=0.4)
    argp.add_argument('--skip-weight', type=float, default=0.8)
    argp.add_argument('--temperature', type=float, default=5.0)
    argp.add_argument('--tanh-constant', type=float, default=2.5) # 2.5 -> 1.5
    argp.add_argument('--entropy-weight', type=float, default=0.001) # 0.1 -> 0.0001
    argp.add_argument('--grad-clip', type=float, default=0.25)
    argp.add_argument('--bl-decay', type=float, default=0.99)
    argp.add_argument('--controller-batch-size', type=int, default=16)


    ## Worker - CNN
    argp.add_argument('--cnn-first-layer-outdim','--cod', type=int, default=24)
    argp.add_argument('--cnn-first-layer-pad','--cp', type=int, default=1)
    argp.add_argument('--cnn-first-layer-kernel','--ck', type=int, default=3)


    return argp.parse_args()
