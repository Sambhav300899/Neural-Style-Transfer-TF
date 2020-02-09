import argparse
from model import style_transfer

if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    ap.add_argument('--style', help = 'path to style image', required = True)
    ap.add_argument('--content', help = 'path to content image', required = True)
    ap.add_argument('--backbone', help = 'backbone to be used', required = True)
    ap.add_argument('--epochs', help = 'number of epochs', required = False, default = 1000, type = int)
    ap.add_argument('--lr', help = 'learning rate', required = False, default = 5, type = int)
    ap.add_argument('--output', help = 'path to save output image', required = False, default = 'output.png')

    args = vars(ap.parse_args())

    style_transfer_net = style_transfer(args['backbone'], args['content'], args['style'])
    style_transfer_net.combine(args['epochs'], args['lr'], args['output'])
