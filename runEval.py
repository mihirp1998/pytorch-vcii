from eval_dataset import get_loader as eval_get_loader
from train_options import parser
from util import get_models, init_lstm, set_train, set_eval
import torch
from evaluate import run_eval
import time
from torch.autograd import Variable
def get_eval_loaders():
  eval_loader = eval_get_loader(is_train=False,root=args.eval, mv_dir=args.eval_mv,n_work=0,args=args)

  return eval_loader




def resume(load_name, index):
  names = ['encoder', 'binarizer', 'decoder', 'unet']
  for net_idx, net in enumerate(nets):
    if net is not None:
      name = names[net_idx]
      checkpoint_path = '{}/{}_{}_epoch_{:08d}.pth'.format(args.model_dir,load_name, name, index)
      print('Loading %s from %s...' % (name, checkpoint_path))
      net.load_state_dict(torch.load(checkpoint_path))


args = parser.parse_args()
print(args)
#print(vid_count,"vid")
eval_loader = get_eval_loaders()
############### Model ###############
encoder, binarizer, decoder, unet = get_models(
  args=args, v_compress=args.v_compress, 
  bits=args.bits,
  encoder_fuse_level=args.encoder_fuse_level,
  decoder_fuse_level=args.decoder_fuse_level,num_vids=vid_count)

nets = [encoder, binarizer, decoder,unet]



just_resumed = False
if args.load_model_name:
    print('Loading %s@iter %d' % (args.load_model_name,
                                  args.load_iter))

    resume(args.load_model_name, args.load_iter)
    just_resumed = True



if just_resumed:
  print('Start evaluation...')
  set_eval(nets)

  eval_begin = time.time()
  
  # id_num = Variable(torch.tensor(2).cuda())
  # wenc,wdec,wbin,unet_kernels,unet_bias = hypernet(id_num)

  eval_loss, mssim, psnr = run_eval(nets, eval_loader, args,
      output_suffix='iter%d' % 0)

  print('Evaluation @iter %d done in %d secs' % (
      0, time.time() - eval_begin))
  print('%s Loss   : ' % '\t'.join(['%.5f' % el for el in eval_loss.tolist()]))
  print('%s MS-SSIM: ' %  '\t'.join(['%.5f' % el for el in mssim.tolist()]))
  print('%s PSNR   : ' % '\t'.join(['%.5f' % el for el in psnr.tolist()]))

