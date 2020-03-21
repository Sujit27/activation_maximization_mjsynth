# IPython log file


import sys
sys.path.append("../library")
from deep_dream import *
net = DictNet(1000)
net.load_state_dict(torch.load("../models/net_1000_0.001_200_0.0.pth"))
dreamer = DeepDreamGAN(net,(1,32,128),(0.47,),(0.14,),use_gaussian_filter=True)
labels = [i for i in range(64)]
im = dreamer.set_batch_dream(labels,0)
dream_plain = dreamer.batch_dream_kernel(im,labels,100,0.1)
org_tensor = dream_plain.data.clone()

imprv_tensor = dreamer.batch_discrim_kernel(dream_plain,labels,50,0.1)

output_org = dreamer.discrim_net(org_tensor)
output_imprv = dreamer.discrim_net(imprv_tensor)

org_output_diff = output_org.data[:,0]-output_org.data[:,1]
imprv_output_diff = output_imprv.data[:,1]-output_imprv.data[:,0]
print(output_imprv.data)
print(np.sum(imprv_output_diff.cpu().numpy() > 0.5))

