import os

root = 'data'
train_dir='train'
test_dir='test'
image_dir = os.path.join(root, train_dir)

annotation_file=os.path.join(root,'annotation.json')

train_imageset_fn=os.listdir(image_dir)
test_imageset_fn=os.listdir(os.path.join(root,test_dir))

train_imageset_fn=list(map(lambda x:os.path.join(train_dir,x),train_imageset_fn))


val_image_dir=os.path.join(root, test_dir)
val_imageset_fn = list(map(lambda x:os.path.join(test_dir,x),test_imageset_fn))
image_ext = '.jpg'

backbone = 'resnet101'
classes = ['pl120',
 'il80',
 'pm30',
 'i2',
 'p12',
 'pl100',
 'w57',
 'p26',
 'ph5',
 'pne',
 'pl40',
 'pl50',
 'pl20',
 'pg',
 'i5',
 'p6',
 'p3',
 'pl80',
 'pl60',
 'pl70',
 'p27',
 'pm20',
 'pl30',
 'ph4',
 'pm55',
 'il60',
 'p19',
 'pr40',
 'i4',
 'p23',
 'pn',
 'w32',
 'pl5',
 'p11',
 'p5',
 'ph4.5',
 'ip',
 'w13',
 'w59',
 'il100',
 'p10',
 'w55']
mean, std = (0.499, 0.523, 0.532), (0.200, 0.202, 0.224)   
scale =None

batch_size =8
lr = 1e-4
momentum = 0.9
weight_decay =0.0002
num_epochs =18
lr_decay_epochs = [12]
num_workers = 8
width,height=512,512
eval_while_training = True
eval_every = 2
