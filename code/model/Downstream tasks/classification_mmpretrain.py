auto_scale_lr = dict(base_batch_size=28)
data_preprocessor = dict(
    mean=[
        127.5,
        127.5,
        127.5,
    ], std=[
        127.5,
        127.5,
        127.5,
    ], to_rgb=True)
dataset_type = 'ImageNe'
default_hooks = dict(
    checkpoint=dict(interval=25, type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(enable=False, type='VisualizationHook'))
default_scope = 'mmpretrain'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
launcher = 'pytorch'
load_from = None
log_level = 'INFO'
model = dict(
    backbone=dict(
        arch='b',
        drop_rate=0.1,
        img_size=512,
        init_cfg=dict(
            checkpoint=
            '',
            prefix='backbone',
            type='Pretrained'),
        patch_size=16,
        type='VisionTransformer'),
    head=dict(
        in_channels=768,
        loss=dict(
            label_smooth_val=0.1, mode='classy_vision',
            type='LabelSmoothLoss'),
        num_classes=7,
        type='VisionTransformerClsHead'),
    neck=None,
    type='ImageClassifier')
optim_wrapper = dict(
    optimizer=dict(lr=0.003, type='AdamW', weight_decay=0.3),
    paramwise_cfg=dict(
        custom_keys=dict({
            '.cls_token': dict(decay_mult=0.0),
            '.pos_embed': dict(decay_mult=0.0)
        })))
param_scheduler = [
    dict(
        begin=0,
        by_epoch=True,
        convert_to_iter_based=True,
        end=30,
        start_factor=0.0001,
        type='LinearLR'),
    dict(
        T_max=70, begin=30, by_epoch=True, end=100, type='CosineAnnealingLR'),
]
pretrained = ''
randomness = dict(deterministic=False, seed=None)
resume = False
test_cfg = dict()
test_dataloader = dict(
    batch_size=28,
    dataset=dict(
        data_root='',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(backend='pillow', edge='short', scale=512, type='ResizeEdge'),
            dict(crop_size=512, type='CenterCrop'),
            dict(type='PackInputs'),
        ],
        split='test',
        type='ImageNe'),
    num_workers=5,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    topk=(
        1,
        3,
    ), type='Accuracy')
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(backend='pillow', edge='short', scale=512, type='ResizeEdge'),
    dict(crop_size=512, type='CenterCrop'),
    dict(type='PackInputs'),
]
train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=25)
train_dataloader = dict(
    batch_size=28,
    dataset=dict(
        data_root='',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(backend='pillow', scale=512, type='RandomResizedCrop'),
            dict(direction='horizontal', prob=0.5, type='RandomFlip'),
            dict(type='PackInputs'),
        ],
        split='train',
        type='ImageNe'),
    num_workers=5,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(backend='pillow', scale=512, type='RandomResizedCrop'),
    dict(direction='horizontal', prob=0.5, type='RandomFlip'),
    dict(type='PackInputs'),
]
val_cfg = dict()
val_dataloader = dict(
    batch_size=28,
    dataset=dict(
        data_root='',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(backend='pillow', edge='short', scale=512, type='ResizeEdge'),
            dict(crop_size=512, type='CenterCrop'),
            dict(type='PackInputs'),
        ],
        split='test',
        type='ImageNe'),
    num_workers=5,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    topk=(
        1,
        3,
    ), type='Accuracy')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    type='UniversalVisualizer', vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = ''
