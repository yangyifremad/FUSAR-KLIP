backbone_norm_cfg = dict(eps=1e-06, requires_grad=True, type='LN')

crop_size = (
    256,
    256,
)
data_preprocessor = dict(
    bgr_to_rgb=True,
    mean=[
        127.5,
        127.5,
        127.5,
    ],
    pad_val=0,
    seg_pad_val=255,
    size=(
        256,
        256,
    ),
    std=[
        127.5,
        127.5,
        127.5,
    ],
    type='SegDataPreProcessor')
data_root = ''
dataset_type = 'CityscapesDataset'
default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=2000, type='CheckpointHook'),
    logger=dict(interval=50, log_metric_by_epoch=False, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='SegVisualizationHook'))
default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
launcher = 'pytorch'
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=False)
model = dict(
    backbone=dict(
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        drop_rate=0.0,
        embed_dims=768,
        final_norm=True,
        img_size=(
            256,
            256,
        ),
        in_channels=3,
        interpolate_mode='bicubic',
        norm_cfg=dict(eps=1e-06, requires_grad=True, type='LN'),
        num_heads=12,
        num_layers=12,
        patch_size=16,
        type='VisionTransformer',
        with_cls_token=True),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            127.5,
            127.5,
            127.5,
        ],
        pad_val=0,
        seg_pad_val=255,
        size=(
            256,
            256,
        ),
        std=[
            127.5,
            127.5,
            127.5,
        ],
        type='SegDataPreProcessor'),
    decode_head=dict(
        channels=768,
        dropout_ratio=0.0,
        embed_dims=768,
        in_channels=768,
        loss_decode=dict(
            loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=False),
        num_classes=4,
        num_heads=12,
        num_layers=2,
        type='SegmenterMaskTransformerHead'),
    pretrained=
    '',
    test_cfg=dict(crop_size=(
        256,
        256,
    ), mode='slide', stride=(
        256,
        256,
    )),
    type='EncoderDecoder')
optim_wrapper = dict(
    clip_grad=None,
    optimizer=dict(lr=0.001, momentum=0.9, type='SGD', weight_decay=0.0),
    type='OptimWrapper')
optimizer = dict(lr=0.001, momentum=0.9, type='SGD', weight_decay=0.0)
param_scheduler = [
    dict(
        begin=0,
        by_epoch=False,
        end=12000,
        eta_min=0.0001,
        power=0.9,
        type='PolyLR'),
]
pretrained = ''
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=10,
    dataset=dict(
        data_prefix=dict(img_path='test_images', seg_map_path='test_labels'),
        data_root='',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(reduce_zero_label=False, type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type='CityscapesDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    iou_metrics=[
        'mIoU',
    ], type='IoUMetric')
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(reduce_zero_label=False, type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]
train_cfg = dict(max_iters=12000, type='IterBasedTrainLoop', val_interval=2000)
train_dataloader = dict(
    batch_size=60,
    dataset=dict(
        data_prefix=dict(img_path='train_images', seg_map_path='train_labels'),
        data_root='',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(reduce_zero_label=False, type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type='CityscapesDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='InfiniteSampler'))
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(reduce_zero_label=False, type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]
tta_model = dict(type='SegTTAModel')
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=10,
    dataset=dict(
        data_prefix=dict(img_path='test_images', seg_map_path='test_labels'),
        data_root='',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(reduce_zero_label=False, type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type='CityscapesDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    iou_metrics=[
        'mIoU',
    ], type='IoUMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='SegLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = ''
