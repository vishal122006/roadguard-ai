_base_ = ['mmdet::faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py']

# RoadGuard research candidate: Swin-T + FPN + Faster R-CNN.
# Any improvement over the prior YOLO model is a hypothesis until measured.
data_root = 'data/RDD2022_coco/'
metainfo = dict(classes=('D00', 'D10', 'D20', 'D40'))

model = dict(
    backbone=dict(
        _delete_=True, type='SwinTransformer', embed_dims=96,
        depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
        window_size=7, mlp_ratio=4, qkv_bias=True, qk_scale=None,
        drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.2,
        patch_norm=True, out_indices=(0, 1, 2, 3), with_cp=False,
        convert_weights=True,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth',
        ),
    ),
    neck=dict(_delete_=True, type='FPN', in_channels=[96, 192, 384, 768],
              out_channels=256, num_outs=5),
    # Denser anchors may improve small/thin-crack proposals but can increase false positives.
    rpn_head=dict(anchor_generator=dict(scales=[4, 8])),
    train_cfg=dict(rpn_proposal=dict(nms_pre=2000, max_per_img=2000)),
    roi_head=dict(bbox_head=dict(num_classes=4)),
    # Low inference score floor preserves candidates for validation threshold tuning.
    test_cfg=dict(
        rpn=dict(nms_pre=2000, max_per_img=1000),
        rcnn=dict(score_thr=0.001, nms=dict(type='nms', iou_threshold=0.5), max_per_img=300),
    ),
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PhotoMetricDistortion'),
    dict(type='RandomChoiceResize', scales=[(640, 640), (704, 704), (768, 768), (800, 800)], keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs'),
]
val_test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(800, 800), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor')),
]

train_dataloader = dict(
    batch_size=2, num_workers=2, persistent_workers=True,
    dataset=dict(_delete_=True, type='CocoDataset', data_root=data_root,
                 ann_file='train.json', data_prefix=dict(img='images/'),
                 metainfo=metainfo, filter_cfg=dict(filter_empty_gt=False, min_size=16),
                 pipeline=train_pipeline),
)
val_dataloader = dict(
    batch_size=1, num_workers=2, persistent_workers=True, drop_last=False,
    dataset=dict(_delete_=True, type='CocoDataset', data_root=data_root,
                 ann_file='val.json', data_prefix=dict(img='images/'),
                 metainfo=metainfo, test_mode=True, pipeline=val_test_pipeline),
)
test_dataloader = dict(
    batch_size=1, num_workers=2, persistent_workers=True, drop_last=False,
    dataset=dict(_delete_=True, type='CocoDataset', data_root=data_root,
                 ann_file='test.json', data_prefix=dict(img='images/'),
                 metainfo=metainfo, test_mode=True, pipeline=val_test_pipeline),
)
val_evaluator = dict(type='CocoMetric', ann_file=data_root + 'val.json', metric='bbox', classwise=True)
test_evaluator = dict(type='CocoMetric', ann_file=data_root + 'test.json', metric='bbox', classwise=True)

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=36, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05),
    paramwise_cfg=dict(custom_keys={
        'absolute_pos_embed': dict(decay_mult=0.0),
        'relative_position_bias_table': dict(decay_mult=0.0),
        'norm': dict(decay_mult=0.0),
        'backbone': dict(lr_mult=0.1),
    }),
    clip_grad=dict(max_norm=1.0, norm_type=2),
)
param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(type='CosineAnnealingLR', begin=0, end=36, by_epoch=True, eta_min=1e-6),
]
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=3,
                    save_best='coco/bbox_mAP', rule='greater'),
    logger=dict(type='LoggerHook', interval=50),
)
default_scope = 'mmdet'
randomness = dict(seed=42, deterministic=True)
work_dir = './work_dirs/roadguard_swin'
