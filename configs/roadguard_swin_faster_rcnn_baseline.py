# Controlled reference recipe: same Swin/FPN/Faster R-CNN architecture and split,
# but fixed 640 resize, default RPN anchor scale, and 24 epochs.
# Compare only after both runs complete; the reference does not guarantee any outcome.
_base_ = ['./roadguard_swin_faster_rcnn.py']

model = dict(
    rpn_head=dict(anchor_generator=dict(scales=[8])),
    train_cfg=dict(rpn_proposal=dict(nms_pre=2000, max_per_img=1000)),
    test_cfg=dict(rcnn=dict(score_thr=0.001, max_per_img=100)),
)
train_pipeline = [
    dict(type='LoadImageFromFile'), dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(640, 640), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5), dict(type='PackDetInputs'),
]
val_test_pipeline = [
    dict(type='LoadImageFromFile'), dict(type='Resize', scale=(640, 640), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor')),
]
train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(pipeline=val_test_pipeline))
test_dataloader = dict(dataset=dict(pipeline=val_test_pipeline))
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=24, val_interval=1)
optim_wrapper = dict(optimizer=dict(type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05),
                     paramwise_cfg=dict(_delete_=True, custom_keys={}),
                     clip_grad=dict(max_norm=1.0, norm_type=2))
param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(type='MultiStepLR', begin=0, end=24, by_epoch=True, milestones=[16, 22], gamma=0.1),
]
default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=3,
                                     save_best='coco/bbox_mAP', rule='greater'),
                     logger=dict(type='LoggerHook', interval=50))
randomness = dict(seed=42, deterministic=True)
work_dir = './work_dirs/roadguard_swin_baseline'
