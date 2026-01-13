import pytest
import torch

def test_mimic_video():
    from mimic_video.mimic_video import MimicVideo

    video_hiddens = torch.randn(2, 64, 77)
    video_mask = torch.randint(0, 2, (2, 64)).bool()

    mimic_video = MimicVideo(512, dim_video_hidden = 77)

    actions = torch.randn(2, 32, 20)

    joint_state = torch.randn(2, 32)

    forward_kwargs = dict(video_hiddens = video_hiddens, context_mask = video_mask, joint_state = joint_state)

    loss = mimic_video(actions, **forward_kwargs)

    assert loss.numel() == 1

    flow = mimic_video(actions, **forward_kwargs, time = torch.tensor([0.5, 0.5]))

    assert flow.shape == actions.shape
