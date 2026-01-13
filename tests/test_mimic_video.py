import pytest
import torch

def test_mimic_video():
    from mimic_video.mimic_video import MimicVideo

    mimic_video = MimicVideo(512)

    actions = torch.randn(1, 32, 20)

    flow = mimic_video(actions)

    assert flow.shape == actions.shape
