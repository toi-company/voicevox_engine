import copy
from logging import getLogger
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pyopenjtalk import tts
from soxr import resample

from ...metas.Metas import StyleId
from ...model import AudioQuery
from ...tts_pipeline.tts_engine import TTSEngine, to_flatten_moras
from ..core.mock import MockCoreWrapper
from coeirocore.coeiro_manager import AudioManager
from coeirocore.query_manager import query2tokens_prosody


class MockTTSEngine(TTSEngine):
    """製品版コア無しに音声合成が可能なモック版TTSEngine"""

    def __init__(self):
        super().__init__(MockCoreWrapper())

        self.default_sampling_rate = 44100

        self.audio_manager = AudioManager(
            fs=self.default_sampling_rate,
            use_gpu=False
        )

    def synthesize_wave(
        self,
        query: AudioQuery,
        style_id: StyleId,
        enable_interrogative_upspeak: bool = True,
    ) -> NDArray[np.float32]:
        """音声合成用のクエリに含まれる読み仮名に基づいてOpenJTalkで音声波形を生成する"""
        tokens = query2tokens_prosody(query)
        return self.audio_manager.synthesis(
            text=tokens,
            style_id=style_id,
            speed_scale=query.speedScale,
            volume_scale=query.volumeScale,
            pitch_scale=query.pitchScale,
            intonation_scale=query.intonationScale,
            pre_phoneme_length=query.prePhonemeLength,
            post_phoneme_length=query.postPhonemeLength,
            output_sampling_rate=query.outputSamplingRate
        )
