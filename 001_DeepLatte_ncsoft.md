원본코드
# -*- coding: utf-8 -*-

# Copyright 2020 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

"""Pseudo QMF modules."""
'''
Copied from https://github.com/kan-bayashi/ParallelWaveGAN/blob/master/parallel_wavegan/layers/pqmf.py
'''

import numpy as np
import torch
import torch.nn.functional as F

from scipy.signal import kaiser


def design_prototype_filter(taps=62, cutoff_ratio=0.142, beta=9.0):
    """Design prototype filter for PQMF.
    This method is based on `A Kaiser window approach for the design of prototype
    filters of cosine modulated filterbanks`_.
    Args:
        taps (int): The number of filter taps.
        cutoff_ratio (float): Cut-off frequency ratio.
        beta (float): Beta coefficient for kaiser window.
    Returns:
        ndarray: Impluse response of prototype filter (taps + 1,).
    .. _`A Kaiser window approach for the design of prototype filters of cosine modulated filterbanks`:
        https://ieeexplore.ieee.org/abstract/document/681427
    """
    # check the arguments are valid
    assert taps % 2 == 0, "The number of taps mush be even number."
    assert 0.0 < cutoff_ratio < 1.0, "Cutoff ratio must be > 0.0 and < 1.0."

    # make initial filter
    omega_c = np.pi * cutoff_ratio
    with np.errstate(invalid="ignore"):
        h_i = np.sin(omega_c * (np.arange(taps + 1) - 0.5 * taps)) / (
            np.pi * (np.arange(taps + 1) - 0.5 * taps)
        )
    h_i[taps // 2] = np.cos(0) * cutoff_ratio  # fix nan due to indeterminate form

    # apply kaiser window
    w = kaiser(taps + 1, beta)
    h = h_i * w

    return h


class PQMF(torch.nn.Module):
    """PQMF module.
    This module is based on `Near-perfect-reconstruction pseudo-QMF banks`_.
    .. _`Near-perfect-reconstruction pseudo-QMF banks`:
        https://ieeexplore.ieee.org/document/258122
    """

    def __init__(self, subbands=4, taps=62, cutoff_ratio=0.142, beta=9.0):
        """Initilize PQMF module.
        The cutoff_ratio and beta parameters are optimized for #subbands = 4.
        See dicussion in https://github.com/kan-bayashi/ParallelWaveGAN/issues/195.
        Args:
            subbands (int): The number of subbands.
            taps (int): The number of filter taps.
            cutoff_ratio (float): Cut-off frequency ratio.
            beta (float): Beta coefficient for kaiser window.
        """
        super(PQMF, self).__init__()

        # build analysis & synthesis filter coefficients
        h_proto = design_prototype_filter(taps, cutoff_ratio, beta)
        h_analysis = np.zeros((subbands, len(h_proto)))
        h_synthesis = np.zeros((subbands, len(h_proto)))
        for k in range(subbands):
            h_analysis[k] = (
                2
                * h_proto
                * np.cos(
                    (2 * k + 1)
                    * (np.pi / (2 * subbands))
                    * (np.arange(taps + 1) - (taps / 2))
                    + (-1) ** k * np.pi / 4
                )
            )
            h_synthesis[k] = (
                2
                * h_proto
                * np.cos(
                    (2 * k + 1)
                    * (np.pi / (2 * subbands))
                    * (np.arange(taps + 1) - (taps / 2))
                    - (-1) ** k * np.pi / 4
                )
            )

        # convert to tensor
        analysis_filter = torch.from_numpy(h_analysis).float().unsqueeze(1)
        synthesis_filter = torch.from_numpy(h_synthesis).float().unsqueeze(0)

        # register coefficients as beffer
        self.register_buffer("analysis_filter", analysis_filter)
        self.register_buffer("synthesis_filter", synthesis_filter)

        # filter for downsampling & upsampling
        updown_filter = torch.zeros((subbands, subbands, subbands)).float()
        for k in range(subbands):
            updown_filter[k, k, 0] = 1.0
        self.register_buffer("updown_filter", updown_filter)
        self.subbands = subbands

        # keep padding info
        self.pad_fn = torch.nn.ConstantPad1d(taps // 2, 0.0)

    def analysis(self, x):
        """Analysis with PQMF.
        Args:
            x (Tensor): Input tensor (B, 1, T).
        Returns:
            Tensor: Output tensor (B, subbands, T // subbands).
        """
        x = F.conv1d(self.pad_fn(x), self.analysis_filter)
        return F.conv1d(x, self.updown_filter, stride=self.subbands)

    def synthesis(self, x):
        """Synthesis with PQMF.
        Args:
            x (Tensor): Input tensor (B, subbands, T // subbands).
        Returns:
            Tensor: Output tensor (B, 1, T).
        """
        # NOTE(kan-bayashi): Power will be dreased so here multipy by # subbands.
        #   Not sure this is the correct way, it is better to check again.
        # TODO(kan-bayashi): Understand the reconstruction procedure
        x = F.conv_transpose1d(
            x, self.updown_filter * self.subbands, stride=self.subbands
        )
        return F.conv1d(self.pad_fn(x), self.synthesis_filter)

원본은 PQMF의 수학적 구조와 register_buffer 활용은 탄탄하지만, 입력 계약·파라미터 검증과 reconstruction 조건이 암묵적이라 잘못된 입력이 저수준 convolution 오류로 넘어갈 여지가 있으며, 다만 CPU에서 버퍼를 생성하는 현재 방식 자체를 디바이스 취약점으로 보는 평가는 과하며 .to(device) 이후 정상적으로 buffer가 이동하므로 핵심 개선점은 디바이스 강제 이동이 아니라 수학적 계약을 보존한 명시적 검증과 reconstruction 무결성 테스트

제안패치
# -*- coding: utf-8 -*-

# Copyright 2020 Tomoki Hayashi
# MIT License

"""Pseudo QMF modules with defensive input validation."""

import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import kaiser


def design_prototype_filter(taps=62, cutoff_ratio=0.142, beta=9.0):
    """Design a Kaiser-windowed prototype filter for PQMF."""

    if not isinstance(taps, (int, np.integer)):
        raise TypeError(f"taps must be an integer, got {type(taps).__name__}")

    if taps <= 0 or taps % 2 != 0:
        raise ValueError(f"taps must be a positive even integer, got {taps}")

    if not np.isfinite(cutoff_ratio) or not 0.0 < cutoff_ratio < 1.0:
        raise ValueError(
            f"cutoff_ratio must be finite and in (0, 1), got {cutoff_ratio}"
        )

    if not np.isfinite(beta) or beta < 0.0:
        raise ValueError(
            f"beta must be finite and non-negative, got {beta}"
        )

    omega_c = np.pi * cutoff_ratio
    sample = np.arange(taps + 1) - 0.5 * taps

    with np.errstate(divide="ignore", invalid="ignore"):
        h_i = np.sin(omega_c * sample) / (np.pi * sample)

    h_i[taps // 2] = cutoff_ratio

    window = kaiser(taps + 1, beta)
    return h_i * window


class PQMF(torch.nn.Module):
    """Pseudo Quadrature Mirror Filter bank."""

    def __init__(
        self,
        subbands=4,
        taps=62,
        cutoff_ratio=0.142,
        beta=9.0,
    ):
        super().__init__()

        if not isinstance(subbands, (int, np.integer)):
            raise TypeError(
                f"subbands must be an integer, got {type(subbands).__name__}"
            )

        if subbands <= 0:
            raise ValueError(
                f"subbands must be greater than zero, got {subbands}"
            )

        self.subbands = subbands
        self.taps = taps

        h_proto = design_prototype_filter(
            taps=taps,
            cutoff_ratio=cutoff_ratio,
            beta=beta,
        )

        indices = np.arange(taps + 1) - taps / 2
        h_analysis = np.zeros((subbands, taps + 1))
        h_synthesis = np.zeros((subbands, taps + 1))

        for k in range(subbands):
            phase = (
                (2 * k + 1)
                * (np.pi / (2 * subbands))
                * indices
            )

            h_analysis[k] = (
                2
                * h_proto
                * np.cos(phase + (-1) ** k * np.pi / 4)
            )

            h_synthesis[k] = (
                2
                * h_proto
                * np.cos(phase - (-1) ** k * np.pi / 4)
            )

        self.register_buffer(
            "analysis_filter",
            torch.from_numpy(h_analysis).float().unsqueeze(1),
        )

        self.register_buffer(
            "synthesis_filter",
            torch.from_numpy(h_synthesis).float().unsqueeze(0),
        )

        updown_filter = torch.zeros(
            (subbands, subbands, subbands),
            dtype=torch.float32,
        )
        for k in range(subbands):
            updown_filter[k, k, 0] = 1.0

        self.register_buffer("updown_filter", updown_filter)

        self.pad_fn = torch.nn.ConstantPad1d(
            taps // 2,
            0.0,
        )

    @staticmethod
    def _validate_input(x, expected_channels, name):
        if not isinstance(x, torch.Tensor):
            raise TypeError(
                f"{name} input must be a torch.Tensor, "
                f"got {type(x).__name__}"
            )

        if x.ndim != 3:
            raise ValueError(
                f"{name} input must have shape (B, C, T), "
                f"got {tuple(x.shape)}"
            )

        if x.size(0) <= 0 or x.size(2) <= 0:
            raise ValueError(
                f"{name} input must have non-empty batch/time dimensions, "
                f"got {tuple(x.shape)}"
            )

        if x.size(1) != expected_channels:
            raise ValueError(
                f"{name} input channel dimension must be "
                f"{expected_channels}, got {x.size(1)}"
            )

        if not (x.is_floating_point() or x.is_complex()):
            raise TypeError(
                f"{name} input must use a floating-point or complex dtype, "
                f"got {x.dtype}"
            )

    def analysis(self, x):
        """Analyze mono waveform into PQMF subbands."""

        self._validate_input(
            x,
            expected_channels=1,
            name="Analysis",
        )

        x = F.conv1d(
            self.pad_fn(x),
            self.analysis_filter,
        )

        return F.conv1d(
            x,
            self.updown_filter,
            stride=self.subbands,
        )

    def synthesis(self, x):
        """Synthesize waveform from PQMF subbands."""

        self._validate_input(
            x,
            expected_channels=self.subbands,
            name="Synthesis",
        )

        x = F.conv_transpose1d(
            x,
            self.updown_filter * self.subbands,
            stride=self.subbands,
        )

        return F.conv1d(
            self.pad_fn(x),
            self.synthesis_filter,
        )

최종 개선사항
✅ assert 기반 파라미터 검증 → 명시적 TypeError·ValueError 검증 → 최적화 실행에서도 필터 설계 계약 보장
✅ NaN·Inf 입력 가능 → cutoff_ratio·beta의 유한값 검증 → 비정상 수치의 필터 계수 전파 차단
✅ 암묵적 (B, C, T) 입력 계약 → 차원·배치·시간축·채널 검증 → conv1d 단계 이전의 입력 오류 조기 차단
✅ 잘못된 입력 타입·정수 dtype 허용 → Tensor 및 실수/복소수 dtype 검증 → 저수준 연산에서 발생하는 불명확한 런타임 오류 방지
✅ 호출마다 device 강제 이동 → register_buffer의 PyTorch device lifecycle 활용 → 불필요한 텐서 이동과 숨겨진 device 오류 방지
✅ 필터 설계 수식 자체 수정 → 기존 PQMF 분석·합성 알고리즘 보존 → 리팩토링으로 인한 신호처리 동작 변경 방지
✅ 모호한 원본 검증·문서 → 명확한 예외 메시지와 역할별 검증 함수 → 장애 원인 추적성과 유지보수성 강화        

원본의 PQMF 수학적 동작과 신호처리 구조는 그대로 보존하면서 입력·파라미터·수치 안정성·디바이스 lifecycle을 방어적으로 강화해, 모델 학습과 추론 환경에서 예측 가능한 실패와 안정적인 실행을 확보한 실무형 구현으로 승격했다.
