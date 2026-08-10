원본코드
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d
from torch.nn.utils import weight_norm
from torch.nn.utils import spectral_norm
from typing import List

from avocodo.pqmf import PQMF
from avocodo.models.utils import get_padding


class MDC(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        strides,
        kernel_size,
        dilations,
        use_spectral_norm=False
    ):
        super(MDC, self).__init__()
        norm_f = weight_norm if not use_spectral_norm else spectral_norm
        self.d_convs = nn.ModuleList()
        for _k, _d in zip(kernel_size, dilations):
            self.d_convs.append(
                norm_f(Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=_k,
                    dilation=_d,
                    padding=get_padding(_k, _d)
                ))
            )
        self.post_conv = norm_f(Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=strides,
            padding=get_padding(_k, _d)
        ))
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x):
        _out = None
        for _l in self.d_convs:
            _x = torch.unsqueeze(_l(x), -1)
            _x = F.leaky_relu(_x, 0.2)
            if _out is None:
                _out = _x
            else:
                _out = torch.cat([_out, _x], axis=-1)
        x = torch.sum(_out, dim=-1)
        x = self.post_conv(x)
        x = F.leaky_relu(x, 0.2)  # @@

        return x


class SBDBlock(torch.nn.Module):
    def __init__(
        self,
        segment_dim,
        strides,
        filters,
        kernel_size,
        dilations,
        use_spectral_norm=False
    ):
        super(SBDBlock, self).__init__()
        norm_f = weight_norm if not use_spectral_norm else spectral_norm
        self.convs = nn.ModuleList()
        filters_in_out = [(segment_dim, filters[0])]
        for i in range(len(filters) - 1):
            filters_in_out.append([filters[i], filters[i + 1]])

        for _s, _f, _k, _d in zip(
            strides,
            filters_in_out,
            kernel_size,
            dilations
        ):
            self.convs.append(MDC(
                in_channels=_f[0],
                out_channels=_f[1],
                strides=_s,
                kernel_size=_k,
                dilations=_d,
                use_spectral_norm=use_spectral_norm
            ))
        self.post_conv = norm_f(Conv1d(
            in_channels=_f[1],
            out_channels=1,
            kernel_size=3,
            stride=1,
            padding=3 // 2
        ))  # @@

    def forward(self, x):
        fmap = []
        for _l in self.convs:
            x = _l(x)
            fmap.append(x)
        x = self.post_conv(x)  # @@

        return x, fmap


class MDCDConfig:
    def __init__(self, h):
        self.pqmf_params = h.pqmf_config["sbd"]
        self.f_pqmf_params = h.pqmf_config["fsbd"]
        self.filters = h.sbd_filters
        self.kernel_sizes = h.sbd_kernel_sizes
        self.dilations = h.sbd_dilations
        self.strides = h.sbd_strides
        self.band_ranges = h.sbd_band_ranges
        self.transpose = h.sbd_transpose
        self.segment_size = h.segment_size


class SBD(torch.nn.Module):
    def __init__(self, h, use_spectral_norm=False):
        super(SBD, self).__init__()
        self.config = MDCDConfig(h)
        self.pqmf = PQMF(
            *self.config.pqmf_params
        )
        if True in h.sbd_transpose:
            self.f_pqmf = PQMF(
                *self.config.f_pqmf_params
            )
        else:
            self.f_pqmf = None

        self.discriminators = torch.nn.ModuleList()

        for _f, _k, _d, _s, _br, _tr in zip(
            self.config.filters,
            self.config.kernel_sizes,
            self.config.dilations,
            self.config.strides,
            self.config.band_ranges,
            self.config.transpose
        ):
            if _tr:
                segment_dim = self.config.segment_size // _br[1] - _br[0]
            else:
                segment_dim = _br[1] - _br[0]

            self.discriminators.append(SBDBlock(
                segment_dim=segment_dim,
                filters=_f,
                kernel_size=_k,
                dilations=_d,
                strides=_s,
                use_spectral_norm=use_spectral_norm
            ))

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        y_in = self.pqmf.analysis(y)
        y_hat_in = self.pqmf.analysis(y_hat)
        if self.f_pqmf is not None:
            y_in_f = self.f_pqmf.analysis(y)
            y_hat_in_f = self.f_pqmf.analysis(y_hat)

        for d, br, tr in zip(
            self.discriminators,
            self.config.band_ranges,
            self.config.transpose
        ):
            if tr:
                _y_in = y_in_f[:, br[0]:br[1], :]
                _y_hat_in = y_hat_in_f[:, br[0]:br[1], :]
                _y_in = torch.transpose(_y_in, 1, 2)
                _y_hat_in = torch.transpose(_y_hat_in, 1, 2)
            else:
                _y_in = y_in[:, br[0]:br[1], :]
                _y_hat_in = y_hat_in[:, br[0]:br[1], :]
            y_d_r, fmap_r = d(_y_in)
            y_d_g, fmap_g = d(_y_hat_in)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs

연구용 프로토타입으로서는 의도한 SBD/MDC 구조를 구현했지만, post_conv의 잘못된 padding 참조와 설정 길이 불일치를 삼켜버리는 zip() 등 핵심 구성요소의 무결성 검증이 없어, 모델 구조가 조용히 왜곡되거나 잘못된 입력에서 뒤늦게 런타임 장애로 폭발할 수 있는 코드다.

제안패치
# SPDX-FileCopyrightText: Ⓒ 2023 NCSOFT Corporation. All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

from typing import List, Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d
from torch.nn.utils import weight_norm, spectral_norm

from avocodo.pqmf import PQMF
from avocodo.models.utils import get_padding


class MDC(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        strides: int,
        kernel_size: List[int],
        dilations: List[int],
        use_spectral_norm: bool = False
    ):
        super().__init__()
        
        # 1. 빈 리스트 및 길이 불일치 엄격 방어
        if not kernel_size or not dilations:
            raise ValueError("kernel_size and dilations lists cannot be empty.")
        if len(kernel_size) != len(dilations):
            raise ValueError("kernel_size and dilations must have the same length.")

        norm_f = weight_norm if not use_spectral_norm else spectral_norm
        self.d_convs = nn.ModuleList()
        
        for _k, _d in zip(kernel_size, dilations):
            self.d_convs.append(
                norm_f(Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=_k,
                    dilation=_d,
                    padding=get_padding(_k, _d)
                ))
            )
        
        post_kernel_size = 3
        self.post_conv = norm_f(Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=post_kernel_size,
            stride=strides,
            padding=get_padding(post_kernel_size, 1)
        ))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected 3D tensor input (B, C, L), got {x.ndim}D tensor.")

        outputs = []
        for conv in self.d_convs:
            _x = conv(x)
            _x = F.leaky_relu(_x, 0.2)
            outputs.append(_x)

        stacked = torch.stack(outputs, dim=-1)
        x = torch.sum(stacked, dim=-1)
        
        x = self.post_conv(x)
        x = F.leaky_relu(x, 0.2)

        return x


class SBDBlock(torch.nn.Module):
    def __init__(
        self,
        segment_dim: int,
        strides: List[int],
        filters: List[int],
        kernel_size: List[List[int]],
        dilations: List[List[int]],
        use_spectral_norm: bool = False
    ):
        super().__init__()
        
        if not filters:
            raise ValueError("filters list cannot be empty.")
            
        # 2. SBDBlock 내부 설정 배열 전체 길이 동기화 계약 엄격 검증
        if not (len(strides) == len(filters) == len(kernel_size) == len(dilations)):
            raise ValueError(
                f"Mismatched configuration lengths in SBDBlock! "
                f"filters({len(filters)}), strides({len(strides)}), "
                f"kernel_size({len(kernel_size)}), dilations({len(dilations)}) must match."
            )

        norm_f = weight_norm if not use_spectral_norm else spectral_norm
        self.convs = nn.ModuleList()
        
        filters_in_out = [(segment_dim, filters[0])]
        for i in range(len(filters) - 1):
            filters_in_out.append((filters[i], filters[i + 1]))

        for _s, _f, _k, _d in zip(strides, filters_in_out, kernel_size, dilations):
            self.convs.append(MDC(
                in_channels=_f[0],
                out_channels=_f[1],
                strides=_s,
                kernel_size=_k,
                dilations=_d,
                use_spectral_norm=use_spectral_norm
            ))
            
        final_out_channels = filters[-1]
        post_kernel_size = 3
        self.post_conv = norm_f(Conv1d(
            in_channels=final_out_channels,
            out_channels=1,
            kernel_size=post_kernel_size,
            stride=1,
            padding=get_padding(post_kernel_size, 1)
        ))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, List[torch.Tensor]]:
        fmap = []
        for conv_layer in self.convs:
            x = conv_layer(x)
            fmap.append(x)
        x = self.post_conv(x)

        return x, fmap


class MDCDConfig:
    def __init__(self, h: Any):
        required_keys = ["sbd", "fsbd"]
        if not hasattr(h, "pqmf_config") or not all(k in h.pqmf_config for k in required_keys):
            raise KeyError("Config 'h.pqmf_config' must contain 'sbd' and 'fsbd' configurations.")

        self.pqmf_params = h.pqmf_config["sbd"]
        self.f_pqmf_params = h.pqmf_config["fsbd"]
        self.filters = getattr(h, "sbd_filters", None)
        self.kernel_sizes = getattr(h, "sbd_kernel_sizes", None)
        self.dilations = getattr(h, "sbd_dilations", None)
        self.strides = getattr(h, "sbd_strides", None)
        self.band_ranges = getattr(h, "sbd_band_ranges", None)
        self.transpose = getattr(h, "sbd_transpose", None)
        self.segment_size = getattr(h, "segment_size", None)

        config_attrs = {
            "sbd_filters": self.filters,
            "sbd_kernel_sizes": self.kernel_sizes,
            "sbd_dilations": self.dilations,
            "sbd_strides": self.strides,
            "sbd_band_ranges": self.band_ranges,
            "sbd_transpose": self.transpose,
            "segment_size": self.segment_size,
        }

        # 3. 설정 누락 및 타입 검증 (리스트 형태 필수 계약 확인)
        for name, val in config_attrs.items():
            if val is None:
                raise ValueError(f"Required configuration attribute '{name}' is missing.")
            if name != "segment_size" and not isinstance(val, list):
                raise TypeError(f"Configuration attribute '{name}' must be a list, got {type(val)}.")


class SBD(torch.nn.Module):
    def __init__(self, h: Any, use_spectral_norm: bool = False):
        super().__init__()
        self.config = MDCDConfig(h)
        
        self.pqmf = PQMF(*self.config.pqmf_params)
        self.f_pqmf = PQMF(*self.config.f_pqmf_params) if True in self.config.transpose else None

        # 4. SBD 최상위 레벨 구성 배열 전체 길이 동기화 전역 검증 (zip 안전장치)
        num_discriminators = len(self.config.filters)
        if not (
            len(self.config.kernel_sizes) == num_discriminators
            and len(self.config.dilations) == num_discriminators
            and len(self.config.strides) == num_discriminators
            and len(self.config.band_ranges) == num_discriminators
            and len(self.config.transpose) == num_discriminators
        ):
            raise ValueError(
                "All SBD configuration lists (filters, kernel_sizes, dilations, strides, band_ranges, transpose) "
                "must have the exact same length to prevent silent truncation during zip."
            )

        self.discriminators = torch.nn.ModuleList()

        for _f, _k, _d, _s, _br, _tr in zip(
            self.config.filters,
            self.config.kernel_sizes,
            self.config.dilations,
            self.config.strides,
            self.config.band_ranges,
            self.config.transpose
        ):
            # 5. band_ranges 무결성 경계 검증
            if not isinstance(_br, (list, tuple)) or len(_br) != 2:
                raise ValueError(f"band_range entry '{_br}' must be a pair [start, end].")
            if _br[0] < 0 or _br[1] <= _br[0]:
                raise ValueError(f"Invalid band_range boundaries: {_br}. Must satisfy 0 <= start < end.")

            # 6. segment_dim 원본 수식 계약 유지 및 괄호 명시화
            if _tr:
                segment_dim = (self.config.segment_size // _br[1]) - _br[0]
            else:
                segment_dim = _br[1] - _br[0]

            if segment_dim <= 0:
                raise ValueError(f"Computed segment_dim {segment_dim} for band_range {_br} must be greater than 0.")

            self.discriminators.append(SBDBlock(
                segment_dim=segment_dim,
                filters=_f,
                kernel_size=_k,
                dilations=_d,
                strides=_s,
                use_spectral_norm=use_spectral_norm
            ))

    def forward(self, y: torch.Tensor, y_hat: torch.Tensor) -> tuple[List[torch.Tensor], List[torch.Tensor], List[List[torch.Tensor]], List[List[torch.Tensor]]]:
        if y.ndim != 3 or y_hat.ndim != 3:
            raise ValueError(f"Input waveforms must be 3D tensors (B, C, L). Got y: {y.ndim}D, y_hat: {y.ndim}D.")
        
        # 7. Real과 Fake 파형 간 shape 일치 계약 검증
        if y.shape != y_hat.shape:
            raise ValueError(f"Shape mismatch between real waveform y {y.shape} and fake waveform y_hat {y_hat.shape}.")

        y_d_rs, y_d_gs = [], []
        fmap_rs, fmap_gs = [], []
        
        y_in = self.pqmf.analysis(y)
        y_hat_in = self.pqmf.analysis(y_hat)
        
        y_in_f, y_hat_in_f = None, None
        if self.f_pqmf is not None:
            y_in_f = self.f_pqmf.analysis(y)
            y_hat_in_f = self.f_pqmf.analysis(y_hat)

        # PQMF 분석 결과 채널 수 확인용
        max_channels = y_in.shape[1]
        max_channels_f = y_in_f.shape[1] if y_in_f is not None else 0

        for d, br, tr in zip(
            self.discriminators,
            self.config.band_ranges,
            self.config.transpose
        ):
            # 8. 런타임 band_range와 실제 PQMF 출력 채널 간 경계 초과 방어
            target_max = max_channels_f if tr else max_channels
            if br[1] > target_max:
                raise IndexError(
                    f"band_range end index {br[1]} exceeds available PQMF channels ({target_max}) "
                    f"for transpose={tr}."
                )

            if tr:
                if y_in_f is None:
                    raise RuntimeError("Transpose PQMF analysis requested but f_pqmf is not initialized.")
                _y_in = y_in_f[:, br[0]:br[1], :]
                _y_hat_in = y_hat_in_f[:, br[0]:br[1], :]
                _y_in = torch.transpose(_y_in, 1, 2)
                _y_hat_in = torch.transpose(_y_hat_in, 1, 2)
            else:
                _y_in = y_in[:, br[0]:br[1], :]
                _y_hat_in = y_hat_in[:, br[0]:br[1], :]
                
            y_d_r, fmap_r = d(_y_in)
            y_d_g, fmap_g = d(_y_hat_in)
            
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs

최종 개선사항
✅ kernel_size/dilations 빈 배열 및 길이 불일치 방치 → MDC 생성 시 설정 계약 검증 → 잘못된 convolution 구성의 초기화 단계 차단
✅ zip()의 조용한 설정 배열 절단 가능 → SBD·SBDBlock 전체 설정 배열 길이 동기화 검증 → discriminator 누락 및 부분 구성 방지
✅ band_range 경계 검증 부재 → [start, end] 형식·범위·segment_dim 유효성 검증 → PQMF 채널 범위 초과와 음수/0 차원 구성 방지
✅ Real/Fake 입력 shape 불일치 방치 → y와 y_hat의 3차원 및 완전 동일 shape 계약 검증 → 학습 중 늦게 발생하는 discriminator 연산 오류 조기 차단
✅ 설정 객체의 필수 속성·타입 의존 → 필수 구성값과 리스트 타입을 초기화 단계에서 검증 → 잘못된 설정으로 인한 런타임 장애를 구성 단계에서 차단
✅ f_pqmf 및 PQMF 출력 채널에 대한 런타임 검증 부재 → transpose 경로와 실제 PQMF 채널 수를 대조 → 잘못된 band slicing으로 인한 데이터 왜곡·인덱싱 오류 방지
✅ MDC의 루프 변수 의존 패딩과 미사용 Softmax → post-conv 패딩을 명시적으로 고정하고 dead code 제거 → 레이어 구성의 결정성 및 코드 무결성 확보

원본의 핵심 discriminator 구조는 유지하면서 설정 무결성·shape 계약·PQMF 경계·초기화 실패 감지까지 방어층을 추가해 연구용 구현을 운영 가능한 수준으로 끌어올린 형태다.        
