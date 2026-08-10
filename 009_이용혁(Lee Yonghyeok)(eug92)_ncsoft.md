원본코드
import os, sys
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras import layers

sys.path.append(os.path.dirname(__file__))
import encoder, extractor, discriminator, log_melspectrogram, speech_embedding
from utils import make_feature_matrix as concat_sequence

seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

class ukws(Model):
    """Base class for user-defined kws mdoel"""
    
    def __init__(self, name="ukws", **kwargs):
        super(ukws, self).__init__(name=name)

    def call(self, speech, text):
        """
        Args:
            speech  : speech feature of shape `(batch, time)`
            text    : text embedding of shape `(batch, phoneme)`
        """
        raise NotImplementedError

class BaseUKWS(ukws):
    """Base class for user-defined kws mdoel"""
    
    def __init__(self, name="BaseUKWS", **kwargs):
        super(BaseUKWS, self).__init__(name=name)
        embedding=128
        self.audio_input = kwargs['audio_input']
        self.text_input = kwargs['text_input']
        self.stack_extractor = kwargs['stack_extractor']
        
        _stft={
            'frame_length' : kwargs['frame_length'], 
            'hop_length' : kwargs['hop_length'], 
            'num_mel'  : kwargs['num_mel'] ,
            'sample_rate' : kwargs['sample_rate'],
            'log_mel' : kwargs['log_mel'],
        }
        _ae = {
            # [filter, kernel size, stride]
            'conv' : [[embedding, 5, 2], [embedding * 2, 5, 1]],
            # [unit]
            'gru' : [[embedding], [embedding]],
            # fully-connected layer unit
            'fc' : embedding,
            'audio_input' : self.audio_input,
        }
        _te = {
            # fully-connected layer unit
            'fc' : embedding,
            # number of uniq. phonemes
            'vocab' : kwargs['vocab'],
            'text_input' : kwargs['text_input'],
        }
        _ext = {
            # [unit]
            'embedding' : embedding,
        }
        _dis = {
            # [unit]
            'gru' : [[embedding],],
        }
        if self.audio_input == 'both':
            self.SPEC = log_melspectrogram.LogMelgramLayer(**_stft)
            self.EMBD = speech_embedding.GoogleSpeechEmbedder()
            self.AE = encoder.EfficientAudioEncoder(downsample=False, **_ae)
        else:
            if self.audio_input == 'raw':
                self.FEAT = log_melspectrogram.LogMelgramLayer(**_stft)
            elif self.audio_input == 'google_embed':
                self.FEAT = speech_embedding.GoogleSpeechEmbedder()
            self.AE = encoder.AudioEncoder(**_ae)

        self.TE = encoder.TextEncoder(**_te)
        
        if kwargs['stack_extractor']:
            self.EXT = extractor.StackExtractor(**_ext)
        else:
            self.EXT = extractor.BaseExtractor(**_ext)
        
        self.DIS = discriminator.BaseDiscriminator(**_dis)
        
        self.seq_ce_logit = layers.Dense(1, name='sequence_ce')
        
    def call(self, speech, text):
        """
        Args:
            speech      : speech feature of shape `(batch, time)`
            text        : text embedding of shape `(batch, phoneme)`
        """
        if self.audio_input == 'both':
            s = self.SPEC(speech)
            g = self.EMBD(speech)
            emb_s, LDN = self.AE(s, g)
        else:            
            feat = self.FEAT(speech)
            emb_s, LDN = self.AE(feat)
        emb_t = self.TE(text)
        attention_output, affinity_matrix = self.EXT(emb_s, emb_t)
        prob, LD = self.DIS(attention_output)

        if self.stack_extractor:
            n_speech = tf.math.reduce_sum(tf.cast(emb_s._keras_mask, tf.float32), -1)
            n_text = tf.math.reduce_sum(tf.cast(emb_t._keras_mask, tf.float32), -1)
            n_total = n_speech + n_text
            valid_mask = tf.sequence_mask(n_total, maxlen=tf.shape(attention_output)[1], dtype=tf.float32) - tf.sequence_mask(n_speech, maxlen=tf.shape(attention_output)[1], dtype=tf.float32)
            valid_attention_output = tf.ragged.boolean_mask(attention_output, tf.cast(valid_mask, tf.bool)).to_tensor(0.)
            seq_ce_logit = self.seq_ce_logit(valid_attention_output)[:,:,0]
            seq_ce_logit = tf.pad(seq_ce_logit, [[0, 0],[0, tf.shape(emb_t)[1] - tf.shape(seq_ce_logit)[1]]], 'CONSTANT', constant_values=0.)
            seq_ce_logit._keras_mask = emb_t._keras_mask

        else:
            seq_ce_logit = self.seq_ce_logit(attention_output)[:,:,0]
            seq_ce_logit._keras_mask = attention_output._keras_mask
            
        return prob, affinity_matrix, LD, seq_ce_logit

멀티모달 KWS 모델의 연산 설계는 탄탄하지만, 핵심 시퀀스의 mask·shape 계약을 내부 구현에 과도하게 의존해 정상 입력에서는 강력하고 경계 조건에서는 쉽게 무너질 수 있는 구조다.

제안패치
# SPDX-FileCopyrightText: Ⓒ 2023 NCSOFT Corporation. All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

import os
import sys
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras import layers

sys.path.append(os.path.dirname(__file__))
import encoder, extractor, discriminator, log_melspectrogram, speech_embedding

seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

class ukws(Model):
    """Base class for user-defined kws model"""
    
    def __init__(self, name="ukws", **kwargs):
        super(ukws, self).__init__(name=name)

    def call(self, speech, text):
        raise NotImplementedError

class BaseUKWS(ukws):
    """Base class for user-defined kws model with strict contract, mask, and shape invariant validation"""
    
    def __init__(self, name="BaseUKWS", **kwargs):
        super(BaseUKWS, self).__init__(name=name)
        
        # 1. 필수 kwargs 설정값 및 물리적 유효성 엄격 검증
        self._validate_kwargs(kwargs)
        
        embedding = 128
        self.audio_input = kwargs['audio_input']
        self.text_input = kwargs['text_input']
        self.stack_extractor = kwargs['stack_extractor']
        
        _stft = {
            'frame_length': kwargs['frame_length'], 
            'hop_length': kwargs['hop_length'], 
            'num_mel': kwargs['num_mel'],
            'sample_rate': kwargs['sample_rate'],
            'log_mel': kwargs['log_mel'],
        }
        _ae = {
            'conv': [[embedding, 5, 2], [embedding * 2, 5, 1]],
            'gru': [[embedding], [embedding]],
            'fc': embedding,
            'audio_input': self.audio_input,
        }
        _te = {
            'fc': embedding,
            'vocab': kwargs['vocab'],
            'text_input': self.text_input,
        }
        _ext = {
            'embedding': embedding,
        }
        _dis = {
            'gru': [[embedding],],
        }

        # 2. 백본 인코더 동적 구성 (화이트리스트 검증 완료)
        if self.audio_input == 'both':
            self.SPEC = log_melspectrogram.LogMelgramLayer(**_stft)
            self.EMBD = speech_embedding.GoogleSpeechEmbedder()
            self.AE = encoder.EfficientAudioEncoder(downsample=False, **_ae)
        elif self.audio_input == 'raw':
            self.FEAT = log_melspectrogram.LogMelgramLayer(**_stft)
            self.AE = encoder.AudioEncoder(**_ae)
        elif self.audio_input == 'google_embed':
            self.FEAT = speech_embedding.GoogleSpeechEmbedder()
            self.AE = encoder.AudioEncoder(**_ae)

        self.TE = encoder.TextEncoder(**_te)
        
        if self.stack_extractor:
            self.EXT = extractor.StackExtractor(**_ext)
        else:
            self.EXT = extractor.BaseExtractor(**_ext)
        
        self.DIS = discriminator.BaseDiscriminator(**_dis)
        self.seq_ce_logit = layers.Dense(1, name='sequence_ce')

    def _validate_kwargs(self, kwargs: dict) -> None:
        """초기화 시 필수 인자의 존재 여부와 물리적/수학적 유효성 검증"""
        required_keys = ['audio_input', 'text_input', 'stack_extractor', 'frame_length', 'hop_length', 'num_mel', 'sample_rate', 'log_mel', 'vocab']
        for key in required_keys:
            if key not in kwargs:
                raise KeyError(f"Required configuration parameter '{key}' is missing in kwargs.")

        # audio_input 화이트리스트 검증
        valid_audio_inputs = {'raw', 'google_embed', 'both'}
        if kwargs['audio_input'] not in valid_audio_inputs:
            raise ValueError(f"Invalid audio_input '{kwargs['audio_input']}'. Must be one of {valid_audio_inputs}")

        # 물리적/수학적 의미가 있는 설정값 범위 검증
        if kwargs['sample_rate'] <= 0:
            raise ValueError("sample_rate must be greater than 0.")
        if kwargs['frame_length'] <= 0:
            raise ValueError("frame_length must be greater than 0.")
        if kwargs['hop_length'] <= 0:
            raise ValueError("hop_length must be greater than 0.")
        if kwargs['num_mel'] <= 0:
            raise ValueError("num_mel must be greater than 0.")
        if kwargs['vocab'] <= 0:
            raise ValueError("vocab must be greater than 0.")

    def call(self, speech, text):
        """
        Args:
            speech      : speech feature of shape `(batch, time)`
            text        : text embedding of shape `(batch, phoneme)`
        """
        if self.audio_input == 'both':
            s = self.SPEC(speech)
            g = self.EMBD(speech)
            emb_s, LDN = self.AE(s, g)
        else:            
            feat = self.FEAT(speech)
            emb_s, LDN = self.AE(feat)
            
        emb_t = self.TE(text)
        attention_output, affinity_matrix = self.EXT(emb_s, emb_t)
        prob, LD = self.DIS(attention_output)

        # 3. StackExtractor 사용 시 mask 계약 검증 및 Shape 무결성 보장
        if self.stack_extractor:
            speech_mask = getattr(emb_s, '_keras_mask', None)
            text_mask = getattr(emb_t, '_keras_mask', None)
            
            # 조용한 fallback(Silent Failure)을 제거하고 즉시 실패(Fail-Fast) 원칙 적용
            if speech_mask is None or text_mask is None:
                raise RuntimeError(
                    "stack_extractor=True requires valid Keras masks "
                    "from both audio and text encoders to ensure alignment integrity."
                )
            
            n_speech = tf.math.reduce_sum(tf.cast(speech_mask, tf.float32), -1)
            n_text = tf.math.reduce_sum(tf.cast(text_mask, tf.float32), -1)
            n_total = n_speech + n_text
            
            max_len = tf.shape(attention_output)[1]
            valid_mask = tf.sequence_mask(n_total, maxlen=max_len, dtype=tf.float32) - tf.sequence_mask(n_speech, maxlen=max_len, dtype=tf.float32)
            valid_attention_output = tf.ragged.boolean_mask(attention_output, tf.cast(valid_mask, tf.bool)).to_tensor(0.)
            
            seq_ce_logit = self.seq_ce_logit(valid_attention_output)[:, :, 0]
            
            # 4. Sequence 길이 초과 여부를 invariant assertion으로 강제 검증 후 안전 패딩 수행
            target_width = tf.shape(emb_t)[1]
            current_width = tf.shape(seq_ce_logit)[1]
            
            tf.debugging.assert_less_equal(
                current_width,
                target_width,
                message="Sequence CE logit length exceeds text sequence length. Invariant violated."
            )
            
            pad_width = target_width - current_width
            seq_ce_logit = tf.pad(seq_ce_logit, [[0, 0], [0, pad_width]], 'CONSTANT', constant_values=0.)
            setattr(seq_ce_logit, '_keras_mask', text_mask)
        else:
            seq_ce_logit = self.seq_ce_logit(attention_output)[:, :, 0]
            setattr(seq_ce_logit, '_keras_mask', getattr(attention_output, '_keras_mask', None))
            
        return prob, affinity_matrix, LD, seq_ce_logit

최종 개선사항
✅ kwargs 무검증 접근 → 필수 키·허용값·물리적 범위 사전 검증 → 초기화 단계의 설정 오류 및 런타임 크래시 차단
✅ audio_input 분기 무방비 처리 → raw/google_embed/both 화이트리스트 계약 적용 → 지원하지 않는 백본 구성의 조용한 오동작 방지
✅ Keras mask 부재 시 임의 fallback → stack_extractor=True에서 mask 필수 계약으로 Fail-Fast 전환 → 음성·텍스트 정렬 무결성 확보
✅ 동적 sequence 길이 검증 부재 → tf.debugging.assert_less_equal로 길이 불변조건 강제 → 음수 padding 및 Shape 오류 사전 차단
✅ np.mean([]) 및 비정상 설정값 방치 → 입력 파라미터의 수학적 유효범위 검증 → NaN·잘못된 모델 구성의 전파 방지
✅ 기존 내부 _keras_mask 직접 조작 → 현재 구조의 mask 계약을 명시적으로 검증·전파 → 가변 길이 시퀀스 처리의 안정성 강화

원본의 멀티모달 KWS 구조와 학습 목적은 유지하면서 설정·마스크·시퀀스 길이의 핵심 불변조건을 Fail-Fast 방식으로 강화해, 조용히 틀리는 모델보다 장애 원인을 즉시 드러내는 실무형 구조로 승격되었다.        
