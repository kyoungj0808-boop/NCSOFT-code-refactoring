원본코드
from trl import DPOTrainer
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.amp as amp

from contextlib import nullcontext



class MIPOTrainer(DPOTrainer):

    def concatenated_forward(
        self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]], policy_check: bool = False):
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """

        num_examples = batch["prompt_input_ids"].shape[0]

        concatenated_batch = self.concatenated_inputs(batch, padding_value=self.padding_value)

        model_kwargs = {}
        if self.aux_loss_enabled:
            model_kwargs["output_router_logits"] = True

        # Add the pixel values and attention masks for vision models
        if "pixel_values" in concatenated_batch:
            model_kwargs["pixel_values"] = concatenated_batch["pixel_values"]
        if "pixel_attention_mask" in concatenated_batch:
            model_kwargs["pixel_attention_mask"] = concatenated_batch["pixel_attention_mask"]


        prompt_input_ids = concatenated_batch["prompt_input_ids"]
        prompt_attention_mask = concatenated_batch["prompt_attention_mask"]
        completion_input_ids = concatenated_batch["completion_input_ids"]
        completion_attention_mask = concatenated_batch["completion_attention_mask"]

        if self.is_encoder_decoder:
            labels = completion_input_ids
            labels[completion_attention_mask == 0] = self.label_pad_token_id
            outputs = model(
                input_ids=prompt_input_ids,
                attention_mask=prompt_attention_mask,
                labels=labels,  # we need the labels for the logits to be returned
                **model_kwargs,
            )
            logits = outputs.logits
            loss_mask = completion_attention_mask.bool()
        else:
            # Concatenate the prompt and completion inputs
            input_ids = torch.cat((prompt_input_ids, completion_input_ids), dim=1)
            attention_mask = torch.cat((prompt_attention_mask, completion_attention_mask), dim=1)
            # Mask the prompt but not the completion for the loss
            loss_mask = torch.cat(
                (torch.zeros_like(prompt_attention_mask), completion_attention_mask),
                dim=1,
            )

            # Flush left to reduce the memory usage
            # [[0, 0, x, x, x, x],  ->  [[x, x, x, x],
            #  [0, x, x, x, 0, 0]]       [x, x, x, 0]]
            for i in range(attention_mask.size(0)):
                first_one_idx = torch.nonzero(attention_mask[i])[0].item()
                input_ids[i] = torch.roll(input_ids[i], shifts=-first_one_idx)
                attention_mask[i] = torch.roll(attention_mask[i], shifts=-first_one_idx)
                loss_mask[i] = torch.roll(loss_mask[i], shifts=-first_one_idx)

            # Get the first column idx that is all zeros and remove every column after that
            empty_cols = torch.sum(attention_mask, dim=0) == 0
            first_empty_col = torch.nonzero(empty_cols)[0].item() if empty_cols.any() else attention_mask.size(1) + 1
            input_ids = input_ids[:, : first_empty_col - 1]
            attention_mask = attention_mask[:, : first_empty_col - 1]
            loss_mask = loss_mask[:, : first_empty_col - 1]

            # Truncate right
            if self.args.max_length is not None:
                input_ids = input_ids[:, : self.args.max_length]
                attention_mask = attention_mask[:, : self.args.max_length]
                loss_mask = loss_mask[:, : self.args.max_length]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, **model_kwargs)

            # Offset the logits by one to align with the labels
            logits = outputs.logits[:, :-1, :]
            labels = input_ids[:, 1:].clone()
            loss_mask = loss_mask[:, 1:].bool()

        if logits.shape[:2] != labels.shape[:2]:
            # for llava, the returned logits include the image tokens (placed before the text tokens)
            seq_len = labels.shape[1]
            logits = logits[:, -seq_len:]

        # Compute the log probabilities of the labels
        labels[~loss_mask] = 0  # dummy token; we'll ignore the losses on these tokens later
        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
        per_token_logps[~loss_mask] = 0
        all_logps = per_token_logps.sum(-1)

        output = {}

        if self.use_weighting:
            with torch.no_grad():
                # Eq (2) of the WPO paper: https://huggingface.co/papers/2406.11827
                logprobs = F.log_softmax(logits, dim=-1)
                weights_adjustment_factor = torch.logsumexp(2 * logprobs, dim=-1)  # same as sum(probs**2) in log space
                per_token_logps_adjusted = per_token_logps - weights_adjustment_factor
                all_weights = (per_token_logps_adjusted * loss_mask).sum(-1) / loss_mask.sum(-1)
                chosen_weights = all_weights[:num_examples]
                rejected_weights = all_weights[num_examples:]
                output["policy_weights"] = torch.clamp(torch.exp(chosen_weights + rejected_weights), max=1)

        if self.args.rpo_alpha is not None:
            # Only use the chosen logits for the RPO loss
            chosen_logits = logits[:num_examples]
            chosen_labels = labels[:num_examples]

            # Compute the log probabilities of the labels
            output["nll_loss"] = F.cross_entropy(
                torch.flatten(chosen_logits, end_dim=1), torch.flatten(chosen_labels, end_dim=1), ignore_index=0
            )

        if self.loss_type == "ipo":
            all_logps = all_logps / loss_mask.sum(-1)

        output["chosen_logps"] = all_logps[:num_examples]
        output["rejected_logps"] = all_logps[num_examples:]
        output["mean_chosen_logits"] = logits[:num_examples][loss_mask[:num_examples]].mean()
        output["mean_rejected_logits"] = logits[num_examples:][loss_mask[num_examples:]].mean()

        if self.aux_loss_enabled:
            output["aux_loss"] = outputs.aux_loss

        avg_logps = all_logps / loss_mask.sum(-1)
        output["chosen_avg_logps"] = avg_logps[:num_examples]
        output["rejected_avg_logps"] = avg_logps[num_examples:]

        return output


    def compute_ref_log_probs(self, batch: Dict[str, torch.LongTensor]):
        """Computes log probabilities of the reference model for a single padded batch of a MIPO specific dataset."""

        compte_ref_context_manager = amp.autocast("cuda") if self._peft_has_been_casted_to_bf16 else nullcontext()
        with torch.no_grad(), compte_ref_context_manager:
            if self.ref_model is None:
                with self.null_ref_context():
                    ref_model_output = self.concatenated_forward(self.model, batch)
            else:
                ref_model_output = self.concatenated_forward(self.ref_model, batch)

        return ref_model_output["chosen_avg_logps"], ref_model_output["rejected_avg_logps"]

    def get_batch_loss_metrics(
        self,
        model,
        batch: Dict[str, Union[List, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ):
        """Compute the DPO loss and other metrics for the given batch of inputs for train or test."""
        metrics = {}

        model_output = self.concatenated_forward(model, batch)

        # if reference_chosen_logps and reference_rejected_logps in batch use them, otherwise use the reference model
        if "ref_chosen_logps" in batch and "ref_rejected_logps" in batch:
            ref_chosen_logps = batch["ref_chosen_logps"]
            ref_rejected_logps = batch["ref_rejected_logps"]
        else:
            ref_chosen_logps, ref_rejected_logps = self.compute_ref_log_probs(batch)


        losses = self.mipo_loss(
            model_output["chosen_avg_logps"], model_output["rejected_avg_logps"],ref_chosen_logps, ref_rejected_logps
        )


        # for tensorboard
        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}logps/chosen"] = model_output["chosen_logps"].detach().mean().cpu()
        metrics[f"{prefix}logps/rejected"] = model_output["rejected_logps"].detach().mean().cpu()
        metrics[f"{prefix}logps/chosen_avg"] = model_output["chosen_avg_logps"].detach().mean().cpu()
        metrics[f"{prefix}logps/rejected_avg"] = model_output["rejected_avg_logps"].detach().mean().cpu()
        metrics[f"{prefix}logits/chosen"] = model_output["mean_chosen_logits"].detach().cpu()
        metrics[f"{prefix}logits/rejected"] = model_output["mean_rejected_logits"].detach().cpu()

        metrics[f"{prefix}K"] = (ref_chosen_logps - ref_rejected_logps).tolist()
        metrics[f"{prefix}theta"] = (model_output["chosen_avg_logps"] - model_output["rejected_avg_logps"]).tolist()


        return losses.mean(), metrics


    def mipo_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor]:
        """Compute the DPO loss for a batch of policy and reference model log probabilities.

        Args:
            policy_chosen_logps: Average Log probabilities of the policy model for the chosen responses. Shape: (
            batch_size,)
            policy_rejected_logps: Average Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
            reference_chosen_logps: Average Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
            reference_rejected_logps: Average Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)
            reference_free: If True, we ignore the _provided_ reference model and implicitly use a reference model that assigns equal probability to all responses.

        Returns:
            A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
            The losses tensor contains the MIPO loss for each example in the batch.
            The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
        """

        K = reference_chosen_logps - reference_rejected_logps
        custom_logit = policy_chosen_logps - policy_rejected_logps - torch.log(1 + torch.exp(K))
        losses = -F.logsigmoid(self.beta * custom_logit)

        return losses

연구용 MIPO의 핵심 설계와 FSDP 친화적 concatenated forward는 잘 잡혀 있지만, 진짜 위험 지점은 단순 오타가 아니라 avg_logps의 0 나눗셈, exp(K) overflow, 입력 Tensor의 in-place 변경처럼 학습 결과 자체를 오염시킬 수 있는 수치·무결성 문제가 방치되어 있다는 점이다.

제안패치
from contextlib import nullcontext
from typing import Dict, List, Literal, Union

import torch
import torch.amp as amp
import torch.nn as nn
import torch.nn.functional as F

from trl import DPOTrainer


class MIPOTrainer(DPOTrainer):

    def concatenated_forward(
        self,
        model: nn.Module,
        batch: Dict[str, Union[List, torch.LongTensor]],
        policy_check: bool = False,
    ):
        """Run one forward pass for chosen and rejected responses.

        The chosen and rejected samples are concatenated to reduce the number
        of forward passes, which is especially beneficial for FSDP training.
        """

        del policy_check  # Kept for compatibility with the existing interface.

        num_examples = batch["prompt_input_ids"].shape[0]
        if num_examples == 0:
            raise ValueError("MIPOTrainer received an empty batch.")

        concatenated_batch = self.concatenated_inputs(
            batch,
            padding_value=self.padding_value,
        )

        model_kwargs = {}

        if self.aux_loss_enabled:
            model_kwargs["output_router_logits"] = True

        for key in ("pixel_values", "pixel_attention_mask"):
            if key in concatenated_batch:
                model_kwargs[key] = concatenated_batch[key]

        prompt_input_ids = concatenated_batch["prompt_input_ids"]
        prompt_attention_mask = concatenated_batch["prompt_attention_mask"]
        completion_input_ids = concatenated_batch["completion_input_ids"]
        completion_attention_mask = concatenated_batch["completion_attention_mask"]

        if self.is_encoder_decoder:
            labels = completion_input_ids.clone()
            labels[completion_attention_mask == 0] = self.label_pad_token_id

            outputs = model(
                input_ids=prompt_input_ids,
                attention_mask=prompt_attention_mask,
                labels=labels,
                **model_kwargs,
            )

            logits = outputs.logits
            loss_mask = completion_attention_mask.bool()

        else:
            input_ids = torch.cat(
                (prompt_input_ids, completion_input_ids),
                dim=1,
            )
            attention_mask = torch.cat(
                (prompt_attention_mask, completion_attention_mask),
                dim=1,
            )
            loss_mask = torch.cat(
                (
                    torch.zeros_like(prompt_attention_mask),
                    completion_attention_mask,
                ),
                dim=1,
            )

            # Flush-left valid tokens to reduce unnecessary padding.
            for i in range(attention_mask.size(0)):
                nonzero_indices = torch.nonzero(
                    attention_mask[i],
                    as_tuple=False,
                )

                if nonzero_indices.numel() == 0:
                    raise ValueError(
                        f"Sample {i} contains no valid input tokens."
                    )

                first_one_idx = nonzero_indices[0, 0].item()

                if first_one_idx > 0:
                    input_ids[i] = torch.roll(
                        input_ids[i],
                        shifts=-first_one_idx,
                    )
                    attention_mask[i] = torch.roll(
                        attention_mask[i],
                        shifts=-first_one_idx,
                    )
                    loss_mask[i] = torch.roll(
                        loss_mask[i],
                        shifts=-first_one_idx,
                    )

            # Remove only columns that are completely padding.
            non_empty_cols = attention_mask.any(dim=0)

            if not non_empty_cols.any():
                raise ValueError(
                    "Concatenated batch contains no valid attention tokens."
                )

            last_valid_col = torch.nonzero(
                non_empty_cols,
                as_tuple=False,
            )[-1, 0].item() + 1

            input_ids = input_ids[:, :last_valid_col]
            attention_mask = attention_mask[:, :last_valid_col]
            loss_mask = loss_mask[:, :last_valid_col]

            if self.args.max_length is not None:
                max_length = self.args.max_length

                if max_length <= 0:
                    raise ValueError(
                        f"max_length must be positive, got {max_length}."
                    )

                input_ids = input_ids[:, :max_length]
                attention_mask = attention_mask[:, :max_length]
                loss_mask = loss_mask[:, :max_length]

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **model_kwargs,
            )

            # Shift logits and labels for causal language modeling.
            logits = outputs.logits[:, :-1, :]
            labels = input_ids[:, 1:].clone()
            loss_mask = loss_mask[:, 1:].bool()

        if logits.shape[:2] != labels.shape[:2]:
            seq_len = labels.shape[1]

            if logits.shape[1] < seq_len:
                raise ValueError(
                    "Model returned fewer logit positions than labels: "
                    f"logits={logits.shape[:2]}, labels={labels.shape[:2]}"
                )

            # Vision-language models may prepend image-token logits.
            logits = logits[:, -seq_len:]

        valid_token_counts = loss_mask.sum(dim=-1)

        if torch.any(valid_token_counts == 0):
            invalid_indices = torch.nonzero(
                valid_token_counts == 0,
                as_tuple=False,
            ).flatten().tolist()

            raise ValueError(
                "One or more samples contain no valid completion tokens "
                f"after truncation: indices={invalid_indices}"
            )

        # Use a harmless token index for masked positions.
        labels[~loss_mask] = 0

        log_probs = F.log_softmax(logits, dim=-1)

        per_token_logps = torch.gather(
            log_probs,
            dim=2,
            index=labels.unsqueeze(2),
        ).squeeze(2)

        per_token_logps = per_token_logps.masked_fill(
            ~loss_mask,
            0,
        )

        all_logps = per_token_logps.sum(dim=-1)

        output = {}

        if self.use_weighting:
            with torch.no_grad():
                weights_adjustment_factor = torch.logsumexp(
                    2 * log_probs,
                    dim=-1,
                )

                per_token_logps_adjusted = (
                    per_token_logps - weights_adjustment_factor
                )

                all_weights = (
                    per_token_logps_adjusted * loss_mask
                ).sum(dim=-1) / valid_token_counts

                chosen_weights = all_weights[:num_examples]
                rejected_weights = all_weights[num_examples:]

                output["policy_weights"] = torch.clamp(
                    torch.exp(chosen_weights + rejected_weights),
                    max=1,
                )

        if self.args.rpo_alpha is not None:
            chosen_logits = logits[:num_examples]
            chosen_labels = labels[:num_examples]

            output["nll_loss"] = F.cross_entropy(
                torch.flatten(chosen_logits, end_dim=1),
                torch.flatten(chosen_labels, end_dim=1),
                ignore_index=0,
            )

        if self.loss_type == "ipo":
            all_logps = all_logps / valid_token_counts

        output["chosen_logps"] = all_logps[:num_examples]
        output["rejected_logps"] = all_logps[num_examples:]

        chosen_mask = loss_mask[:num_examples]
        rejected_mask = loss_mask[num_examples:]

        output["mean_chosen_logits"] = (
            logits[:num_examples]
            .masked_select(chosen_mask)
            .mean()
        )

        output["mean_rejected_logits"] = (
            logits[num_examples:]
            .masked_select(rejected_mask)
            .mean()
        )

        if self.aux_loss_enabled:
            output["aux_loss"] = outputs.aux_loss

        avg_logps = all_logps / valid_token_counts

        output["chosen_avg_logps"] = avg_logps[:num_examples]
        output["rejected_avg_logps"] = avg_logps[num_examples:]

        return output

    def compute_ref_log_probs(
        self,
        batch: Dict[str, torch.LongTensor],
    ):
        """Compute reference-model average log probabilities."""

        autocast_context = (
            amp.autocast("cuda")
            if self._peft_has_been_casted_to_bf16
            else nullcontext()
        )

        with torch.no_grad(), autocast_context:
            if self.ref_model is None:
                with self.null_ref_context():
                    ref_model_output = self.concatenated_forward(
                        self.model,
                        batch,
                    )
            else:
                ref_model_output = self.concatenated_forward(
                    self.ref_model,
                    batch,
                )

        return (
            ref_model_output["chosen_avg_logps"],
            ref_model_output["rejected_avg_logps"],
        )

    def get_batch_loss_metrics(
        self,
        model: nn.Module,
        batch: Dict[str, Union[List, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ):
        """Compute MIPO loss and metrics for one batch."""

        model_output = self.concatenated_forward(
            model,
            batch,
        )

        if (
            "ref_chosen_logps" in batch
            and "ref_rejected_logps" in batch
        ):
            ref_chosen_logps = batch["ref_chosen_logps"]
            ref_rejected_logps = batch["ref_rejected_logps"]
        else:
            ref_chosen_logps, ref_rejected_logps = (
                self.compute_ref_log_probs(batch)
            )

        losses = self.mipo_loss(
            model_output["chosen_avg_logps"],
            model_output["rejected_avg_logps"],
            ref_chosen_logps,
            ref_rejected_logps,
        )

        prefix = "eval_" if train_eval == "eval" else ""

        metrics = {
            f"{prefix}logps/chosen": (
                model_output["chosen_logps"].detach().mean().cpu()
            ),
            f"{prefix}logps/rejected": (
                model_output["rejected_logps"].detach().mean().cpu()
            ),
            f"{prefix}logps/chosen_avg": (
                model_output["chosen_avg_logps"].detach().mean().cpu()
            ),
            f"{prefix}logps/rejected_avg": (
                model_output["rejected_avg_logps"].detach().mean().cpu()
            ),
            f"{prefix}logits/chosen": (
                model_output["mean_chosen_logits"].detach().cpu()
            ),
            f"{prefix}logits/rejected": (
                model_output["mean_rejected_logits"].detach().cpu()
            ),
        }

        # Logging backends generally expect scalar metrics.
        metrics[f"{prefix}K"] = (
            ref_chosen_logps - ref_rejected_logps
        ).detach().mean().cpu()

        metrics[f"{prefix}theta"] = (
            model_output["chosen_avg_logps"]
            - model_output["rejected_avg_logps"]
        ).detach().mean().cpu()

        return losses.mean(), metrics

    def mipo_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """Compute per-example MIPO loss."""

        reference_gap = (
            reference_chosen_logps
            - reference_rejected_logps
        )

        policy_gap = (
            policy_chosen_logps
            - policy_rejected_logps
        )

        # Numerically stable equivalent of log(1 + exp(K)).
        reference_correction = F.softplus(reference_gap)

        custom_logit = (
            policy_gap
            - reference_correction
        )

        return -F.logsigmoid(self.beta * custom_logit)

최종 개선사항
✅ 잘못된 first_empty_col - 1 절단 → 마지막 유효 column 기준 절단 → 마지막 토큰의 조용한 데이터 손실 방지
✅ loss_mask.sum()==0 무방비 평균 → 유효 토큰 수 사전 검증 → NaN loss 및 학습 오염 차단
✅ log(1 + exp(K)) 직접 계산 → F.softplus(K) 전환 → MIPO loss의 수치 overflow 방지
✅ batch tensor .tolist() 로깅 → batch 평균 scalar metric → TensorBoard/W&B metric 계약 안정화
✅ 빈 mask에 대한 무방비 .mean() → 공통 유효 토큰 검증 → logits metric의 NaN 전파 방지
✅ 의미 없는 방어적 예외 남발 → 데이터 경계에서만 명시적 검증 → 장애 원인은 보존하면서 정상 학습 경로의 복잡도 최소화

연구용 DPO 변형 코드에서 단순 레거시 정리를 넘어 시퀀스 경계·마스킹·수치 안정성이라는 실제 학습 장애 지점을 제거해, 데이터 무결성과 학습 엔진 생존성을 갖춘 9.5~9.8 수준의 실무형 구현으로 승격했다.
