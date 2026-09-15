import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _gen_bias_mask(max_length: int) -> torch.Tensor:
    """
    自己注意機構で未来のタイムステップをマスクするためのバイアス行列を生成
    """
    np_mask = np.triu(np.full([max_length, max_length], -np.inf), 1)
    torch_mask = torch.from_numpy(np_mask).float()
    return torch_mask.unsqueeze(0).unsqueeze(1)


def _gen_timing_signal(
    length: int, channels: int, min_timescale: float = 1.0, max_timescale: float = 1.0e4
) -> torch.Tensor:
    """
    正弦波に基づくタイミング信号（Positional Encoding）を生成
    """
    position = np.arange(length)
    num_timescales = channels // 2
    log_timescale_increment = math.log(float(max_timescale) / float(min_timescale)) / (
        float(num_timescales) - 1
    )
    inv_timescales = min_timescale * np.exp(
        np.arange(num_timescales, dtype=np.float64) * -log_timescale_increment
    )
    scaled_time = np.expand_dims(position, 1) * np.expand_dims(inv_timescales, 0)
    signal = np.concatenate([np.sin(scaled_time), np.cos(scaled_time)], axis=1)
    signal = np.pad(
        signal, [[0, 0], [0, channels % 2]], "constant", constant_values=[0.0, 0.0]
    )
    signal = signal.reshape([1, length, channels])
    return torch.from_numpy(signal).float()


class LayerNorm(nn.Module):
    def __init__(self, features: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class OutputLayer(nn.Module):
    def __init__(
        self, hidden_size: int, output_size: int, probs_out: bool = False
    ) -> None:
        super().__init__()
        self.output_size = output_size
        self.output_projection = nn.Linear(hidden_size, output_size)
        self.probs_out = probs_out
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=int(hidden_size / 2),
            batch_first=True,
            bidirectional=True,
        )
        self.hidden_size = hidden_size


class SoftmaxOutputLayer(OutputLayer):
    def forward(self, hidden: torch.Tensor):
        logits = self.output_projection(hidden)
        if self.probs_out:
            return logits
        probs = F.softmax(logits, -1)
        topk, indices = torch.topk(probs, 2)
        predictions = indices[:, :, 0]
        second = indices[:, :, 1]
        return predictions, second


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        input_depth: int,
        total_key_depth: int,
        total_value_depth: int,
        output_depth: int,
        num_heads: int,
        bias_mask: torch.Tensor | None = None,
        dropout: float = 0.0,
        attention_map: bool = False,
    ) -> None:
        super().__init__()
        if total_key_depth % num_heads != 0:
            raise ValueError(
                f"Key depth ({total_key_depth}) must be divisible by num_heads ({num_heads})"
            )
        if total_value_depth % num_heads != 0:
            raise ValueError(
                f"Value depth ({total_value_depth}) must be divisible by num_heads ({num_heads})"
            )

        self.attention_map = attention_map
        self.num_heads = num_heads
        self.query_scale = (total_key_depth // num_heads) ** -0.5
        self.bias_mask = bias_mask
        self.query_linear = nn.Linear(input_depth, total_key_depth, bias=False)
        self.key_linear = nn.Linear(input_depth, total_key_depth, bias=False)
        self.value_linear = nn.Linear(input_depth, total_value_depth, bias=False)
        self.output_linear = nn.Linear(total_value_depth, output_depth, bias=False)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        return x.view(
            shape[0], shape[1], self.num_heads, shape[2] // self.num_heads
        ).permute(0, 2, 1, 3)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        return (
            x.permute(0, 2, 1, 3)
            .contiguous()
            .view(shape[0], shape[2], shape[3] * self.num_heads)
        )

    def forward(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor):
        queries = self.query_linear(queries)
        keys = self.key_linear(keys)
        values = self.value_linear(values)

        queries = self._split_heads(queries)
        keys = self._split_heads(keys)
        values = self._split_heads(values)

        queries *= self.query_scale
        logits = torch.matmul(queries, keys.permute(0, 1, 3, 2))

        if self.bias_mask is not None:
            logits += self.bias_mask[
                :, :, : logits.shape[-2], : logits.shape[-1]
            ].type_as(logits.data)

        weights = F.softmax(logits, dim=-1)
        weights = self.dropout(weights)

        contexts = torch.matmul(weights, values)
        contexts = self._merge_heads(contexts)
        outputs = self.output_linear(contexts)

        if self.attention_map:
            return outputs, weights
        return outputs


class Conv(nn.Module):
    def __init__(
        self, input_size: int, output_size: int, kernel_size: int, pad_type: str
    ) -> None:
        super().__init__()
        padding = (
            (kernel_size - 1, 0)
            if pad_type == "left"
            else (kernel_size // 2, (kernel_size - 1) // 2)
        )
        self.pad = nn.ConstantPad1d(padding, 0)
        self.conv = nn.Conv1d(
            input_size, output_size, kernel_size=kernel_size, padding=0
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = self.pad(inputs.permute(0, 2, 1))
        outputs = self.conv(inputs).permute(0, 2, 1)
        return outputs


class PositionwiseFeedForward(nn.Module):
    def __init__(
        self,
        input_depth: int,
        filter_size: int,
        output_depth: int,
        layer_config: str = "ll",
        padding: str = "left",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layers = []
        sizes = (
            [(input_depth, filter_size)]
            + [(filter_size, filter_size)] * (len(layer_config) - 2)
            + [(filter_size, output_depth)]
        )
        for lc, s in zip(list(layer_config), sizes):
            if lc == "l":
                layers.append(nn.Linear(s[0], s[1]))
            elif lc == "c":
                layers.append(Conv(s[0], s[1], kernel_size=3, pad_type=padding))
            else:
                raise ValueError(f"Unknown layer type: {lc}")

        self.layers = nn.ModuleList(layers)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers):
                x = self.relu(x)
                x = self.dropout(x)
        return x


class self_attention_block(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        total_key_depth: int,
        total_value_depth: int,
        filter_size: int,
        num_heads: int,
        bias_mask: torch.Tensor | None = None,
        layer_dropout: float = 0.0,
        attention_dropout: float = 0.0,
        relu_dropout: float = 0.0,
        attention_map: bool = False,
    ) -> None:
        super().__init__()
        self.attention_map = attention_map
        self.multi_head_attention = MultiHeadAttention(
            hidden_size,
            total_key_depth,
            total_value_depth,
            hidden_size,
            num_heads,
            bias_mask,
            attention_dropout,
            attention_map,
        )
        self.positionwise_convolution = PositionwiseFeedForward(
            hidden_size,
            filter_size,
            hidden_size,
            layer_config="cc",
            padding="both",
            dropout=relu_dropout,
        )
        self.dropout = nn.Dropout(layer_dropout)
        self.layer_norm_mha = LayerNorm(hidden_size)
        self.layer_norm_ffn = LayerNorm(hidden_size)

    def forward(self, inputs: torch.Tensor):
        x = inputs
        x_norm = self.layer_norm_mha(x)
        weights = None
        if self.attention_map:
            y, weights = self.multi_head_attention(x_norm, x_norm, x_norm)
        else:
            y = self.multi_head_attention(x_norm, x_norm, x_norm)
        x = self.dropout(x + y)
        x_norm = self.layer_norm_ffn(x)
        y = self.positionwise_convolution(x_norm)
        y = self.dropout(x + y)
        if self.attention_map and weights is not None:
            return y, weights
        return y


class bi_directional_self_attention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        total_key_depth: int,
        total_value_depth: int,
        filter_size: int,
        num_heads: int,
        max_length: int,
        layer_dropout: float = 0.0,
        attention_dropout: float = 0.0,
        relu_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.attn_block = self_attention_block(
            hidden_size=hidden_size,
            total_key_depth=total_key_depth or hidden_size,
            total_value_depth=total_value_depth or hidden_size,
            filter_size=filter_size,
            num_heads=num_heads,
            bias_mask=_gen_bias_mask(max_length),
            layer_dropout=layer_dropout,
            attention_dropout=attention_dropout,
            relu_dropout=relu_dropout,
            attention_map=True,
        )
        self.backward_attn_block = self_attention_block(
            hidden_size=hidden_size,
            total_key_depth=total_key_depth or hidden_size,
            total_value_depth=total_value_depth or hidden_size,
            filter_size=filter_size,
            num_heads=num_heads,
            bias_mask=torch.transpose(_gen_bias_mask(max_length), dim0=2, dim1=3),
            layer_dropout=layer_dropout,
            attention_dropout=attention_dropout,
            relu_dropout=relu_dropout,
            attention_map=True,
        )
        self.linear = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, inputs):
        x, list_w = inputs
        encoder_outputs, weights = self.attn_block(x)
        reverse_outputs, reverse_weights = self.backward_attn_block(x)
        outputs = torch.cat((encoder_outputs, reverse_outputs), dim=2)
        y = self.linear(outputs)
        list_w.append(weights)
        list_w.append(reverse_weights)
        return y, list_w


class bi_directional_self_attention_layers(nn.Module):
    def __init__(
        self,
        embedding_size: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        total_key_depth: int,
        total_value_depth: int,
        filter_size: int,
        max_length: int = 108,
        input_dropout: float = 0.0,
        layer_dropout: float = 0.0,
        attention_dropout: float = 0.0,
        relu_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.timing_signal = _gen_timing_signal(max_length, hidden_size)
        params = (
            hidden_size,
            total_key_depth or hidden_size,
            total_value_depth or hidden_size,
            filter_size,
            num_heads,
            max_length,
            layer_dropout,
            attention_dropout,
            relu_dropout,
        )
        self.embedding_proj = nn.Linear(embedding_size, hidden_size, bias=False)
        self.self_attn_layers = nn.Sequential(
            *[bi_directional_self_attention(*params) for _ in range(num_layers)]
        )
        self.layer_norm = LayerNorm(hidden_size)
        self.input_dropout = nn.Dropout(input_dropout)

    def forward(self, inputs: torch.Tensor):
        x = self.input_dropout(inputs)
        x = self.embedding_proj(x)
        x += self.timing_signal[:, : inputs.shape[1], :].type_as(inputs.data)
        y, weights_list = self.self_attn_layers((x, []))
        y = self.layer_norm(y)
        return y, weights_list


class BTC_model(nn.Module):
    """
    Bi-directional Transformer for Musical Chord Recognition (BTC)
    (Park et al., ISMIR 2019)
    """

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.timestep = config["timestep"]
        self.probs_out = config.get("probs_out", True)
        params = (
            config["feature_size"],
            config["hidden_size"],
            config["num_layers"],
            config["num_heads"],
            config["total_key_depth"],
            config["total_value_depth"],
            config["filter_size"],
            config["timestep"],
            config["input_dropout"],
            config["layer_dropout"],
            config["attention_dropout"],
            config["relu_dropout"],
        )
        self.self_attn_layers = bi_directional_self_attention_layers(*params)
        self.output_layer = SoftmaxOutputLayer(
            hidden_size=config["hidden_size"],
            output_size=config["num_chords"],
            probs_out=self.probs_out,
        )

    def forward(self, x: torch.Tensor):
        self_attn_output, weights_list = self.self_attn_layers(x)
        if self.probs_out:
            return self.output_layer(self_attn_output)
        prediction, second = self.output_layer(self_attn_output)
        return prediction, second
