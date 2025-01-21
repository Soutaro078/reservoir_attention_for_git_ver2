import torch.nn as nn
import torch


# data shape: (batch, sequence_len, d_model)

class TransformerModel(nn.Module):
    def __init__(
        self,
        seq_len: int,
        d_obs: int,
        d_model: int,
        num_heads: int,
        enc_num_layers: int,
        dec_num_layers: int,
        enc_dropout: float = 0.2,
        dec_dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.encoder = self.TransformerEncoder(
            seq_len, d_obs, d_model, num_heads, enc_num_layers, enc_dropout
        )
        self.decoder = self.TransformerDecoder(
            d_obs, d_model, num_heads, dec_num_layers, dec_dropout
        )

    def forward(self, enc_input: torch.Tensor, dec_input: torch.Tensor,
                enc_mask: torch.Tensor, dec_mask: torch.Tensor) -> torch.Tensor:

        enc_output = self.encoder(enc_input)
        dec_output = self.decoder(dec_input, enc_output, enc_mask, dec_mask)
        return dec_output

    class TransformerEncoder(nn.Module):
        def __init__(
            self,
            seq_len: int,
            d_obs: int,
            d_model: int,
            num_heads: int,
            enc_num_layers: int,
            enc_dropout: float,
        ) -> None:
            super().__init__()

            self.input_layer = nn.Linear(in_features=d_obs, out_features=d_model)
            self.pe_layer = PositionalEncoding(d_model=d_model, seq_len=seq_len)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=num_heads, dropout=enc_dropout, batch_first=True
            )
            self.encoder_layers = nn.TransformerEncoder(
                encoder_layer=encoder_layer, num_layers=enc_num_layers
            )

        def forward(self, enc_input: torch.Tensor) -> torch.Tensor:

            int_input = self.input_layer(enc_input)
            int_input = self.pe_layer(int_input)
            enc_output = self.encoder_layers(int_input)
            return enc_output

    class TransformerDecoder(nn.Module):
        def __init__(
            self,
            d_obs: int,
            d_model: int,
            num_heads: int,
            dec_num_layers: int,
            dec_dropout: float,
        ) -> None:
            super().__init__()

            self.input_layer = nn.Linear(in_features=d_obs, out_features=d_model)
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=d_model, nhead=num_heads, dropout=dec_dropout, batch_first=True
            )
            self.decoder_layers = nn.TransformerDecoder(
                decoder_layer=decoder_layer, num_layers=dec_num_layers
            )
            self.output_layer = nn.Linear(in_features=d_model, out_features=d_obs)

        def forward(self, dec_input: torch.Tensor, enc_output: torch.Tensor,
                    enc_mask: torch.Tensor, dec_mask: torch.Tensor) -> torch.Tensor:

            int_input = self.input_layer(dec_input)
            int_output = self.decoder_layers(
                tgt=int_input,
                tgt_mask=dec_mask,
                memory=enc_output,
                memory_mask=enc_mask,
            )
            dec_output = self.output_layer(int_output)
            return dec_output



    class TransformerEncoderLayer(nn.TransformerEncoderLayer):
        def __init__(self, d_model, nhead, esn, dim_feedforward=2048, dropout=0.1, activation="relu", batch_first=False, norm_first=False, **kwargs):
            super(TransformerEncoderLayer, self).__init__(d_model, nhead, dim_feedforward=dim_feedforward, dropout=dropout, activation=activation, batch_first=batch_first, norm_first=norm_first, **kwargs)
            self.esn = esn
            self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first, **kwargs)
            # Implementation of Feedforward model
            self.linear1 = nn.Linear(d_model, dim_feedforward)
            self.dropout = nn.Dropout(dropout)
            self.linear2 = nn.Linear(dim_feedforward, d_model)

            self.norm_first = norm_first
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.dropout1 = nn.Dropout(dropout)
            self.dropout2 = nn.Dropout(dropout)

            self.activation = nn.functional.relu if activation == "relu" else nn.functional.gelu

        def forward(self, src, src_mask=None, src_key_padding_mask=None):
            # ESN の forward メソッドを呼び出して内部状態を更新
            x = self.esn._esn_cell(src)  # 直接 ESNCell を呼び出して隠れ状態を取得
            if self.norm_first:
                x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
                x = x + self._ff_block(self.norm2(x))
            else:
                x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
                x = self.norm2(x + self._ff_block(x))
            return x

        # Self-attention block
        def _sa_block(self, x, attn_mask, key_padding_mask):
            x, _ = self.self_attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
            return self.dropout1(x)

        # Feed forward block
        def _ff_block(self, x):
            x = self.linear2(self.dropout(self.activation(self.linear1(x))))
            return self.dropout2(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int) -> None:
        super().__init__()

        p_pos = torch.arange(0, seq_len).unsqueeze(1)
        p_i = torch.arange(0, d_model)

        PE = (p_pos / (1000**(2*p_i/d_model))).unsqueeze(0)
        PE[0, :, 0::2] = torch.sin(PE[:, :, 0::2])
        PE[0, :, 1::2] = torch.cos(PE[:, :, 1::2])
        self.PE = PE

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.PE[:, :x.shape[1], :]
