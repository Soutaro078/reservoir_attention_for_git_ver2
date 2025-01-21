import torch
import torch.nn as nn
from echotorch.nn import LiESN

class TransformerEncoder(nn.Module):
    def __init__(self, seq_len, d_obs, d_model, num_heads, enc_num_layers, enc_dropout, esn):
        super(TransformerEncoder, self).__init__()
        self.input_layer = nn.Linear(d_obs, d_model)
        self.pe_layer = PositionalEncoding(d_model, seq_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model, num_heads, dropout=enc_dropout, batch_first=True)
        self.encoder_layers = nn.TransformerEncoder(encoder_layer, num_layers=enc_num_layers)

    def forward(self, enc_input: torch.Tensor) -> torch.Tensor:
        int_input = self.input_layer(enc_input)
        int_input = self.pe_layer(int_input)    
        # エンコーダレイヤーに入力
        enc_output = self.encoder_layers(int_input)
        return enc_output

class TransformerDecoder(nn.Module):
    def __init__(self, d_obs, d_model, num_heads, dec_num_layers, dec_dropout):
        super(TransformerDecoder, self).__init__()
        self.input_layer = nn.Linear(d_obs, d_model)
        decoder_layer = nn.TransformerDecoderLayer(d_model, num_heads, dropout=dec_dropout, batch_first=True)
        self.decoder_layers = nn.TransformerDecoder(decoder_layer, num_layers=dec_num_layers)
        self.output_layer = nn.Linear(d_model, d_obs)

    def forward(self, dec_input, enc_output, enc_mask, dec_mask):
        x = self.input_layer(dec_input)
        x = self.decoder_layers(tgt=x, memory=enc_output, tgt_mask=dec_mask, memory_mask=enc_mask)
        dec_output = self.output_layer(x)
        return dec_output

class TransformerEncoderLayer_New(nn.TransformerEncoderLayer):
    def __init__(self, d_model, nhead, esn, dim_feedforward=2048, dropout=0.1, activation="relu", batch_first=False, norm_first=False, **kwargs):
        super(TransformerEncoderLayer_New, self).__init__(d_model, nhead, dim_feedforward=dim_feedforward, dropout=dropout, activation=activation, batch_first=batch_first, norm_first=norm_first, **kwargs)
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
        # ESN の forward メソッドを呼び出して Wout を通した後の出力を取得
        esn_output = self.esn(src)  # 直接 ESN の出力を取得
        if self.norm_first:
            x = esn_output + self._sa_block(self.norm1(esn_output), src_mask, src_key_padding_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(esn_output + self._sa_block(esn_output, src_mask, src_key_padding_mask))
            x = self.norm2(x + self._ff_block(x))
        return x

    # Self-attention block
    def _sa_block(self, x, attn_mask, key_padding_mask):
        # attn_mask のサイズを調整
        if attn_mask is not None and attn_mask.size() != (x.size(0), x.size(0)):
            attn_mask = attn_mask.expand(x.size(0), x.size(0))
        return self.self_attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)[0]

    # Feed forward block
    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)

# class ReservoirWithAttention(nn.Module):
#     def __init__(self, seq_len, d_obs, d_model, num_heads, enc_num_layers, dec_num_layers, enc_dropout, dec_dropout, esn):
#         super(ReservoirWithAttention, self).__init__()
#         self.encoder = TransformerEncoder(seq_len, d_obs, d_model, num_heads, enc_num_layers, enc_dropout, esn)
#         self.decoder = TransformerDecoder(d_obs, d_model, num_heads, dec_num_layers, dec_dropout)
#         self.mlp = nn.Sequential(
#             nn.Linear(d_model, d_model),
#             nn.ReLU(),
#             nn.Linear(d_model, d_obs)
#         )
#         self.esn = esn
#         if self.esn._esn_cell.washout is None:
#             print("washout is None")  # デバッグ用
#             self.esn._esn_cell.washout = 0  # 適切なwashout値を設定

#     def forward(self, enc_input, dec_input, enc_mask, dec_mask):
#         print(f"enc_input size: {enc_input.size()}, dec_input size: {dec_input.size()}")  # デバッグ用
#         esn_output = self.esn(enc_input)
#         enc_output = self.encoder(esn_output)
#         dec_output = self.decoder(dec_input, enc_output, enc_mask, dec_mask)
#         return dec_output

class ReservoirWithAttention(nn.Module):
    def __init__(self, seq_len, d_obs, d_model, num_heads, enc_num_layers, dec_num_layers, enc_dropout, dec_dropout, esn):
        super(ReservoirWithAttention, self).__init__()
        self.encoder = TransformerEncoder(seq_len, d_obs, d_model, num_heads, enc_num_layers, enc_dropout, esn)
        self.decoder = TransformerDecoder(d_obs, d_model, num_heads, dec_num_layers, dec_dropout)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_obs)
        )
        self.esn = esn
        if self.esn._esn_cell.washout is None:
            self.esn._esn_cell.washout = 20  # 適切なwashout値を設定

    def forward(self, enc_input, dec_input, enc_mask, dec_mask):
        print(f"enc_input size: {enc_input.size()}, dec_input size: {dec_input.size()}")  # デバッグ用
        # ESN に渡す前に次元を調整
        #esn_input = enc_input.view(-1, enc_input.size(2))
        esn_output = self.esn(enc_input)
        #esn_output = self.esn(esn_input)
        print(f"esn_output size: {esn_output.size()}")  # デバッグ用
        #esn_output = esn_output.view(enc_input.size(0), enc_input.size(1), -1)
        enc_output = self.encoder(esn_output)
        dec_output = self.decoder(dec_input, enc_output, enc_mask, dec_mask)
        return dec_output

    # def forward(self, enc_input, dec_input, enc_mask, dec_mask):
    #     # ESN の forward メソッドを呼び出して Wout を通した後の出力を取得
    #     esn_output = self.esn(enc_input)  # 直接 ESN の出力を取得
    #     #
    #     # Transformerエンコーダの出力を取得
    #     enc_output = self.encoder(esn_output)
    #     # enc_output = self.encoder(enc_input)

    #     # Transformerデコーダの出力を取得
    #     dec_output = self.decoder(dec_input, enc_output, enc_mask, dec_mask)
        
    #     # MLPを通して最終出力を取得
    #     # output = self.mlp(dec_output)
    #     return dec_output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, seq_len):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
    #    pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        print(f"x size: {x.size()}, self.pe size: {self.pe.size()}")  # デバッグ用
        x = x + self.pe[:x.size(0), :]
        return x


# import torch
# import torch.nn as nn
# from echotorch.nn import LiESN

# class ReservoirWithAttention(nn.Module):
#     def __init__(self, hidden_dim, output_dim, n_heads, embed_dim, esn):
#         super(ReservoirWithAttention, self).__init__()
#         self.esn = esn
#         self.norm1 = nn.LayerNorm(hidden_dim)
#         self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=n_heads)
#         self.norm2 = nn.LayerNorm(hidden_dim)
#         self.mlp = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, output_dim)
#         )

#     def forward(self, x):
#         # ESN の forward メソッドを呼び出して内部状態を更新
#         hidden_states, _ = self.esn(x)  # ESN の forward メソッドを呼び出して隠れ状態を取得
        
#         # 正規化と注意機構の適用
#         x = self.norm1(hidden_states)
#         attn_out, _ = self.attention(x, x, x)
#         x = self.norm2(attn_out + x)
#         x = self.mlp(x + attn_out)
#         return x