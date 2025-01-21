import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Tuple

from utils import get_mask


class TransformerDataset(Dataset):
    def __init__(
        self,
        fpath: str,
        enc_seq_len: int,
        target_seq_len: int,
        random_perm: bool = True,
        #random_perm: データのランダムな順序を指定するフラグ。
    ) -> None:
        super().__init__()
        #時系列データを読み込む
        total_timeseries = torch.from_numpy(np.load(fpath))
        #時系列データをいくつかのサブシーケンスに分割する
        self.sequences = self._sliding_window_construction(
            data=total_timeseries, N=(enc_seq_len+target_seq_len)
        )
        #ランダムな順序でデータを取得するかどうかでインデックスが変わる
        if random_perm:
            self.indices = torch.randperm(len(self.sequences))
        else:
            self.indices = torch.arange(len(self.sequences))
        self.enc_seq_len = enc_seq_len
        self.target_seq_len = target_seq_len

        self.enc_mask, self.dec_mask = self._get_decoder_masks()

    def __len__(self):
        return len(self.sequences)
    '''
    データセットの長さを取得するためのメソッド
    '''

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sequence = self.sequences[self.indices[index]]
        enc_input = sequence[:self.enc_seq_len]
        dec_input = sequence[self.enc_seq_len-1:-1]
        target = sequence[self.enc_seq_len:]
        return enc_input, dec_input, target

    def _sliding_window_construction(self, data: torch.Tensor, N: int) -> torch.Tensor:
        windows = []
        for i in range(len(data)-N):
            windows.append(data[i:N+i])
        return torch.stack(windows)

    def _get_decoder_masks(self) -> Tuple[torch.Tensor, torch.Tensor]:
        enc_mask = get_mask((self.target_seq_len, self.enc_seq_len))
        dec_mask = get_mask((self.target_seq_len, self.target_seq_len))
        return enc_mask, dec_mask
