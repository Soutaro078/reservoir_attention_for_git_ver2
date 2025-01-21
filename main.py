import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime
import os
import numpy as np
import matplotlib.pyplot as plt
from dataset import TransformerDataset
from reservoir_attention import ReservoirWithAttention
from echotorch.nn import LiESN
import echotorch.utils.matrix_generation as mg

#ローレンツ96モデルの知識があれば十分

def T_step_prediction(
    model: nn.Module, initial_enc_input: torch.Tensor, initial_dec_input: torch.Tensor, enc_mask: torch.Tensor, dec_mask: torch.Tensor, T: int
) -> torch.Tensor:
    # 最初の次元にバッチ次元を追加する
    if len(initial_enc_input.shape) < 3:
        initial_enc_input = initial_enc_input.unsqueeze(0)
    if len(initial_dec_input.shape) < 3:
        initial_dec_input = initial_dec_input.unsqueeze(0)

    prediction = initial_dec_input

    enc_input = initial_enc_input
    dec_input = initial_dec_input

    for t in range(T):
        seq_len = dec_input.size(1)
        dec_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        with torch.no_grad():
            dec_output = model(enc_input, dec_input, enc_mask=None, dec_mask=dec_mask)
        
        enc_input = torch.cat((enc_input[:, 1:, :], dec_output), dim=1)
        dec_input = dec_output
        prediction = torch.cat((prediction, dec_output), dim=1)

    return prediction

def compare_spectra(model: nn.Module, test_data: np.ndarray, enc_seq_len: int, dec_seq_len: int, sigma: float):
    if len(test_data.shape) < 3:
        test_data = test_data[None, :, :]
    print(test_data.shape)
    initial_enc_input = torch.zeros((1, enc_seq_len, test_data.shape[-1]))
    initial_dec_input = torch.randn((1, 1, test_data.shape[-1]))
    enc_mask = torch.ones((enc_seq_len, enc_seq_len))
    if enc_mask.size(0) != enc_seq_len:
        enc_mask = enc_mask.expand(enc_seq_len, enc_seq_len)
    dec_mask = torch.ones((dec_seq_len, dec_seq_len))

    prediction = T_step_prediction(
        model, initial_enc_input, initial_dec_input, enc_mask, dec_mask, test_data.shape[1]
    )
    prediction = prediction[:, 1:, :].numpy()

    ps_error = power_spectrum_error(prediction, test_data, sigma=sigma)

    # 生成したローレンツの軌跡を保存
    fig1 = plt.figure(figsize=(10, 8))
    ax1 = fig1.add_subplot(111, projection="3d")
    prediction = prediction.squeeze()
    ax1.plot(prediction[:, 0], prediction[:, 1], prediction[:, 2], color="red", label="Generated Trajectory")
    ax1.set_title("Generated Trajectory", fontsize=12)
    fig1_path = "/app/result/lorenz96_reservoir_attention_generated_trajectory.pdf"  # PDF形式で保存
    plt.savefig(fig1_path, format='pdf')
    plt.show()

    # パワースペクトルを保存
    fig2 = plt.figure(figsize=(10, 8))
    ax2 = fig2.add_subplot(111)
    ax2.plot(ps_error[1].squeeze(), color="blue", label="True Spectrum")
    ax2.plot(ps_error[2].squeeze(), color="red", label="Generated Spectrum")
    ax2.set_yscale("log")
    ax2.set_xscale("log")
    ax2.legend(frameon=False, fontsize=10)
    ax2.set_title("Powerspectra", fontsize=12)
    plt.tight_layout()
    fig2_path = "/app/result/lorenz96_reservoir_attention_powerspectra.pdf"  # PDF形式で保存
    plt.savefig(fig2_path, format='pdf')
    plt.show()

    return prediction, ps_error

def power_spectrum_error(prediction, test_data, sigma):
    return np.random.rand(3, 100), np.random.rand(100), np.random.rand(100)

def train(
    fpath: str,
    enc_seq_len: int,
    target_seq_len: int,
    d_obs: int,
    d_model: int,
    num_heads: int,
    enc_num_layers: int,
    dec_num_layers: int,
    num_epochs: int,
    batchsize: int,
    enc_dropout: int = .2,
    dec_dropout: int = .2,
    learning_rate: dict = {0: 1e-3},
) -> nn.Module:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_dataset = TransformerDataset(fpath, enc_seq_len, target_seq_len)
    print(f"Train dataset size: {len(train_dataset)}")  # デバッグ用
    train_loader = DataLoader(train_dataset, batchsize)
    enc_mask = train_dataset.enc_mask
    dec_mask = train_dataset.dec_mask

    input_dim = d_obs
    hidden_dim = d_obs
    output_dim = d_obs
    leaky_rate = 0.1
    spectral_radius = 0.99

    w_generator = mg.NormalMatrixGenerator(size=(hidden_dim, hidden_dim), spectral_radius=spectral_radius)
    win_generator = mg.NormalMatrixGenerator(size=(input_dim, hidden_dim))
    wbias_generator = mg.NormalMatrixGenerator(size=(1, hidden_dim))

    esn = LiESN(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        leaky_rate=leaky_rate,
        w_generator=w_generator,
        win_generator=win_generator,
        wbias_generator=wbias_generator,
        washout=0  # 適切なwashout値を設定
    )

    model = ReservoirWithAttention(
        seq_len=enc_seq_len,
        d_obs=d_obs,
        d_model=d_model,
        num_heads=num_heads,
        enc_num_layers=enc_num_layers,
        dec_num_layers=dec_num_layers,
        enc_dropout=enc_dropout,
        dec_dropout=dec_dropout,
        esn=esn
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate[0])
    loss_fn = nn.MSELoss()

    mse_losses = []  # 各エポックごとの MSE を記録するリスト

    for epoch in range(num_epochs):
        epoch_iterator = tqdm(train_loader)
        sum_loss = 0
        model.train()
        if epoch in learning_rate.keys():
            for g in optimizer.param_groups:
                g['lr'] = learning_rate[epoch]
            print("Changed learning rate to:", optimizer.param_groups[0]['lr'])
        for (enc_input, dec_input, target) in epoch_iterator:
            enc_input, dec_input, target = enc_input.to(device), dec_input.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(enc_input, dec_input, enc_mask, dec_mask)
            loss = loss_fn(output, target)
            sum_loss += loss.item()
            epoch_iterator.set_description(f"Loss={loss.item()}")

            loss.backward()
            optimizer.step()
        avg_loss = sum_loss / len(train_loader)
        mse_losses.append(avg_loss)
        print(f"Epoch {epoch} Loss: {avg_loss}")

    # エポックごとの MSE をプロット
    plt.figure()
    plt.plot(range(num_epochs), mse_losses, marker='o', linestyle='-', color='b')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.title('Training Loss over Epochs')
    plt.grid(True)
    plt.savefig('/app/result/training_loss_01.pdf', format='pdf')  # PDF形式で保存
    plt.show()

    return model

if __name__ == "__main__":
    print('start')
    model = train(
        '/app/data/lorenz96_on0.05_train.npy',
        enc_seq_len=10,
        target_seq_len=4,
        d_obs=20,
        d_model=512,
        num_heads=8,
        enc_num_layers=4,
        dec_num_layers=4,
        num_epochs=1,
        batchsize=512,
        enc_dropout=0.4,
        dec_dropout=0.4
    )

    os.makedirs('/app/result', exist_ok=True)
    model_path = f"/app/result/lorenz96_reservoir_attention_{datetime.now():%Y%m%d_%H%M%S}.pt"
    torch.save(model, model_path)

    test_data = np.load('/app/data/lorenz96_test.npy')

    prediction, ps_error = compare_spectra(model, test_data, enc_seq_len=10, dec_seq_len=4, sigma=0.1)
    print(test_data.shape)
    print("Prediction and power spectrum error calculated.")



#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# やりたいことをやると良い
'''
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime
import os
import numpy as np
import matplotlib.pyplot as plt

from dataset import TransformerDataset
from reservoir_attention import ReservoirWithAttention
from echotorch.nn import LiESN
import echotorch.utils.matrix_generation as mg


# ===============================
#EarlyStopping クラス
# ===============================
class EarlyStopping:
    def __init__(self, patience=10, delta=0.001, save_path='./models/best_model.pt'):
        self.patience = patience
        self.delta = delta
        self.save_path = save_path
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False

    def __call__(self, loss, model, epoch, optimizer):
        if loss < self.best_loss - self.delta:
            self.best_loss = loss
            self.counter = 0
            # ベストモデルを保存
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, self.save_path)
            print(f" Best Model Saved at Epoch {epoch}")
        else:
            self.counter += 1
            print(f" EarlyStopping Counter: {self.counter} / {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True


# ===============================
# T-step 予測関数
# ===============================
def T_step_prediction(
    model: nn.Module, initial_enc_input: torch.Tensor, initial_dec_input: torch.Tensor, enc_mask: torch.Tensor, dec_mask: torch.Tensor, T: int
) -> torch.Tensor:
    if len(initial_enc_input.shape) < 3:
        initial_enc_input = initial_enc_input.unsqueeze(0)
    if len(initial_dec_input.shape) < 3:
        initial_dec_input = initial_dec_input.unsqueeze(0)

    prediction = initial_dec_input
    enc_input = initial_enc_input
    dec_input = initial_dec_input

    for t in range(T):
        with torch.no_grad():
            dec_output = model(enc_input, dec_input, enc_mask, dec_mask)
        enc_input = torch.cat((enc_input[:, 1:, :], dec_output), dim=1)
        dec_input = dec_output
        prediction = torch.cat((prediction, dec_output), dim=1)

    return prediction


# ===============================
# スペクトル比較関数
# ===============================
def compare_spectra(model: nn.Module, test_data: np.ndarray, enc_seq_len: int, dec_seq_len: int, sigma: float):
    if len(test_data.shape) < 3:
        test_data = test_data[None, :, :]

    initial_enc_input = torch.zeros((1, enc_seq_len, test_data.shape[-1]))
    initial_dec_input = torch.randn((1, 1, test_data.shape[-1]))
    enc_mask = torch.ones((dec_seq_len, enc_seq_len))
    dec_mask = torch.ones((dec_seq_len, dec_seq_len))

    prediction = T_step_prediction(
        model, initial_enc_input, initial_dec_input, enc_mask, dec_mask, test_data.shape[1]
    )
    prediction = prediction[:, 1:, :].numpy()

    ps_error = power_spectrum_error(prediction, test_data, sigma=sigma)

    fig = plt.figure(figsize=(10, 8))
    ax1 = fig.add_subplot(221, projection="3d")
    prediction = prediction.squeeze()
    ax1.plot(prediction[:, 0], prediction[:, 1], prediction[:, 2], color="red", label="Generated Trajectory")
    ax1.set_title("Generated Trajectory", fontsize=12)
    ax2 = fig.add_subplot(222, projection="3d")
    test_data = test_data.squeeze()
    ax2.plot(test_data[:, 0], test_data[:, 1], test_data[:, 2], color="blue", label="True Trajectory")
    ax2.set_title("True Trajectory", fontsize=12)
    ax3 = fig.add_subplot(212)
    ax3.plot(ps_error[1].squeeze(), color="blue", label="True Spectrum")
    ax3.plot(ps_error[2].squeeze(), color="red", label="Generated Spectrum")
    ax3.set_yscale("log")
    ax3.set_xscale("log")
    ax3.legend(frameon=False, fontsize=10)
    ax3.set_title("Powerspectra", fontsize=12)
    plt.tight_layout()

    return prediction, ps_error


def power_spectrum_error(prediction, test_data, sigma):
    return np.random.rand(3, 100), np.random.rand(100), np.random.rand(100)


# ===============================
# 学習関数
# ===============================
def train(
    fpath: str,
    enc_seq_len: int,
    target_seq_len: int,
    d_obs: int,
    d_model: int,
    num_heads: int,
    enc_num_layers: int,
    dec_num_layers: int,
    num_epochs: int,
    batchsize: int,
    enc_dropout: int = .2,
    dec_dropout: int = .2,
    learning_rate: dict = {0: 1e-3},
    resume_from_checkpoint: str = None
) -> nn.Module:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_dataset = TransformerDataset(fpath, enc_seq_len, target_seq_len)
    train_loader = DataLoader(train_dataset, batchsize)
    enc_mask = train_dataset.enc_mask
    dec_mask = train_dataset.dec_mask

    model = ReservoirWithAttention(...)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate[0])
    loss_fn = nn.MSELoss()
    start_epoch = 0

    if resume_from_checkpoint:
        checkpoint = torch.load(resume_from_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1

    early_stopping = EarlyStopping()

    for epoch in range(start_epoch, num_epochs):
        sum_loss = 0
        for enc_input, dec_input, target in tqdm(train_loader):
            optimizer.zero_grad()
            output = model(enc_input.to(device), dec_input.to(device), enc_mask, dec_mask)
            loss = loss_fn(output, target.to(device))
            loss.backward()
            optimizer.step()
            sum_loss += loss.item()
        
        avg_loss = sum_loss / len(train_loader)
        early_stopping(avg_loss, model, epoch, optimizer)
        if early_stopping.early_stop:
            break

    return model


# ===============================
# メイン関数
# ===============================
if __name__ == "__main__":
    model = train(
        '/app/data/lorenz63_test.npy',
        enc_seq_len=10,
        target_seq_len=4,
        d_obs=3,
        d_model=512,
        num_heads=8,
        enc_num_layers=4,
        dec_num_layers=4,
        num_epochs=15,
        batchsize=512,
        resume_from_checkpoint='./models/best_model.pt'
    )

'''

#------------------------------------------------------------------------------------------


# # GPUが使えない場合のやつ
# #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader
# from tqdm import tqdm
# from datetime import datetime
# import os
# import numpy as np
# import matplotlib.pyplot as plt

# from dataset import TransformerDataset
# from reservoir_attention import ReservoirWithAttention
# from echotorch.nn import LiESN
# import echotorch.utils.matrix_generation as mg

# def T_step_prediction(
#     model: nn.Module, initial_enc_input: torch.Tensor, initial_dec_input: torch.Tensor, enc_mask: torch.Tensor, dec_mask: torch.Tensor, T: int
# ) -> torch.Tensor:
#     # 最初の次元にバッチ次元を追加する
#     if len(initial_enc_input.shape) < 3:
#         initial_enc_input = initial_enc_input.unsqueeze(0)
#     if len(initial_dec_input.shape) < 3:
#         initial_dec_input = initial_dec_input.unsqueeze(0)

#     prediction = initial_dec_input

#     enc_input = initial_enc_input
#     dec_input = initial_dec_input

#     for t in range(T):
#         with torch.no_grad():
#             dec_output = model(enc_input, dec_input, enc_mask, dec_mask)
#         # シーケンス次元（時間次元）方向に結合
#         enc_input = torch.cat((enc_input[:, 1:, :], dec_output), dim=1)
#         dec_input = dec_output
#         prediction = torch.cat((prediction, dec_output), dim=1)

#     return prediction

# def compare_spectra(model: nn.Module, test_data: np.ndarray, enc_seq_len: int, dec_seq_len: int, sigma: float):
#     if len(test_data.shape) < 3:
#         test_data = test_data[None, :, :]

#     initial_enc_input = torch.zeros((1, enc_seq_len, test_data.shape[-1]))
#     initial_dec_input = torch.randn((1, 1, test_data.shape[-1]))
#     enc_mask = torch.ones((dec_seq_len, enc_seq_len))
#     dec_mask = torch.ones((dec_seq_len, dec_seq_len))

#     prediction = T_step_prediction(
#         model, initial_enc_input, initial_dec_input, enc_mask, dec_mask, test_data.shape[1]
#     )
#     prediction = prediction[:, 1:, :].numpy()

#     ps_error = power_spectrum_error(prediction, test_data, sigma=sigma)

#     fig = plt.figure(figsize=(10, 8))
#     ax1 = fig.add_subplot(221, projection="3d")
#     prediction = prediction.squeeze()
#     ax1.plot(prediction[:, 0], prediction[:, 1], prediction[:, 2], color="red", label="Generated Trajectory")
#     ax1.set_title("Generated Trajectory", fontsize=12)
#     ax2 = fig.add_subplot(222, projection="3d")
#     test_data = test_data.squeeze()
#     ax2.plot(test_data[:, 0], test_data[:, 1], test_data[:, 2], color="blue", label="True Trajectory")
#     ax2.set_title("True Trajectory", fontsize=12)
#     ax3 = fig.add_subplot(212)
#     ax3.plot(ps_error[1].squeeze(), color="blue", label="True Spectrum")
#     ax3.plot(ps_error[2].squeeze(), color="red", label="Generated Spectrum")
#     ax3.set_yscale("log")
#     ax3.set_xscale("log")
#     ax3.legend(frameon=False, fontsize=10)
#     ax3.set_title("Powerspectra", fontsize=12)
#     plt.tight_layout()

#     return prediction, ps_error

# def power_spectrum_error(prediction, test_data, sigma):
#     # ダミーの実装。実際の実装に置き換えてください。
#     return np.random.rand(3, 100), np.random.rand(100), np.random.rand(100)

# def train(
#     fpath: str,
#     enc_seq_len: int,
#     target_seq_len: int,
#     d_obs: int,
#     d_model: int,
#     num_heads: int,
#     enc_num_layers: int,
#     dec_num_layers: int,
#     num_epochs: int,
#     batchsize: int,
#     enc_dropout: int = .2,
#     dec_dropout: int = .2,
#     learning_rate: dict = {0: 1e-3},
# ) -> nn.Module:
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     train_dataset = TransformerDataset(fpath, enc_seq_len, target_seq_len)
#     train_loader = DataLoader(train_dataset, batchsize)
#     enc_mask = train_dataset.enc_mask
#     dec_mask = train_dataset.dec_mask

#     # モデルのパラメータ
#     input_dim = d_obs
#     hidden_dim = d_model
#     output_dim = d_model
#     leaky_rate = 0.3
#     spectral_radius = 0.9

#     # 重み行列の生成
#     w_generator = mg.NormalMatrixGenerator(size=(hidden_dim, hidden_dim), spectral_radius=spectral_radius)
#     win_generator = mg.NormalMatrixGenerator(size=(input_dim, hidden_dim))
#     wbias_generator = mg.NormalMatrixGenerator(size=(1, hidden_dim))

#     # LiESN モデルの作成
#     esn = LiESN(
#         input_dim=input_dim,
#         hidden_dim=hidden_dim,
#         output_dim=output_dim,
#         leaky_rate=leaky_rate,
#         w_generator=w_generator,
#         win_generator=win_generator,
#         wbias_generator=wbias_generator
#     )

#     model = ReservoirWithAttention(hidden_dim=d_model, output_dim=d_obs, n_heads=num_heads, embed_dim=d_model, esn=esn).to(device)

#     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate[0])
#     loss_fn = nn.MSELoss()

#     for epoch in range(num_epochs):
#         epoch_iterator = tqdm(train_loader)
#         sum_loss = 0
#         model.train()
#         if epoch in learning_rate.keys():
#             for g in optimizer.param_groups:
#                 g['lr'] = learning_rate[epoch]
#             print("Changed learning rate to:", optimizer.param_groups[0]['lr'])
#         for (enc_input, dec_input, target) in epoch_iterator:
#             enc_input, dec_input, target = enc_input.to(device), dec_input.to(device), target.to(device)
#             optimizer.zero_grad()
#             output = model(enc_input, dec_input, enc_mask, dec_mask)
#             loss = loss_fn(output, target)
#             sum_loss += loss.item()
#             epoch_iterator.set_description(f"Loss={loss.item()}")

#             loss.backward()
#             optimizer.step()
#         print(f"Epoch {epoch} Loss: {sum_loss/len(train_loader)}")
        
#     return model

# if __name__ == "__main__":
#     print('start')
#     model = train(
#         '/app/data/lorenz63_test.npy',
#         enc_seq_len=10,
#         target_seq_len=4,
#         d_obs=3,
#         d_model=512,
#         num_heads=8,
#         enc_num_layers=4,
#         dec_num_layers=4,
#         num_epochs=5,
#         batchsize=512,
#         enc_dropout=0.4,
#         dec_dropout=0.4
#     )

#     # models ディレクトリが存在しない場合は作成
#     os.makedirs('./models', exist_ok=True)
#     torch.save(model, f"./models/model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt")

#     # テストデータの読み込み
#     test_data = np.load('/app/data/lorenz63_test.npy')

#     # モデルのテスト
#     prediction, ps_error = compare_spectra(model, test_data, enc_seq_len=10, dec_seq_len=4, sigma=0.1)
#     print("Prediction and power spectrum error calculated.")

# GPUが使える場合のやつ
# ------------------------------------------------------------------------------------------
# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader
# from tqdm import tqdm
# from datetime import datetime
# import os
# import numpy as np
# import matplotlib.pyplot as plt

# from dataset import TransformerDataset
# from reservoir_attention import ReservoirWithAttention
# from echotorch.nn import LiESN
# import echotorch.utils.matrix_generation as mg

# def T_step_prediction(
#     model: nn.Module, initial_enc_input: torch.Tensor, initial_dec_input: torch.Tensor, enc_mask: torch.Tensor, dec_mask: torch.Tensor, T: int
# ) -> torch.Tensor:
#     # 最初の次元にバッチ次元を追加する
#     if len(initial_enc_input.shape) < 3:
#         initial_enc_input = initial_enc_input.unsqueeze(0)
#     if len(initial_dec_input.shape) < 3:
#         initial_dec_input = initial_dec_input.unsqueeze(0)

#     prediction = initial_dec_input

#     enc_input = initial_enc_input
#     dec_input = initial_dec_input

#     for t in range(T):
#         with torch.no_grad():
#             dec_output = model(enc_input, dec_input, enc_mask, dec_mask)
#         # シーケンス次元（時間次元）方向に結合
#         enc_input = torch.cat((enc_input[:, 1:, :], dec_output), dim=1)
#         dec_input = dec_output
#         prediction = torch.cat((prediction, dec_output), dim=1)

#     return prediction

# def compare_spectra(model: nn.Module, test_data: np.ndarray, enc_seq_len: int, dec_seq_len: int, sigma: float):
#     if len(test_data.shape) < 3:
#         test_data = test_data[None, :, :]

#     initial_enc_input = torch.zeros((1, enc_seq_len, test_data.shape[-1]))
#     initial_dec_input = torch.randn((1, 1, test_data.shape[-1]))
#     enc_mask = torch.ones((dec_seq_len, enc_seq_len))
#     dec_mask = torch.ones((dec_seq_len, dec_seq_len))

#     prediction = T_step_prediction(
#         model, initial_enc_input, initial_dec_input, enc_mask, dec_mask, test_data.shape[1]
#     )
#     prediction = prediction[:, 1:, :].numpy()

#     ps_error = power_spectrum_error(prediction, test_data, sigma=sigma)

#     fig = plt.figure(figsize=(10, 8))
#     ax1 = fig.add_subplot(221, projection="3d")
#     prediction = prediction.squeeze()
#     ax1.plot(prediction[:, 0], prediction[:, 1], prediction[:, 2], color="red", label="Generated Trajectory")
#     ax1.set_title("Generated Trajectory", fontsize=12)
#     ax2 = fig.add_subplot(222, projection="3d")
#     test_data = test_data.squeeze()
#     ax2.plot(test_data[:, 0], test_data[:, 1], test_data[:, 2], color="blue", label="True Trajectory")
#     ax2.set_title("True Trajectory", fontsize=12)
#     ax3 = fig.add_subplot(212)
#     ax3.plot(ps_error[1].squeeze(), color="blue", label="True Spectrum")
#     ax3.plot(ps_error[2].squeeze(), color="red", label="Generated Spectrum")
#     ax3.set_yscale("log")
#     ax3.set_xscale("log")
#     ax3.legend(frameon=False, fontsize=10)
#     ax3.set_title("Powerspectra", fontsize=12)
#     plt.tight_layout()

#     return prediction, ps_error

# def power_spectrum_error(prediction, test_data, sigma):
#     # ダミーの実装。実際の実装に置き換えてください。
#     return np.random.rand(3, 100), np.random.rand(100), np.random.rand(100)

# def train(
#     fpath: str,
#     enc_seq_len: int,
#     target_seq_len: int,
#     d_obs: int,
#     d_model: int,
#     num_heads: int,
#     enc_num_layers: int,
#     dec_num_layers: int,
#     num_epochs: int,
#     batchsize: int,
#     enc_dropout: int = .2,
#     dec_dropout: int = .2,
#     learning_rate: dict = {0: 1e-3},
# ) -> nn.Module:
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     train_dataset = TransformerDataset(fpath, enc_seq_len, target_seq_len)
#     train_loader = DataLoader(train_dataset, batchsize)
#     enc_mask = train_dataset.enc_mask
#     dec_mask = train_dataset.dec_mask

#     # モデルのパラメータ
#     input_dim = d_obs
#     hidden_dim = d_model
#     output_dim = d_model
#     leaky_rate = 0.3
#     spectral_radius = 0.9

#     # 重み行列の生成
#     w_generator = mg.NormalMatrixGenerator(size=(hidden_dim, hidden_dim), spectral_radius=spectral_radius)
#     win_generator = mg.NormalMatrixGenerator(size=(input_dim, hidden_dim))
#     wbias_generator = mg.NormalMatrixGenerator(size=(1, hidden_dim))

#     # LiESN モデルの作成
#     esn = LiESN(
#         input_dim=input_dim,
#         hidden_dim=hidden_dim,
#         output_dim=output_dim,
#         leaky_rate=leaky_rate,
#         w_generator=w_generator,
#         win_generator=win_generator,
#         wbias_generator=wbias_generator
#     )

#     model = ReservoirWithAttention(hidden_dim=d_model, output_dim=d_obs, n_heads=num_heads, embed_dim=d_model, esn=esn).to(device)

#     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate[0])
#     loss_fn = nn.MSELoss()

#     for epoch in range(num_epochs):
#         epoch_iterator = tqdm(train_loader)
#         sum_loss = 0
#         model.train()
#         if epoch in learning_rate.keys():
#             for g in optimizer.param_groups:
#                 g['lr'] = learning_rate[epoch]
#             print("Changed learning rate to:", optimizer.param_groups[0]['lr'])
#         for (enc_input, dec_input, target) in epoch_iterator:
#             enc_input, dec_input, target = enc_input.to(device), dec_input.to(device), target.to(device)
#             optimizer.zero_grad()
#             output = model(enc_input, dec_input, enc_mask, dec_mask)
#             loss = loss_fn(output, target)
#             sum_loss += loss.item()
#             epoch_iterator.set_description(f"Loss={loss.item()}")

#             loss.backward()
#             optimizer.step()
#         print(f"Epoch {epoch} Loss: {sum_loss/len(train_loader)}")
        
#     return model

# if __name__ == "__main__":
#     print('start')
#     model = train(
#         '/app/data/lorenz63_test.npy',
#         enc_seq_len=10,
#         target_seq_len=4,
#         d_obs=3,
#         d_model=512,
#         num_heads=8,
#         enc_num_layers=4,
#         dec_num_layers=4,
#         num_epochs=5,
#         batchsize=512,
#         enc_dropout=0.4,
#         dec_dropout=0.4
#     )

#     # models ディレクトリが存在しない場合は作成
#     os.makedirs('./models', exist_ok=True)
#     torch.save(model, f"./models/model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt")

#     # テストデータの読み込み
#     test_data = np.load('/app/data/lorenz63_test.npy')

#     # モデルのテスト
#     prediction, ps_error = compare_spectra(model, test_data, enc_seq_len=10, dec_seq_len=4, sigma=0.1)
#     print("Prediction and power spectrum error calculated.")


# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader
# from tqdm import tqdm
# from datetime import datetime
# import os

# from dataset import TransformerDataset
# from reservoir_attention import ReservoirWithAttention
# from echotorch.nn import LiESN
# import echotorch.utils.matrix_generation as mg

# def train(
#     fpath: str,
#     enc_seq_len: int,
#     target_seq_len: int,
#     d_obs: int,
#     d_model: int,
#     num_heads: int,
#     enc_num_layers: int,
#     dec_num_layers: int,
#     num_epochs: int,
#     batchsize: int,
#     enc_dropout: int = .2,
#     dec_dropout: int = .2,
#     learning_rate: dict = {0: 1e-3},
# ) -> None:
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     train_dataset = TransformerDataset(fpath, enc_seq_len, target_seq_len)
#     train_loader = DataLoader(train_dataset, batchsize)
#     enc_mask = train_dataset.enc_mask
#     dec_mask = train_dataset.dec_mask

#     # モデルのパラメータ
#     input_dim = d_obs
#     hidden_dim = d_model
#     output_dim = d_model
#     leaky_rate = 0.3
#     spectral_radius = 0.9

#     # 重み行列の生成
#     w_generator = mg.NormalMatrixGenerator(size=(hidden_dim, hidden_dim), spectral_radius=spectral_radius)
#     win_generator = mg.NormalMatrixGenerator(size=(input_dim, hidden_dim))
#     wbias_generator = mg.NormalMatrixGenerator(size=(1, hidden_dim))

#     # LiESN モデルの作成
#     esn = LiESN(
#         input_dim=input_dim,
#         hidden_dim=hidden_dim,
#         output_dim=output_dim,
#         leaky_rate=leaky_rate,
#         w_generator=w_generator,
#         win_generator=win_generator,
#         wbias_generator=wbias_generator
#     )

#     model = ReservoirWithAttention(hidden_dim=d_model, output_dim=d_obs, n_heads=num_heads, embed_dim=d_model, esn=esn).to(device)

#     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate[0])
#     loss_fn = nn.MSELoss()

#     for epoch in range(num_epochs):
#         epoch_iterator = tqdm(train_loader)
#         sum_loss = 0
#         model.train()
#         if epoch in learning_rate.keys():
#             for g in optimizer.param_groups:
#                 g['lr'] = learning_rate[epoch]
#             print("Changed learning rate to:", optimizer.param_groups[0]['lr'])
#         for (enc_input, dec_input, target) in epoch_iterator:
#             enc_input, dec_input, target = enc_input.to(device), dec_input.to(device), target.to(device)
#             optimizer.zero_grad()
#             output = model(enc_input, dec_input, enc_mask, dec_mask)
#             loss = loss_fn(output, target)
#             sum_loss += loss.item()
#             epoch_iterator.set_description(f"Loss={loss.item()}")

#             loss.backward()
#             optimizer.step()
#         print(f"Epoch {epoch} Loss: {sum_loss/len(train_loader)}")
        
#     return model

# if __name__ == "__main__":
#     print('start')
#     model = train(
#         '/app/data/lorenz63_test.npy',
#         enc_seq_len=10,
#         target_seq_len=4,
#         d_obs=3,
#         d_model=512,
#         num_heads=8,
#         enc_num_layers=4,
#         dec_num_layers=4,
#         num_epochs=5,
#         batchsize=512,
#         enc_dropout=0.4,
#         dec_dropout=0.4
#     )

#     # models ディレクトリが存在しない場合は作成
#     os.makedirs('./models', exist_ok=True)
#     torch.save(model, f"./models/model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt")



# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader
# from tqdm import tqdm
# from datetime import datetime
# import os

# from dataset import TransformerDataset
# from reservoir_attention import ReservoirWithAttention
# from echotorch.nn import LiESN
# import echotorch.utils.matrix_generation as mg

# def train(
#     fpath: str,
#     enc_seq_len: int,
#     target_seq_len: int,
#     d_obs: int,
#     d_model: int,
#     num_heads: int,
#     enc_num_layers: int,
#     dec_num_layers: int,
#     num_epochs: int,
#     batchsize: int,
#     enc_dropout: int = .2,
#     dec_dropout: int = .2,
#     learning_rate: dict = {0: 1e-3},
# ) -> None:
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     train_dataset = TransformerDataset(fpath, enc_seq_len, target_seq_len)
#     train_loader = DataLoader(train_dataset, batchsize)
#     enc_mask = train_dataset.enc_mask
#     dec_mask = train_dataset.dec_mask

#     # モデルのパラメータ（ESN用）
#     input_dim = d_obs
#     hidden_dim = d_model
#     output_dim = d_model
#     leaky_rate = 0.3
#     spectral_radius = 0.9

#     # 重み行列の生成（用）
#     w_generator = mg.NormalMatrixGenerator(size=(hidden_dim, hidden_dim), spectral_radius=spectral_radius)
#     win_generator = mg.NormalMatrixGenerator(size=(input_dim, hidden_dim))
#     wbias_generator = mg.NormalMatrixGenerator(size=(1, hidden_dim))

#     # LiESN モデルの作成
#     esn = LiESN(
#         input_dim=input_dim,
#         hidden_dim=hidden_dim,
#         output_dim=output_dim,
#         leaky_rate=leaky_rate,
#         w_generator=w_generator,
#         win_generator=win_generator,
#         wbias_generator=wbias_generator
#     )

#     model = ReservoirWithAttention(hidden_dim=d_model, output_dim=d_obs, n_heads=num_heads, embed_dim=d_model, esn=esn).to(device)

#     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate[0])
#     loss_fn = nn.MSELoss()

#     for epoch in range(num_epochs):
#         epoch_iterator = tqdm(train_loader)
#         sum_loss = 0
#         model.train()
#         if epoch in learning_rate.keys():
#             for g in optimizer.param_groups:
#                 g['lr'] = learning_rate[epoch]
#             print("Changed learning rate to:", optimizer.param_groups[0]['lr'])
#         for (enc_input, dec_input, target) in epoch_iterator:
#             enc_input, dec_input, target = enc_input.to(device), dec_input.to(device), target.to(device)
#             optimizer.zero_grad()
#             output = model(enc_input)
#             loss = loss_fn(output, target)
#             sum_loss += loss.item()
#             epoch_iterator.set_description(f"Loss={loss.item()}")

#             loss.backward()
#             optimizer.step()
#         print(f"Epoch {epoch} Loss: {sum_loss/len(train_loader)}")
        
#     return model

# if __name__ == "__main__":
#     print('start')
#     model = train(
#         '/app/data/lorenz63_test.npy',
#         enc_seq_len=10,
#         target_seq_len=4,
#         d_obs=3,
#         d_model=512,
#         num_heads=8,
#         enc_num_layers=4,
#         dec_num_layers=4,
#         num_epochs=5,
#         batchsize=512,
#         enc_dropout=0.4,
#         dec_dropout=0.4
#     )

#     # models ディレクトリが存在しない場合は作成
#     os.makedirs('./models', exist_ok=True)
#     torch.save(model, f"./models/model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt")

# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader
# from tqdm import tqdm
# from datetime import datetime

# from dataset import TransformerDataset
# from model import TransformerModel
# import os

# 	# •	fpath: データセット（.npy ファイル）のパス。
# 	# •	enc_seq_len: エンコーダに入力するシーケンス長。
# 	# •	target_seq_len: デコーダで予測するシーケンス長。
# 	# •	d_obs: 観測データの次元数（入力データの特徴量数）。
# 	# •	d_model: Transformer内部の隠れ層の次元数。
# 	# •	num_heads: マルチヘッドアテンションのヘッド数。
# 	# •	enc_num_layers: エンコーダの層数。
# 	# •	dec_num_layers: デコーダの層数。
# 	# •	num_epochs: トレーニングのエポック数。
# 	# •	batchsize: バッチサイズ。
# 	# •	enc_dropout: エンコーダのドロップアウト率。
# 	# •	dec_dropout: デコーダのドロップアウト率。
# 	# •	learning_rate: エポックごとに設定できる学習率の辞書。

# def train(
#     fpath: str,
#     enc_seq_len: int,
#     target_seq_len: int,
#     d_obs: int,
#     d_model: int,
#     num_heads: int,
#     enc_num_layers: int,
#     dec_num_layers: int,
#     num_epochs: int,
#     batchsize: int,
#     enc_dropout: int = .2,
#     dec_dropout: int = .2,
#     learning_rate: dict = {0: 1e-3},
# ) -> None:
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     train_dataset = TransformerDataset(fpath, enc_seq_len, target_seq_len)
#     train_loader = DataLoader(train_dataset, batchsize)#ここでバッチ単位でデータを取得することができる
#     enc_mask = train_dataset.enc_mask
#     dec_mask = train_dataset.dec_mask

#     model = TransformerModel(
#         seq_len=enc_seq_len+target_seq_len,
#         d_obs=d_obs,
#         d_model=d_model,
#         num_heads=num_heads,
#         enc_num_layers=enc_num_layers,
#         dec_num_layers=dec_num_layers,
#         enc_dropout=enc_dropout,
#         dec_dropout=dec_dropout,
#     )

#     optimizer = torch.optim.Adam(params=model.parameters())
#     loss_fn = nn.MSELoss()
#     train_length = len(train_dataset)

#     history = {"train_loss": []}
#     train_loss = 0


#     for epoch in range(num_epochs):
#         # バッチごとの処理進捗を視覚的に表示。
#         epoch_iterator = tqdm(train_loader)
#         sum_loss = 0
#         model.train()
#         # エポックごとに学習率を変更する場合の処理。
#         if epoch in learning_rate.keys():
#             optimizer.lr = learning_rate[epoch]
#             print("Changed learning rate to:", optimizer.lr)
#         for (enc_input, dec_input, target) in epoch_iterator:
#             # epoch += 1
#             loss = 0
#             optimizer.zero_grad()

#             output = model(enc_input, dec_input, enc_mask, dec_mask)
#             loss = loss_fn(output, target)
#             sum_loss += loss.item()
#             epoch_iterator.set_description(f"Loss={loss.item()}")

#             loss.backward()
#             optimizer.step()
#         print(f"Epoch {epoch} Loss: {sum_loss/len(train_loader)}")
        

#     return model


# if __name__ == "__main__":
#     print('start')
#     model = train(
#         '/app/data/lorenz63_test.npy',
#         enc_seq_len=10,
#         target_seq_len=4,
#         d_obs=3,
#         d_model=512,
#         num_heads=8,
#         enc_num_layers=4,
#         dec_num_layers=4,
#         num_epochs=5,
#         batchsize=512,
#         enc_dropout=0.4,
#         dec_dropout=0.4
#     )

#     # models ディレクトリが存在しない場合は作成
#     os.makedirs('./models', exist_ok=True)
#     torch.save(model, f"./models/model_{datetime.now()}.pt")
