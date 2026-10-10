import torch
import torchaudio
from transformers import AutoModel


def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            rl = rl.float().detach()
            gl = gl.float()
            loss += torch.mean(torch.abs(rl - gl))

    return loss * 2


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        dr = dr.float()
        dg = dg.float()
        r_loss = torch.mean((1 - dr) ** 2)
        g_loss = torch.mean(dg**2)
        loss += r_loss + g_loss
        # generator_loss と同様テンソルのまま返す（毎ステップの GPU 同期を防ぐ。使うのはログ出力時だけ）
        r_losses.append(r_loss.detach())
        g_losses.append(g_loss.detach())

    return loss, r_losses, g_losses


def generator_loss(disc_outputs):
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        dg = dg.float()
        gen_loss = torch.mean((1 - dg) ** 2)
        # discriminator_loss と同様、ログ用のリストは detach して返す（毎ステップの GPU 同期を防ぐ）
        gen_losses.append(gen_loss.detach())
        loss += gen_loss

    return loss, gen_losses


def kl_loss(z_p, logs_q, m_p, logs_p, z_mask):
    """
    z_p, logs_q: [b, h, t_t]
    m_p, logs_p: [b, h, t_t]
    """
    z_p = z_p.float()
    logs_q = logs_q.float()
    m_p = m_p.float()
    logs_p = logs_p.float()
    z_mask = z_mask.float()

    kl = logs_p - logs_q - 0.5
    kl += 0.5 * ((z_p - m_p) ** 2) * torch.exp(-2.0 * logs_p)
    kl = torch.sum(kl * z_mask)
    loss = kl / torch.sum(z_mask)
    return loss


class WavLMLoss(torch.nn.Module):
    """SLM(WavLM) ベースの敵対学習損失。

    1ステップあたりの WavLM forward を最小化するため、埋め込み計算（encode）を呼び出し側で共有する:
    - 実音声（no_grad）: 判別器損失と feature matching 損失で共有
    - 生成音声（no_grad）: 判別器損失のみ
    - 生成音声（勾配あり）: feature matching 損失と generator 敵対損失で共有
    """

    def __init__(self, model, wd, model_sr, slm_sr=16000):
        super().__init__()
        self.wavlm = AutoModel.from_pretrained(model)
        self.wd = wd
        self.resample = torchaudio.transforms.Resample(model_sr, slm_sr)
        self.wavlm.eval()
        for param in self.wavlm.parameters():
            param.requires_grad = False

    def encode(self, wav: torch.Tensor, requires_grad: bool = True):
        """音声（16kHz へリサンプル済みでなくとも可。内部で resample）から WavLM の hidden states を返す。

        Args:
            wav (torch.Tensor): (batch, time) の波形
            requires_grad (bool): False の場合は no_grad で計算する（実音声・判別器用の埋め込み）
        """
        if requires_grad:
            wav_16 = self.resample(wav)
            return self.wavlm(
                input_values=wav_16, output_hidden_states=True
            ).hidden_states
        with torch.no_grad():
            wav_16 = self.resample(wav)
            return self.wavlm(
                input_values=wav_16, output_hidden_states=True
            ).hidden_states

    @staticmethod
    def _stack(embeddings) -> torch.Tensor:
        """hidden states のリストを判別器へ渡す形式へ積む。"""
        return (
            torch.stack(embeddings, dim=1)
            .transpose(-1, -2)
            .flatten(start_dim=1, end_dim=2)
        )

    def forward(self, wav_embeddings, y_rec_embeddings):
        """feature matching 損失（loss_lm）。実音声側は no_grad、生成音声側は勾配ありの埋め込みを渡す。"""
        floss = 0
        for er, eg in zip(wav_embeddings, y_rec_embeddings):
            floss += torch.mean(torch.abs(er - eg))

        return floss.mean()

    def generator(self, y_rec_embeddings):
        """生成器側の敵対損失（loss_lm_gen）。勾配ありの生成音声埋め込みを渡す。"""
        y_rec_embeddings = self._stack(y_rec_embeddings)
        y_df_hat_g = self.wd(y_rec_embeddings)
        loss_gen = torch.mean((1 - y_df_hat_g) ** 2)

        return loss_gen

    def discriminator(self, wav_embeddings, y_rec_embeddings):
        """判別器側の損失。両方とも no_grad の埋め込みを渡す（wd のパラメータのみ更新される）。"""
        y_embeddings = self._stack(wav_embeddings)
        y_rec_embeddings = self._stack(y_rec_embeddings)

        y_d_rs = self.wd(y_embeddings)
        y_d_gs = self.wd(y_rec_embeddings)

        r_loss = torch.mean((1 - y_d_rs) ** 2)
        g_loss = torch.mean((y_d_gs) ** 2)

        loss_disc_f = r_loss + g_loss

        return loss_disc_f.mean()

    def discriminator_forward(self, wav):
        with torch.no_grad():
            y_embeddings = self._stack(self.encode(wav, requires_grad=False))

        y_d_rs = self.wd(y_embeddings)

        return y_d_rs
