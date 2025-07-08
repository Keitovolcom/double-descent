import torch
import numpy as np

def label_aware_mixup(inputs, labels, noise_info, alpha, device):
    """
    バッチ内のノイズサンプルに対して、同じラベルを持つ他のサンプルとMixupを適用します。

    Args:
        inputs (torch.Tensor): 入力データ
        labels (torch.Tensor): ラベルデータ
        noise_info (torch.Tensor): ノイズ情報 (1がノイズあり)
        alpha (float): ベータ分布のαパラメータ。0より大きい場合のみMixupが有効。
        device (torch.device): 使用するデバイス

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: 
        (Mixup後の入力, ラベルa, ラベルb, Mixup係数ラムダのテンソル)
    """
    if alpha <= 0:
        # Mixupが無効な場合は、元のデータをそのまま返す
        return inputs, labels, labels, torch.ones(inputs.size(0), device=device)

    mixed_inputs = inputs.clone()
    labels_a, labels_b = labels.clone(), labels.clone()
    lam_tensor = torch.ones(inputs.size(0), device=device)

    # ノイズありサンプルのインデックスを取得
    noisy_indices = torch.where(noise_info == 1)[0]

    for i in noisy_indices:
        current_label = labels[i]
        
        # 同じラベルを持つ、自分以外のサンプルのインデックスをバッチ内から探す
        same_label_indices = torch.where(labels == current_label)[0]
        partners = same_label_indices[same_label_indices != i]

        if len(partners) > 0:
            # パートナーが見つかった場合
            # ランダムにパートナーを1つ選択
            partner_idx = partners[torch.randint(len(partners), (1,))]

            # ベータ分布からMixupの係数λを生成
            lam = np.random.beta(alpha, alpha)
            lam_tensor[i] = lam

            # Mixupを適用
            mixed_inputs[i] = lam * inputs[i] + (1 - lam) * inputs[partner_idx]
            
            # 損失計算のために、パートナーのラベルを保存（今回は同じラベルなので必須ではないが、汎用的な形式）
            labels_a[i] = labels[i]
            labels_b[i] = labels[partner_idx]
            
    return mixed_inputs, labels_a, labels_b, lam_tensor


def mixup_criterion(criterion, outputs, labels_a, labels_b, lam_tensor):
    """
    Mixupされたバッチに対する損失を計算します。
    loss = λ * loss(y_a) + (1 - λ) * loss(y_b)
    """
    return lam_tensor * criterion(outputs, labels_a) + (1.0 - lam_tensor) * criterion(outputs, labels_b)