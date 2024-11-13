import torch


def compute_metric(metric, s1_audio, s2_audio, speaker_1, speaker_2, mix_audio):
    m_s1_s1 = metric(s1_audio, speaker_1)
    m_s1_s2 = metric(s1_audio, speaker_2)
    m_s2_s1 = metric(s2_audio, speaker_1)
    m_s2_s2 = metric(s2_audio, speaker_2)

    m_mix_s1 = metric(s1_audio, mix_audio)
    m_mix_s2 = metric(s2_audio, mix_audio)

    mi_s1 = m_s1_s1 - m_mix_s1
    mi_s2 = m_s2_s2 - m_mix_s2
    mean_mi_perm1 = (mi_s1 + mi_s2) / 2

    mi_perm2_s1 = m_s1_s2 - m_mix_s1
    mi_perm2_s2 = m_s2_s1 - m_mix_s2
    mean_mi_perm2 = (mi_perm2_s1 + mi_perm2_s2) / 2

    result_metric = torch.maximum(mean_mi_perm1, mean_mi_perm2)
    return result_metric.mean()
