import torch

def compute_metric(metric, s1_audio, s2_audio, speaker_1, speaker_2, mix_audio):

    result_metrics = []

    if s1_audio is not None:
        min_length = min(
            s1_audio.shape[-1],
            speaker_1.shape[-1],
            speaker_2.shape[-1],
            mix_audio.shape[-1],
        )
        s1_audio = s1_audio[..., :min_length]
        speaker_1_s1 = speaker_1[..., :min_length]
        speaker_2_s1 = speaker_2[..., :min_length]
        mix_audio_s1 = mix_audio[..., :min_length]

        m_s1_s1 = metric(speaker_1_s1, s1_audio)
        m_s2_s1 = metric(speaker_2_s1, s1_audio)
        m_mix_s1 = metric(mix_audio_s1, s1_audio)

        si_snr_i_speaker_1 = torch.max(m_s1_s1 - m_mix_s1, m_s2_s1 - m_mix_s1)
        result_metrics.append(si_snr_i_speaker_1.mean())

    if s2_audio is not None:
        min_length = min(
            s2_audio.shape[-1],
            speaker_1.shape[-1],
            speaker_2.shape[-1],
            mix_audio.shape[-1],
        )
        s2_audio = s2_audio[..., :min_length]
        speaker_1_s2 = speaker_1[..., :min_length]
        speaker_2_s2 = speaker_2[..., :min_length]
        mix_audio_s2 = mix_audio[..., :min_length]

        m_s1_s2 = metric(speaker_1_s2, s2_audio)
        m_s2_s2 = metric(speaker_2_s2, s2_audio)
        m_mix_s2 = metric(mix_audio_s2, s2_audio)

        si_snr_i_speaker_2 = torch.max(m_s1_s2 - m_mix_s2, m_s2_s2 - m_mix_s2)
        result_metrics.append(si_snr_i_speaker_2.mean())

    if result_metrics:
        return torch.stack(result_metrics).mean()
    else:
        return torch.tensor(0.0)
