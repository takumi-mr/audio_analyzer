#!/usr/bin/env python
"""
ベンチマーク用テスト音源合成スクリプト
理論的音楽プロパティ (BPM, Key, Chord, Chorus) を持つマルチトラック音源 (44.1kHz 16-bit WAV) を合成・生成します。
"""

import os

import numpy as np
import soundfile as sf

SR = 44100


def note_to_freq(note_name: str) -> float:
    """音名 (例: 'C4', 'A#3', 'Eb2') を周波数 (Hz) に変換"""
    pitch_map = {
        "C": 0,
        "C#": 1,
        "DB": 1,
        "D": 2,
        "D#": 3,
        "EB": 3,
        "E": 4,
        "F": 5,
        "F#": 6,
        "GB": 6,
        "G": 7,
        "G#": 8,
        "AB": 8,
        "A": 9,
        "A#": 10,
        "BB": 10,
        "B": 11,
    }
    name = note_name.strip().upper()
    octave = int(name[-1])
    p_name = name[:-1]
    semitone = pitch_map[p_name]
    # A4 = 440Hz, MIDI 69
    midi = 12 + octave * 12 + semitone
    return 440.0 * (2.0 ** ((midi - 69) / 12.0))


def synth_kick(sr: int, duration: float = 0.4) -> np.ndarray:
    """バスドラム (ピッチドロップ正弦波 + アタッククリック)"""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # 160Hz -> 45Hz sweep
    f_env = 45.0 + 115.0 * np.exp(-t * 28.0)
    phase = 2 * np.pi * np.cumsum(f_env) / sr
    amp_env = np.exp(-t * 12.0)
    click = np.random.normal(0, 0.2, len(t)) * np.exp(-t * 100.0)
    kick = (np.sin(phase) + click) * amp_env
    return kick


def synth_snare(sr: int, duration: float = 0.35) -> np.ndarray:
    """スネアドラム (ノイズ成分 + 180Hz トーン)"""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    tone = np.sin(2 * np.pi * 185.0 * t) * np.exp(-t * 18.0)
    noise = np.random.normal(0, 0.4, len(t)) * np.exp(-t * 12.0)
    return (0.4 * tone + 0.6 * noise) * 0.9


def synth_hihat(sr: int, duration: float = 0.1) -> np.ndarray:
    """ハイハット (高域ノイズバースト)"""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    noise = np.random.normal(0, 0.3, len(t))
    # 簡易高域強調 (微分)
    hp = np.diff(noise, prepend=0)
    return hp * np.exp(-t * 40.0) * 0.5


def synth_bass_note(sr: int, freq: float, duration: float) -> np.ndarray:
    """ベース音 (基音 + 2倍音 + 3倍音のアナログシンセベース)"""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    sig = (
        0.7 * np.sin(2 * np.pi * freq * t)
        + 0.3 * np.sin(2 * np.pi * 2 * freq * t)
        + 0.15 * np.sin(2 * np.pi * 3 * freq * t)
    )
    # ADSRエンベロープ
    attack = int(sr * 0.015)
    release = int(sr * 0.04)
    env = np.ones(len(t))
    if len(t) > attack + release:
        env[:attack] = np.linspace(0, 1, attack)
        env[-release:] = np.linspace(1, 0, release)
    return sig * env * 0.6


def synth_chord_pad(sr: int, chord_notes: list[str], duration: float) -> np.ndarray:
    """コード伴奏音 (豊かな倍音を持つピアノ/パッド和音)"""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    total_sig = np.zeros_like(t)
    for note in chord_notes:
        f = note_to_freq(note)
        # 基音 + 第2倍音 + 第3倍音 + 第4倍音
        tone = (
            0.50 * np.sin(2 * np.pi * f * t)
            + 0.25 * np.sin(2 * np.pi * 2 * f * t)
            + 0.15 * np.sin(2 * np.pi * 3 * f * t)
            + 0.08 * np.sin(2 * np.pi * 4 * f * t)
        )
        total_sig += tone
    # ADSR
    attack = int(sr * 0.03)
    release = int(sr * 0.06)
    env = np.ones(len(t))
    if len(t) > attack + release:
        env[:attack] = np.linspace(0, 1, attack)
        env[-release:] = np.linspace(1, 0, release)
    return (total_sig / max(1, len(chord_notes))) * env * 0.55


def synth_lead_vocal(
    sr: int, notes: list[tuple[str, float, float]], total_dur: float
) -> np.ndarray:
    """リードボーカル・メロディ音 (ビブラート + フォルマント)"""
    total_samples = int(sr * total_dur)
    vocal = np.zeros(total_samples, dtype=np.float32)
    for note, start_t, dur in notes:
        f0 = note_to_freq(note)
        n_samples = int(sr * dur)
        start_idx = int(sr * start_t)
        end_idx = min(start_idx + n_samples, total_samples)
        actual_len = end_idx - start_idx
        if actual_len <= 0:
            continue
        t = np.linspace(0, actual_len / sr, actual_len, endpoint=False)
        # 5Hz ビブラート
        vibrato = 1.0 + 0.008 * np.sin(2 * np.pi * 5.2 * t)
        inst_freq = f0 * vibrato
        phase = 2 * np.pi * np.cumsum(inst_freq) / sr
        # フォルマント倍音
        tone = 0.6 * np.sin(phase) + 0.3 * np.sin(2 * phase) + 0.15 * np.sin(3 * phase)
        attack = int(sr * 0.02)
        release = int(sr * 0.03)
        env = np.ones(actual_len)
        if actual_len > attack + release:
            env[:attack] = np.linspace(0, 1, attack)
            env[-release:] = np.linspace(1, 0, release)
        vocal[start_idx:end_idx] += (tone * env * 0.45).astype(np.float32)
    return vocal


def generate_track_acoustic_ballad(out_dir: str, mix_out_path: str):
    """
    Test Track 1: Acoustic Pop Ballad
    - BPM: 72.0
    - Key: C Major
    - Chords: C -> G -> Am -> F -> C
    - Chorus: 6.67s - 13.33s
    """
    bpm = 72.0
    beat_sec = 60.0 / bpm
    bars = 5
    beats_per_bar = 4
    total_beats = bars * beats_per_bar
    total_dur = total_beats * beat_sec
    total_samples = int(SR * total_dur)

    drums = np.zeros(total_samples, dtype=np.float32)
    bass = np.zeros(total_samples, dtype=np.float32)
    other = np.zeros(total_samples, dtype=np.float32)
    vocals = np.zeros(total_samples, dtype=np.float32)

    # 1. ドラム合成 (サビ 6.67〜13.33s にフルドラム、Aメロはソフトドラム)
    kick = synth_kick(SR)
    snare = synth_snare(SR)
    hihat = synth_hihat(SR)

    for b in range(total_beats):
        b_time = b * beat_sec
        b_idx = int(SR * b_time)
        _bar_idx = b // beats_per_bar
        beat_in_bar = b % beats_per_bar

        # サビ (bar 2, 3: beats 8..15)
        is_chorus = 8 <= b < 16

        # Hi-hat on every beat
        h_len = min(len(hihat), total_samples - b_idx)
        if h_len > 0:
            drums[b_idx : b_idx + h_len] += hihat[:h_len] * (0.8 if is_chorus else 0.4)

        # 8分音符ハイハット
        half_idx = int(b_idx + SR * (beat_sec / 2))
        if half_idx + len(hihat) <= total_samples:
            drums[half_idx : half_idx + len(hihat)] += hihat * (
                0.6 if is_chorus else 0.3
            )

        if is_chorus:
            # Kick on 1 and 3
            if beat_in_bar in [0, 2]:
                k_len = min(len(kick), total_samples - b_idx)
                if k_len > 0:
                    drums[b_idx : b_idx + k_len] += kick[:k_len] * 0.9
            # Snare on 2 and 4
            if beat_in_bar in [1, 3]:
                s_len = min(len(snare), total_samples - b_idx)
                if s_len > 0:
                    drums[b_idx : b_idx + s_len] += snare[:s_len] * 0.85
        else:
            # Verse: soft kick on beat 1, soft rim on beat 3
            if beat_in_bar == 0:
                k_len = min(len(kick), total_samples - b_idx)
                if k_len > 0:
                    drums[b_idx : b_idx + k_len] += kick[:k_len] * 0.5
            if beat_in_bar == 2:
                s_len = min(len(snare), total_samples - b_idx)
                if s_len > 0:
                    drums[b_idx : b_idx + s_len] += snare[:s_len] * 0.4

    # 2. コードとベース合成
    chord_progression = [
        ("C", "C2", ["C4", "E4", "G4"]),  # bar 0: C
        ("G", "G1", ["G3", "B3", "D4"]),  # bar 1: G
        ("Am", "A1", ["A3", "C4", "E4"]),  # bar 2: Am (Chorus Start)
        ("F", "F1", ["F3", "A3", "C4"]),  # bar 3: F
        ("C", "C2", ["C4", "E4", "G4"]),  # bar 4: C (Outro)
    ]

    for bar, (_chord_name, bass_note, chord_notes) in enumerate(chord_progression):
        bar_start_t = bar * beats_per_bar * beat_sec
        bar_dur = beats_per_bar * beat_sec

        # Other (Chord Pad): 1小節ずつコード演奏
        c_sig = synth_chord_pad(SR, chord_notes, bar_dur)
        c_idx = int(SR * bar_start_t)
        c_len = min(len(c_sig), total_samples - c_idx)
        if c_len > 0:
            other[c_idx : c_idx + c_len] += c_sig[:c_len]

        # Bass: 2拍ごとにルート音を打鍵
        f_bass = note_to_freq(bass_note)
        for sub_beat in range(0, beats_per_bar, 2):
            b_start_t = bar_start_t + sub_beat * beat_sec
            b_sig = synth_bass_note(SR, f_bass, 1.8 * beat_sec)
            b_idx = int(SR * b_start_t)
            b_len = min(len(b_sig), total_samples - b_idx)
            if b_len > 0:
                bass[b_idx : b_idx + b_len] += b_sig[:b_len]

    # 3. ボーカル合成 (Chorus: 6.67〜13.33s に感情豊かなサビメロディ)
    vocal_notes = [
        # Verse (Bar 1: 3.33〜6.67s) - ささやかなフレーズ
        ("E4", 3.8, 0.7),
        ("D4", 4.6, 0.6),
        ("C4", 5.4, 1.0),
        # Chorus (Bar 2, 3: 6.67〜13.33s) - 高音サビフック
        ("A4", 6.8, 0.9),
        ("G4", 7.8, 0.8),
        ("E4", 8.8, 1.2),
        ("C5", 10.2, 1.0),
        ("A4", 11.4, 0.9),
        ("G4", 12.4, 0.8),
    ]
    vocals = synth_lead_vocal(SR, vocal_notes, total_dur)

    # ミックス
    mixture = drums * 0.7 + bass * 0.9 + other * 0.85 + vocals * 0.85
    peak = max(np.max(np.abs(mixture)), 1e-6)
    if peak > 0.95:
        scale = 0.95 / peak
        mixture *= scale
        drums *= scale
        bass *= scale
        other *= scale
        vocals *= scale

    # 書き出し
    os.makedirs(out_dir, exist_ok=True)
    sf.write(os.path.join(out_dir, "mixture.wav"), mixture, SR, subtype="PCM_16")
    sf.write(os.path.join(out_dir, "drums.wav"), drums, SR, subtype="PCM_16")
    sf.write(os.path.join(out_dir, "bass.wav"), bass, SR, subtype="PCM_16")
    sf.write(os.path.join(out_dir, "other.wav"), other, SR, subtype="PCM_16")
    sf.write(os.path.join(out_dir, "vocals.wav"), vocals, SR, subtype="PCM_16")

    # ルートまたは指定ミックス出力
    sf.write(mix_out_path, mixture, SR, subtype="PCM_16")
    print(f"[Generated] Acoustic Pop Ballad -> {mix_out_path}")


def generate_track_city_funk(out_dir: str, mix_out_path: str):
    """
    Test Track 2: Neo-Soul City Funk
    - BPM: 116.0
    - Key: A Minor
    - Chords: Am7 -> Dm7 -> G7 -> Cmaj7 -> Fmaj7 -> E7
    - Chorus: 4.14s - 10.34s
    """
    bpm = 116.0
    beat_sec = 60.0 / bpm
    bars = 6
    beats_per_bar = 4
    total_beats = bars * beats_per_bar
    total_dur = total_beats * beat_sec
    total_samples = int(SR * total_dur)

    drums = np.zeros(total_samples, dtype=np.float32)
    bass = np.zeros(total_samples, dtype=np.float32)
    other = np.zeros(total_samples, dtype=np.float32)
    vocals = np.zeros(total_samples, dtype=np.float32)

    kick = synth_kick(SR, duration=0.3)
    snare = synth_snare(SR, duration=0.25)
    hihat = synth_hihat(SR, duration=0.08)

    # 1. タイトなファンクビート
    for b in range(total_beats):
        b_time = b * beat_sec
        b_idx = int(SR * b_time)
        beat_in_bar = b % beats_per_bar
        is_chorus = 8 <= b < 20

        # 16分音符ハイハット
        for sixteenth in range(4):
            t_six = b_idx + int(SR * (sixteenth * beat_sec / 4))
            if t_six + len(hihat) <= total_samples:
                drums[t_six : t_six + len(hihat)] += hihat * (
                    0.6 if sixteenth % 2 == 0 else 0.35
                )

        # Kick on 1 and 3 (with syncopation on 2.5 in chorus)
        if beat_in_bar == 0:
            k_len = min(len(kick), total_samples - b_idx)
            if k_len > 0:
                drums[b_idx : b_idx + k_len] += kick[:k_len] * 0.95
        elif beat_in_bar == 2:
            k_len = min(len(kick), total_samples - b_idx)
            if k_len > 0:
                drums[b_idx : b_idx + k_len] += kick[:k_len] * 0.85

        if is_chorus and beat_in_bar == 1:
            t_sync = b_idx + int(SR * (beat_sec * 0.75))
            if t_sync + len(kick) <= total_samples:
                drums[t_sync : t_sync + len(kick)] += kick * 0.7

        # Snare on 2 and 4
        if beat_in_bar in [1, 3]:
            s_len = min(len(snare), total_samples - b_idx)
            if s_len > 0:
                drums[b_idx : b_idx + s_len] += snare[:s_len] * 0.9

    # 2. セブンスコード進行 (Rhodes / E-Piano + Funk Bass)
    chord_progression = [
        ("Am7", "A1", ["A3", "C4", "E4", "G4"]),  # bar 0: Am7
        ("Dm7", "D2", ["D3", "F3", "A3", "C4"]),  # bar 1: Dm7
        ("G7", "G1", ["G3", "B3", "D4", "F4"]),  # bar 2: G7 (Chorus Start)
        ("Cmaj7", "C2", ["C4", "E4", "G4", "B4"]),  # bar 3: Cmaj7
        ("Fmaj7", "F1", ["F3", "A3", "C4", "E4"]),  # bar 4: Fmaj7
        ("E7", "E1", ["E3", "G#3", "B3", "D4"]),  # bar 5: E7 (Turnaround)
    ]

    for bar, (_chord_name, bass_note, chord_notes) in enumerate(chord_progression):
        bar_start_t = bar * beats_per_bar * beat_sec

        # 2拍ごとにカッティング演奏 (ファンクコード)
        for half in range(2):
            c_start = bar_start_t + half * 2 * beat_sec
            c_sig = synth_chord_pad(SR, chord_notes, 1.9 * beat_sec)
            c_idx = int(SR * c_start)
            c_len = min(len(c_sig), total_samples - c_idx)
            if c_len > 0:
                other[c_idx : c_idx + c_len] += c_sig[:c_len]

        # スラップ・ファンクベース (ルート打鍵 + オクターブ)
        f_bass = note_to_freq(bass_note)
        for beat in range(beats_per_bar):
            b_start = bar_start_t + beat * beat_sec
            f_use = f_bass * (2.0 if beat in [1, 3] else 1.0)
            b_sig = synth_bass_note(SR, f_use, 0.85 * beat_sec)
            b_idx = int(SR * b_start)
            b_len = min(len(b_sig), total_samples - b_idx)
            if b_len > 0:
                bass[b_idx : b_idx + b_len] += b_sig[:b_len]

    # 3. ブラス/ボーカルリード (Chorus: 4.14〜10.34s)
    vocal_notes = [
        # Verse (Bar 1: 2.07〜4.14s)
        ("A4", 2.2, 0.4),
        ("C5", 2.8, 0.4),
        ("A4", 3.4, 0.5),
        # Chorus (Bar 2..4: 4.14〜10.34s)
        ("B4", 4.3, 0.6),
        ("D5", 5.0, 0.7),
        ("E5", 6.3, 0.8),
        ("G4", 7.2, 0.5),
        ("B4", 7.8, 0.6),
        ("A4", 8.4, 0.7),
        ("C5", 9.2, 0.8),
    ]
    vocals = synth_lead_vocal(SR, vocal_notes, total_dur)

    # ミックス
    mixture = drums * 0.75 + bass * 0.85 + other * 0.8 + vocals * 0.8
    peak = max(np.max(np.abs(mixture)), 1e-6)
    if peak > 0.95:
        scale = 0.95 / peak
        mixture *= scale
        drums *= scale
        bass *= scale
        other *= scale
        vocals *= scale

    # 書き出し
    os.makedirs(out_dir, exist_ok=True)
    sf.write(os.path.join(out_dir, "mixture.wav"), mixture, SR, subtype="PCM_16")
    sf.write(os.path.join(out_dir, "drums.wav"), drums, SR, subtype="PCM_16")
    sf.write(os.path.join(out_dir, "bass.wav"), bass, SR, subtype="PCM_16")
    sf.write(os.path.join(out_dir, "other.wav"), other, SR, subtype="PCM_16")
    sf.write(os.path.join(out_dir, "vocals.wav"), vocals, SR, subtype="PCM_16")

    sf.write(mix_out_path, mixture, SR, subtype="PCM_16")
    print(f"[Generated] Neo-Soul City Funk -> {mix_out_path}")


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_dir)
    musdb_test_dir = os.path.join(base_dir, "musdb18", "test")

    # Track 1: Acoustic Pop Ballad
    t1_dir = os.path.join(musdb_test_dir, "Acoustic Pop Ballad")
    t1_mix = os.path.join(project_root, "acoustic_ballad.wav")
    generate_track_acoustic_ballad(t1_dir, t1_mix)

    # Track 2: Neo-Soul City Funk
    t2_dir = os.path.join(musdb_test_dir, "Neo-Soul City Funk")
    t2_mix = os.path.join(project_root, "city_funk.wav")
    generate_track_city_funk(t2_dir, t2_mix)


if __name__ == "__main__":
    main()
