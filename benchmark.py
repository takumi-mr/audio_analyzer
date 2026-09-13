#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Audio Analyzer Benchmark Tool (MIR & MUSDB18 自動精度検証スクリプト)

音楽情報処理 (MIR: Music Information Retrieval) の世界標準メトリクスに基づき、
楽曲の「コード進行推定」「主キー判定」「テンポ/BPM検出」「サビ検出」および
「MUSDB18 / MUSDB18-HQ による音源分離精度(SDR/相関度)」を
自動で検証・スコアリング・レポート出力するベンチマークツールです。
"""

import argparse
import json
import os
import sys
import glob
from typing import Dict, Any, List, Tuple, Optional
import numpy as np

# コア解析コンポーネント
from analyzer.AudioAnalyzer import AudioAnalyzer
from analyzer.IAnalysisStrategy import IAnalysisStrategy
from analyzer.strategy.SimpleBeatStrategy import SimpleBeatStrategy
from analyzer.strategy.KeyDetectionStrategy import KeyDetectionStrategy
from analyzer.strategy.ChordEstimationStrategy import ChordEstimationStrategy
from analyzer.strategy.ChorusDetectionBeatSSMStrategy import ChorusDetectionBeatSSMStrategy
from reader.LibrosaAudioReader import LibrosaAudioReader
from writer.IResultWriter import IResultWriter
from model.AudioSignal import AudioSignal

from analyzer.filter.IAudioFilter import IAudioFilter
from analyzer.filter.BandSplitterFilter import BandSplitterFilter
from analyzer.filter.CompressorFilter import CompressorFilter
from analyzer.filter.SourceSeparatorFilter import SourceSeparatorFilter


# ==============================================================================
# 音楽理論 & MIRメトリクス計算モジュール
# ==============================================================================

PITCH_MAP = {
    "C": 0, "B#": 0,
    "C#": 1, "DB": 1, "D-": 1,
    "D": 2,
    "D#": 3, "EB": 3, "E-": 3,
    "E": 4, "FB": 4,
    "F": 5, "E#": 5,
    "F#": 6, "GB": 6, "G-": 6,
    "G": 7,
    "G#": 8, "AB": 8, "A-": 8,
    "A": 9,
    "A#": 10, "BB": 10, "B-": 10,
    "B": 11, "CB": 11
}

PITCH_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

def parse_pitch_class(note_str: str) -> Optional[int]:
    """音名文字列を 0〜11 のピッチクラスに変換"""
    clean = note_str.strip().upper()
    return PITCH_MAP.get(clean, None)

def parse_chord(chord_name: str) -> Tuple[Optional[int], str, str]:
    """
    コード名を (ルート音ピッチクラス, トライアド種別, 詳細クオリティ) に分解。
    スラッシュコード (例: C/E, Gm/Bb) にも対応。
    """
    name = chord_name.strip()
    if not name or name.upper() in ["N", "NO CHORD", "NONE"]:
        return None, "none", "none"
        
    slash_bass = None
    if "/" in name:
        parts = name.split("/", 1)
        name = parts[0].strip()
        slash_bass = parse_pitch_class(parts[1].strip())

    root_pitch = None
    root_len = 0
    if len(name) >= 2 and name[:2].upper() in PITCH_MAP:
        root_pitch = PITCH_MAP[name[:2].upper()]
        root_len = 2
    elif len(name) >= 1 and name[:1].upper() in PITCH_MAP:
        root_pitch = PITCH_MAP[name[:1].upper()]
        root_len = 1
        
    if root_pitch is None:
        return None, "unknown", name
        
    raw_suffix = name[root_len:].replace(" ", "")
    
    triad = "maj"
    quality = "maj"
    
    if raw_suffix in ["", "maj", "major"]:
        triad = "maj"
        quality = "maj"
    elif raw_suffix in ["M7", "maj7", "Major7", "maj7(9)"]:
        triad = "maj"
        quality = "maj7"
    elif raw_suffix in ["m", "min", "minor"]:
        triad = "min"
        quality = "min"
    elif raw_suffix in ["m7", "min7", "minor7"]:
        triad = "min"
        quality = "min7"
    elif raw_suffix in ["m7(9)", "min7(9)"]:
        triad = "min"
        quality = "min7(9)"
    elif "m7b5" in raw_suffix.lower() or "half-dim" in raw_suffix.lower():
        triad = "dim"
        quality = "m7b5"
    elif "dim" in raw_suffix.lower():
        triad = "dim"
        quality = "dim7" if "7" in raw_suffix else "dim"
    elif "aug" in raw_suffix.lower() or "+" in raw_suffix:
        triad = "aug"
        quality = "aug"
    elif "sus4" in raw_suffix.lower():
        triad = "sus4"
        quality = "sus4"
    elif "7(9)" in raw_suffix.lower() or "9" in raw_suffix.lower():
        triad = "maj"
        quality = "7(9)"
    elif "7" in raw_suffix and "m" not in raw_suffix:
        triad = "maj"
        quality = "7"
    else:
        triad = "maj"
        quality = raw_suffix
        
    return root_pitch, triad, quality

def evaluate_chords(gt_spans: List[Dict[str, Any]], pred_spans: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    時間重み付きコード一致率 (WCSR: Weighted Chord Symbol Recall) の計算
    """
    if not gt_spans or not pred_spans:
        return {"exact_wcsr": 0.0, "triad_wcsr": 0.0, "root_wcsr": 0.0, "total_duration": 0.0}

    total_gt_duration = 0.0
    exact_matched_duration = 0.0
    triad_matched_duration = 0.0
    root_matched_duration = 0.0

    for gt in gt_spans:
        g_s = float(gt["start_sec"])
        g_e = float(gt["end_sec"])
        dur = max(0.0, g_e - g_s)
        total_gt_duration += dur
        
        gt_root, gt_triad, gt_qual = parse_chord(gt["chord"])

        for pred in pred_spans:
            p_s = float(pred["start_sec"])
            p_e = float(pred["end_sec"])
            
            overlap_s = max(g_s, p_s)
            overlap_e = min(g_e, p_e)
            overlap = max(0.0, overlap_e - overlap_s)
            
            if overlap <= 0.0:
                continue
                
            p_root, p_triad, p_qual = parse_chord(pred["chord"])
            
            if gt_root is not None and p_root is not None:
                if gt_root == p_root:
                    root_matched_duration += overlap
                    if gt_triad == p_triad:
                        triad_matched_duration += overlap
                        if gt_qual == p_qual:
                            exact_matched_duration += overlap

    if total_gt_duration <= 0.0:
        return {"exact_wcsr": 0.0, "triad_wcsr": 0.0, "root_wcsr": 0.0, "total_duration": 0.0}

    return {
        "exact_wcsr": round((exact_matched_duration / total_gt_duration) * 100.0, 2),
        "triad_wcsr": round((triad_matched_duration / total_gt_duration) * 100.0, 2),
        "root_wcsr": round((root_matched_duration / total_gt_duration) * 100.0, 2),
        "total_duration": round(total_gt_duration, 2)
    }

def evaluate_key(gt_key_str: str, pred_key_str: str) -> Dict[str, Any]:
    """
    MIREX 世界標準キー評価スコアリング
    """
    def parse_key(k_str: str) -> Tuple[Optional[int], str]:
        parts = k_str.strip().split()
        if not parts:
            return None, "major"
        pitch = parse_pitch_class(parts[0])
        mode = parts[1].lower() if len(parts) > 1 else "major"
        return pitch, mode

    gt_pitch, gt_mode = parse_key(gt_key_str)
    pred_pitch, pred_mode = parse_key(pred_key_str)

    if gt_pitch is None or pred_pitch is None:
        return {"score": 0.0, "match_type": "None", "exact": False}

    if gt_pitch == pred_pitch and gt_mode == pred_mode:
        return {"score": 1.0, "match_type": "Exact Match", "exact": True}

    if gt_mode == pred_mode:
        if (gt_pitch + 7) % 12 == pred_pitch or (gt_pitch + 5) % 12 == pred_pitch:
            return {"score": 0.5, "match_type": "Fifth Match", "exact": False}

    if gt_mode == "major" and pred_mode == "minor":
        if (gt_pitch - 3) % 12 == pred_pitch:
            return {"score": 0.3, "match_type": "Relative Match", "exact": False}
    elif gt_mode == "minor" and pred_mode == "major":
        if (gt_pitch + 3) % 12 == pred_pitch:
            return {"score": 0.3, "match_type": "Relative Match", "exact": False}

    if gt_pitch == pred_pitch and gt_mode != pred_mode:
        return {"score": 0.2, "match_type": "Parallel Match", "exact": False}

    return {"score": 0.0, "match_type": "Mismatch", "exact": False}

def evaluate_tempo(gt_bpm: float, pred_bpm: float) -> Dict[str, Any]:
    """
    テンポ / BPM 検出精度 (P-Score: ±4%, ±8%, 倍テン/半テン許容)
    """
    if gt_bpm <= 0.0 or pred_bpm <= 0.0:
        return {"p_score_4": False, "p_score_8": False, "octave_match": False, "relative_error": 1.0}

    err = abs(pred_bpm - gt_bpm) / gt_bpm
    p4 = err <= 0.04
    p8 = err <= 0.08

    err_double = abs(pred_bpm - 2.0 * gt_bpm) / (2.0 * gt_bpm)
    err_half = abs(pred_bpm - 0.5 * gt_bpm) / (0.5 * gt_bpm)
    octave_match = p8 or (err_double <= 0.08) or (err_half <= 0.08)

    return {
        "p_score_4": p4,
        "p_score_8": p8,
        "octave_match": octave_match,
        "relative_error": round(err * 100.0, 2)
    }

def evaluate_chorus(gt_sections: List[Dict[str, Any]], pred_sections: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    サビ検出の Overlap 評価 (Precision, Recall, F1-measure)
    """
    if not gt_sections or not pred_sections:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "overlap_sec": 0.0}

    total_gt_dur = sum(max(0.0, float(g["end_sec"]) - float(g["start_sec"])) for g in gt_sections)
    total_pred_dur = sum(max(0.0, float(p["end_sec"]) - float(p["start_sec"])) for p in pred_sections)

    total_overlap = 0.0
    for g in gt_sections:
        gs = float(g["start_sec"])
        ge = float(g["end_sec"])
        for p in pred_sections:
            ps = float(p["start_sec"])
            pe = float(p["end_sec"])
            overlap = max(0.0, min(ge, pe) - max(gs, ps))
            total_overlap += overlap

    prec = (total_overlap / total_pred_dur) if total_pred_dur > 0 else 0.0
    rec = (total_overlap / total_gt_dur) if total_gt_dur > 0 else 0.0
    f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

    return {
        "precision": round(prec * 100.0, 2),
        "recall": round(rec * 100.0, 2),
        "f1": round(f1 * 100.0, 2),
        "overlap_sec": round(total_overlap, 2)
    }


# ==============================================================================
# MUSDB18 音源分離評価モジュール
# ==============================================================================

def evaluate_source_separation(
    pred_signals: Dict[str, AudioSignal],
    gt_paths: Dict[str, str],
    reader: LibrosaAudioReader
) -> Dict[str, Dict[str, float]]:
    """
    Demucs音源分離の推定ステムと正解ステムの波形相関度(r)およびSDR(dB)を算出
    """
    results: Dict[str, Dict[str, float]] = {}
    
    stem_map = {
        "vocals": "target_vocal",
        "drums": "target_drums",
        "bass": "target_bass",
        "other": "target_other"
    }

    for stem_name, pred_key in stem_map.items():
        if stem_name not in gt_paths or pred_key not in pred_signals:
            continue
            
        gt_path = gt_paths[stem_name]
        if not os.path.exists(gt_path):
            continue
            
        gt_sig = reader.read(gt_path)
        pred_sig = pred_signals[pred_key]

        # 波形長のアラインメント
        min_len = min(len(gt_sig.data), len(pred_sig.data))
        if min_len == 0:
            continue
            
        s = gt_sig.data[:min_len]
        s_hat = pred_sig.data[:min_len]

        # 1. Pearson相関係数 r
        s_std = np.std(s)
        s_hat_std = np.std(s_hat)
        if s_std > 1e-6 and s_hat_std > 1e-6:
            r = float(np.corrcoef(s, s_hat)[0, 1])
        else:
            r = 1.0 if s_std <= 1e-6 and s_hat_std <= 1e-6 else 0.0
            
        # 2. SDR (Signal to Distortion Ratio, dB)
        error = s - s_hat
        s_energy = float(np.sum(s ** 2))
        err_energy = float(np.sum(error ** 2))
        sdr = 10.0 * np.log10(s_energy / (err_energy + 1e-8)) if s_energy > 0 else 0.0

        results[stem_name] = {
            "correlation_r": round(max(0.0, r), 3),
            "correlation_pct": round(max(0.0, r) * 100.0, 1),
            "sdr_db": round(sdr, 2)
        }

    return results

def scan_musdb_tracks(musdb_root: str, max_tracks: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    MUSDB18 / MUSDB18-HQ のディレクトリ構造をスキャンしてトラック情報を取得
    """
    tracks = []
    if not os.path.exists(musdb_root):
        return tracks

    # 再帰的に mixture.wav または mixture.flac を探す
    candidates = []
    for ext in [".wav", ".flac", ".mp3"]:
        candidates.extend(glob.glob(os.path.join(musdb_root, "**", f"mixture{ext}"), recursive=True))

    for mix_path in sorted(candidates):
        track_dir = os.path.dirname(mix_path)
        title = os.path.basename(track_dir)
        
        # 各ステムのパス探索
        stems = {"mixture": mix_path}
        for stem_name in ["vocals", "drums", "bass", "other"]:
            for ext in [".wav", ".flac", ".mp3"]:
                stem_file = os.path.join(track_dir, f"{stem_name}{ext}")
                if os.path.exists(stem_file):
                    stems[stem_name] = stem_file
                    break

        tracks.append({
            "title": title,
            "audio": mix_path,
            "stems": stems
        })
        
        if max_tracks and len(tracks) >= max_tracks:
            break

    return tracks


# ==============================================================================
# メモリ内実行用ライター
# ==============================================================================
class InMemoryWriter(IResultWriter):
    def __init__(self):
        self.result: Dict[str, Any] = {}
    def write(self, filepath: str, result: Dict[str, Any]) -> None:
        self.result.update(result)


# ==============================================================================
# ベンチマーク実行エンジン
# ==============================================================================

def run_benchmark(
    dataset_path: Optional[str] = None,
    musdb_root: Optional[str] = None,
    strategies_to_eval: Optional[List[str]] = None,
    chord_engine: str = "hybrid",
    no_separation: bool = False,
    musdb_separation: bool = False,
    max_tracks: Optional[int] = None,
    verbose: bool = False
) -> Dict[str, Any]:
    
    strategies_to_eval = strategies_to_eval or ["chord", "key", "beat", "chorus"]

    # トラックアイテムのロード
    data_items = []
    is_musdb_mode = False

    if musdb_root:
        is_musdb_mode = True
        data_items = scan_musdb_tracks(musdb_root, max_tracks=max_tracks)
        if not data_items:
            raise FileNotFoundError(f"No MUSDB18 tracks (mixture.wav) found in: {musdb_root}")
    elif dataset_path:
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
        with open(dataset_path, "r", encoding="utf-8") as f:
            data_items = json.load(f)
        if not isinstance(data_items, list):
            data_items = [data_items]
        if max_tracks:
            data_items = data_items[:max_tracks]

    reader = LibrosaAudioReader()
    mem_writer = InMemoryWriter()

    # フィルターチェーン
    filters: List[IAudioFilter] = []
    sep_filter: Optional[SourceSeparatorFilter] = None
    if not no_separation:
        sep_filter = SourceSeparatorFilter(target_key="target")
        filters.append(sep_filter)
    filters.append(BandSplitterFilter(target_key="target_drums", cutoff_hz=150.0))
    filters.append(CompressorFilter(target_key="target_drums_low", threshold=0.2, ratio=3.0))

    strategy_instances: List[IAnalysisStrategy] = []
    if "key" in strategies_to_eval:
        strategy_instances.append(KeyDetectionStrategy())
    if "beat" in strategies_to_eval:
        strategy_instances.append(SimpleBeatStrategy())
    if "chord" in strategies_to_eval:
        strategy_instances.append(ChordEstimationStrategy(engine=chord_engine))
    if "chorus" in strategies_to_eval:
        strategy_instances.append(ChorusDetectionBeatSSMStrategy())

    analyzer = AudioAnalyzer(reader, mem_writer, SimpleBeatStrategy(), filters=filters)

    all_results = []
    print(f"\n=======================================================")
    print(f" 音声解析自動ベンチマーク実行開始 (全 {len(data_items)} 曲)")
    print(f" モード: {'MUSDB18 Dataset' if is_musdb_mode else 'Custom Annotations'}")
    print(f" 評価項目: {strategies_to_eval} | 音源分離: {'OFF' if no_separation else 'ON'}")
    if musdb_separation:
        print(f" 音源分離SDR評価: ON")
    print(f"=======================================================\n")

    for idx, item in enumerate(data_items, 1):
        title = item.get("title", f"Track {idx}")
        audio_path = item.get("audio")

        if dataset_path and not os.path.isabs(audio_path) and not os.path.exists(audio_path):
            dataset_dir = os.path.dirname(os.path.abspath(dataset_path))
            candidate = os.path.join(dataset_dir, audio_path)
            if os.path.exists(candidate):
                audio_path = candidate

        print(f"[{idx}/{len(data_items)}] 解析中: {title} ({audio_path})...")

        if not os.path.exists(audio_path):
            print(f"  [Error] 音声ファイルが見つかりません: {audio_path}", file=sys.stderr)
            continue

        # 解析実行
        mem_writer.result = {}
        try:
            # 音源分離ステムを後で評価するために保存するフック
            signals = {"target": reader.read(audio_path)}
            for f in filters:
                signals = f.apply(signals)
            
            merged_result = {"status": "success"}
            for s in strategy_instances:
                r = s.analyze(signals, params=merged_result)
                for k, v in r.items():
                    if k != "status":
                        merged_result[k] = v
            mem_writer.result = merged_result
        except Exception as e:
            print(f"  [Error] 解析失敗: {e}", file=sys.stderr)
            continue

        pred = mem_writer.result
        item_scores: Dict[str, Any] = {"title": title, "audio": audio_path}

        # 1. コード進行評価
        if "chord" in strategies_to_eval and "chords" in item:
            c_res = evaluate_chords(item["chords"], pred.get("chords", []))
            item_scores["chord"] = c_res
            print(f"  - [Chord] Exact WCSR: {c_res['exact_wcsr']}% | Triad: {c_res['triad_wcsr']}% | Root: {c_res['root_wcsr']}%")

        # 2. キー評価
        if "key" in strategies_to_eval and "key" in item:
            k_res = evaluate_key(item["key"], pred.get("estimated_key", ""))
            item_scores["key"] = {
                "ground_truth": item["key"],
                "prediction": pred.get("estimated_key", ""),
                **k_res
            }
            print(f"  - [Key] 正解: '{item['key']}' vs 推定: '{pred.get('estimated_key')}' -> {k_res['match_type']} (Score: {k_res['score']})")

        # 3. テンポ評価
        if "beat" in strategies_to_eval and "tempo_bpm" in item:
            t_res = evaluate_tempo(float(item["tempo_bpm"]), float(pred.get("tempo_bpm", 0.0)))
            item_scores["tempo"] = {
                "ground_truth": item["tempo_bpm"],
                "prediction": pred.get("tempo_bpm", 0.0),
                **t_res
            }
            print(f"  - [Tempo] 正解: {item['tempo_bpm']} vs 推定: {pred.get('tempo_bpm')} (誤差: {t_res['relative_error']}%, P-score±8%: {t_res['p_score_8']})")

        # 4. サビ評価
        if "chorus" in strategies_to_eval and "chorus" in item:
            pred_chorus = pred.get("chorus_sections_beat_ssm", pred.get("chorus_sections", []))
            ch_res = evaluate_chorus(item["chorus"], pred_chorus)
            item_scores["chorus"] = ch_res
            print(f"  - [Chorus] Precision: {ch_res['precision']}% | Recall: {ch_res['recall']}% | F1: {ch_res['f1']}%")

        # 5. MUSDB18 音源分離評価
        if musdb_separation and "stems" in item:
            sep_res = evaluate_source_separation(signals, item["stems"], reader)
            item_scores["separation"] = sep_res
            print(f"  - [Separation (Demucs)]")
            for stem_k, s_val in sep_res.items():
                print(f"      {stem_k:<7}: 相関度 {s_val['correlation_pct']:>5.1f}% | SDR: {s_val['sdr_db']:>5.1f} dB")

        all_results.append(item_scores)
        print()

    summary = compute_summary(all_results, strategies_to_eval, musdb_separation)
    summary["chord_engine"] = chord_engine
    return {"summary": summary, "tracks": all_results, "chord_engine": chord_engine}

def compute_summary(tracks: List[Dict[str, Any]], strategies: List[str], musdb_separation: bool = False) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"track_count": len(tracks)}

    if not tracks:
        return summary

    if "chord" in strategies:
        chord_items = [t["chord"] for t in tracks if "chord" in t]
        if chord_items:
            summary["chord"] = {
                "mean_exact_wcsr": round(sum(c["exact_wcsr"] for c in chord_items) / len(chord_items), 2),
                "mean_triad_wcsr": round(sum(c["triad_wcsr"] for c in chord_items) / len(chord_items), 2),
                "mean_root_wcsr": round(sum(c["root_wcsr"] for c in chord_items) / len(chord_items), 2)
            }

    if "key" in strategies:
        key_items = [t["key"] for t in tracks if "key" in t]
        if key_items:
            exact_count = sum(1 for k in key_items if k["exact"])
            mean_score = sum(k["score"] for k in key_items) / len(key_items)
            summary["key"] = {
                "exact_accuracy": round((exact_count / len(key_items)) * 100.0, 2),
                "mirex_weighted_score": round(mean_score, 3)
            }

    if "beat" in strategies:
        tempo_items = [t["tempo"] for t in tracks if "tempo" in t]
        if tempo_items:
            p4_count = sum(1 for t in tempo_items if t["p_score_4"])
            p8_count = sum(1 for t in tempo_items if t["p_score_8"])
            oct_count = sum(1 for t in tempo_items if t["octave_match"])
            summary["tempo"] = {
                "p_score_4_accuracy": round((p4_count / len(tempo_items)) * 100.0, 2),
                "p_score_8_accuracy": round((p8_count / len(tempo_items)) * 100.0, 2),
                "octave_invariant_accuracy": round((oct_count / len(tempo_items)) * 100.0, 2)
            }

    if "chorus" in strategies:
        chorus_items = [t["chorus"] for t in tracks if "chorus" in t]
        if chorus_items:
            summary["chorus"] = {
                "mean_precision": round(sum(c["precision"] for c in chorus_items) / len(chorus_items), 2),
                "mean_recall": round(sum(c["recall"] for c in chorus_items) / len(chorus_items), 2),
                "mean_f1": round(sum(c["f1"] for c in chorus_items) / len(chorus_items), 2)
            }

    if musdb_separation:
        sep_tracks = [t["separation"] for t in tracks if "separation" in t]
        if sep_tracks:
            stem_summary: Dict[str, Any] = {}
            for stem in ["vocals", "drums", "bass", "other"]:
                corr_list = [t[stem]["correlation_pct"] for t in sep_tracks if stem in t]
                sdr_list = [t[stem]["sdr_db"] for t in sep_tracks if stem in t]
                if corr_list and sdr_list:
                    stem_summary[stem] = {
                        "mean_correlation_pct": round(sum(corr_list) / len(corr_list), 1),
                        "mean_sdr_db": round(sum(sdr_list) / len(sdr_list), 2)
                    }
            summary["separation"] = stem_summary

    return summary


# ==============================================================================
# レポート出力モジュール (Console / Markdown)
# ==============================================================================

def print_console_summary(benchmark_data: Dict[str, Any]) -> None:
    summary = benchmark_data["summary"]
    print("=======================================================")
    print(" 総合ベンチマーク結果サマリー")
    print("=======================================================")
    print(f" 評価楽曲数: {summary.get('track_count', 0)} 曲\n")

    if "chord" in summary:
        c = summary["chord"]
        print(f" [コード進行推定 (WCSR)]")
        print(f"   - 完全一致率 (Exact WCSR) : {c['mean_exact_wcsr']:>6.2f} %")
        print(f"   - トライアド一致率 (Triad) : {c['mean_triad_wcsr']:>6.2f} %")
        print(f"   - ルート音一致率 (Root)   : {c['mean_root_wcsr']:>6.2f} %")
        print()

    if "key" in summary:
        k = summary["key"]
        print(f" [主キー判定 (Key Detection)]")
        print(f"   - 完全一致率 (Exact Match): {k['exact_accuracy']:>6.2f} %")
        print(f"   - MIREX 重み付きスコア   : {k['mirex_weighted_score']:>6.3f} / 1.000")
        print()

    if "tempo" in summary:
        t = summary["tempo"]
        print(f" [テンポ・BPM検出 (Beat/Tempo)]")
        print(f"   - P-Score (±4% 許容)      : {t['p_score_4_accuracy']:>6.2f} %")
        print(f"   - P-Score (±8% 許容)      : {t['p_score_8_accuracy']:>6.2f} %")
        print(f"   - 倍テン許容一致率        : {t['octave_invariant_accuracy']:>6.2f} %")
        print()

    if "chorus" in summary:
        ch = summary["chorus"]
        print(f" [サビ自動検出 (Chorus Detection)]")
        print(f"   - 適合率 (Precision)      : {ch['mean_precision']:>6.2f} %")
        print(f"   - 再現率 (Recall)         : {ch['mean_recall']:>6.2f} %")
        print(f"   - F1スコア (F-measure)    : {ch['mean_f1']:>6.2f} %")
        print()

    if "separation" in summary:
        s = summary["separation"]
        print(f" [MUSDB18 音源分離精度 (Demucs Evaluation)]")
        for stem_name, vals in s.items():
            print(f"   - {stem_name.capitalize():<7}: 相関度 {vals['mean_correlation_pct']:>5.1f}% | 平均 SDR: {vals['mean_sdr_db']:>5.2f} dB")
        print()

    print("=======================================================\n")

def generate_markdown_report(benchmark_data: Dict[str, Any], filepath: str) -> None:
    summary = benchmark_data["summary"]
    tracks = benchmark_data["tracks"]

    engine_name = summary.get("chord_engine", "hybrid")
    lines = [
        "# 音声解析エンジン 自動ベンチマークレポート\n",
        f"**評価楽曲総数**: {summary.get('track_count', 0)} 曲 | **コード認識エンジン**: `{engine_name}`\n",
        "## 1. 総合スコアサマリー\n",
        "| 解析項目 | 指標 | スコア | 評価基準 |",
        "| :--- | :--- | :--- | :--- |"
    ]

    if "chord" in summary:
        c = summary["chord"]
        lines.append(f"| **コード進行** | **Exact WCSR** | **{c['mean_exact_wcsr']:.1f}%** | 秒単位の時間重み付き完全一致率 |")
        lines.append(f"| | **Triad Accuracy** | **{c['mean_triad_wcsr']:.1f}%** | Major/Minorトライアド一致率 |")
        lines.append(f"| | **Root Accuracy** | **{c['mean_root_wcsr']:.1f}%** | ルート音（根音）一致率 |")

    if "key" in summary:
        k = summary["key"]
        lines.append(f"| **主キー判定** | **Exact Accuracy** | **{k['exact_accuracy']:.1f}%** | 主音および長調/短調の完全一致率 |")
        lines.append(f"| | **MIREX Score** | **{k['mirex_weighted_score']:.3f}** | 国際コンペMIREX基準重み付きスコア |")

    if "tempo" in summary:
        t = summary["tempo"]
        lines.append(f"| **テンポ / BPM** | **P-Score (±4%)** | **{t['p_score_4_accuracy']:.1f}%** | 誤差4%以内の高精度一致率 |")
        lines.append(f"| | **Octave Invariant** | **{t['octave_invariant_accuracy']:.1f}%** | 倍テン・半テンを許容した調和一致率 |")

    if "chorus" in summary:
        ch = summary["chorus"]
        lines.append(f"| **サビ検出** | **F1 Score** | **{ch['mean_f1']:.1f}%** | サビ区間オーバーラップ調和平均 |")

    if "separation" in summary:
        s = summary["separation"]
        lines.append(f"| **音源分離 (Demucs)** | **Vocals Correlation** | **{s.get('vocals', {}).get('mean_correlation_pct', 0.0)}%** | 正解ボーカルとの波形相関度 |")
        lines.append(f"| | **Drums Correlation** | **{s.get('drums', {}).get('mean_correlation_pct', 0.0)}%** | 正解ドラムとの波形相関度 |")
        lines.append(f"| | **Bass Correlation** | **{s.get('bass', {}).get('mean_correlation_pct', 0.0)}%** | 正解ベースとの波形相関度 |")
        lines.append(f"| | **Other Correlation** | **{s.get('other', {}).get('mean_correlation_pct', 0.0)}%** | 正解伴奏との波形相関度 |")

    lines.append("\n## 2. 楽曲別詳細評価結果\n")
    has_sep = any("separation" in tr for tr in tracks)
    
    if has_sep:
        lines.append("| 楽曲タイトル | 正解キー | 推定キー | 正解BPM | 推定BPM | ボーカル相関 | ドラム相関 | ベース相関 | 伴奏相関 |")
        lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
        for tr in tracks:
            title = tr.get("title", "Unknown")
            k_gt = tr.get("key", {}).get("ground_truth", "-")
            k_pred = tr.get("key", {}).get("prediction", "-")
            t_gt = tr.get("tempo", {}).get("ground_truth", "-")
            t_pred = tr.get("tempo", {}).get("prediction", "-")
            sep = tr.get("separation", {})
            v_corr = f"{sep.get('vocals', {}).get('correlation_pct', '-')}%"
            d_corr = f"{sep.get('drums', {}).get('correlation_pct', '-')}%"
            b_corr = f"{sep.get('bass', {}).get('correlation_pct', '-')}%"
            o_corr = f"{sep.get('other', {}).get('correlation_pct', '-')}%"
            lines.append(f"| {title} | {k_gt} | {k_pred} | {t_gt} | {t_pred} | {v_corr} | {d_corr} | {b_corr} | {o_corr} |")
    else:
        lines.append("| 楽曲タイトル | 正解キー | 推定キー | 正解BPM | 推定BPM | コード一致率(Triad) | サビF1 |")
        lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
        for tr in tracks:
            title = tr.get("title", "Unknown")
            k_gt = tr.get("key", {}).get("ground_truth", "-")
            k_pred = tr.get("key", {}).get("prediction", "-")
            t_gt = tr.get("tempo", {}).get("ground_truth", "-")
            t_pred = tr.get("tempo", {}).get("prediction", "-")
            c_triad = f"{tr.get('chord', {}).get('triad_wcsr', 0.0)}%" if "chord" in tr else "-"
            ch_f1 = f"{tr.get('chorus', {}).get('f1', 0.0)}%" if "chorus" in tr else "-"
            lines.append(f"| {title} | {k_gt} | {k_pred} | {t_gt} | {t_pred} | {c_triad} | {ch_f1} |")

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f">>> ベンチマークレポートを Markdown ファイルに出力しました: {filepath}")


# ==============================================================================
# エントリポイント
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Audio Analyzer 自動ベンチマーク評価スクリプト (MIR & MUSDB18対応)")
    parser.add_argument("-d", "--dataset", default="benchmarks/dataset.json", help="データセット定義JSONパス (デフォルト: benchmarks/dataset.json)")
    parser.add_argument("--musdb", help="MUSDB18 / MUSDB18-HQ 形式のデータセットルートフォルダパス")
    parser.add_argument("--musdb-separation", action="store_true", help="MUSDB18正解ステムとの音源分離精度評価 (SDR / 相関度) を実施")
    parser.add_argument("--max-tracks", type=int, help="評価する最大楽曲数")
    parser.add_argument(
        "-s", "--strategies",
        nargs="+",
        choices=["chord", "key", "beat", "chorus", "all"],
        default=["all"],
        help="評価する解析項目を指定 (デフォルト: all)"
    )
    parser.add_argument(
        "--chord-engine",
        choices=["hybrid", "btc", "heuristic"],
        default="hybrid",
        help="コード認識エンジン (hybrid: BTC Transformer + YIN [推奨], btc: BTC単体, heuristic: ルールベースHMM)"
    )
    parser.add_argument("--no-separation", action="store_true", help="Demucs音源分離をスキップして直接解析 (高速)")
    parser.add_argument("-o", "--output", help="評価レポートファイル出力先 (.md または .json)")
    parser.add_argument("-v", "--verbose", action="store_true", help="詳細ログ出力")

    args = parser.parse_args()

    selected_strategies = args.strategies
    if "all" in selected_strategies:
        selected_strategies = ["chord", "key", "beat", "chorus"]

    # MUSDBモードか通常JSONモードかの判定
    dataset_file = None if args.musdb else args.dataset

    benchmark_data = run_benchmark(
        dataset_path=dataset_file,
        musdb_root=args.musdb,
        strategies_to_eval=selected_strategies,
        chord_engine=args.chord_engine,
        no_separation=args.no_separation,
        musdb_separation=args.musdb_separation,
        max_tracks=args.max_tracks,
        verbose=args.verbose
    )

    print_console_summary(benchmark_data)

    if args.output:
        if args.output.endswith(".json"):
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(benchmark_data, f, indent=2, ensure_ascii=False)
            print(f">>> ベンチマーク結果を JSON ファイルに出力しました: {args.output}")
        else:
            generate_markdown_report(benchmark_data, args.output)

if __name__ == "__main__":
    main()
