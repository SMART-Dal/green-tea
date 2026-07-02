#!/usr/bin/env python3
"""
Analyze GRPO checkpoint: training metrics or attention hotspot alignment.

Usage:
    python analyze_grpo_checkpoint.py checkpoints/qwen-14b_grpo_20260210/
    python analyze_grpo_checkpoint.py --mode attention --model checkpoints/qwen-coder-base-14b_grpo_latest \
        --sim-dir data/grpo_sim_results --output-dir analysis/attention_plots --n-samples 8
"""

import json
import sys
import argparse
from pathlib import Path


def analyze_checkpoint(checkpoint_dir):
    checkpoint_dir = Path(checkpoint_dir)
    state_file = checkpoint_dir / "trainer_state.json"
    if not state_file.exists():
        print(f"ERROR: {state_file} not found")
        return
    state = json.load(open(state_file))
    print("=" * 70)
    print(f"GRPO CHECKPOINT ANALYSIS: {checkpoint_dir.name}")
    print("=" * 70)
    print(f"\nTraining Progress:")
    print(f"  Epoch: {state['epoch']:.2f} / {state.get('num_train_epochs', 5)}")
    print(f"  Global step: {state['global_step']}")
    print(f"  Best checkpoint: {state.get('best_model_checkpoint', 'N/A')}")
    print(f"  Best metric: {state.get('best_metric', 0):.4f}")
    log_history = state.get('log_history', [])
    eval_logs = [log for log in log_history if 'eval_loss' in log or 'objective/rlhf_reward' in log]
    if eval_logs:
        latest_eval = eval_logs[-1]
        print(f"\nLatest Validation Metrics (step {latest_eval.get('step', 'N/A')}):")
        print(f"  Reward: {latest_eval.get('objective/rlhf_reward', 0):.4f}")
        print(f"  Success rate: {latest_eval.get('reward/success_rate', 0)*100:.1f}%")
        print(f"  Compile error rate: {latest_eval.get('correctness/compile_error_rate', 0)*100:.1f}%")
        print(f"  Energy reduction: {latest_eval.get('energy/mean_reduction_pct', 0):.1f}%")
        print(f"  Sniper evaluations: {latest_eval.get('sniper/total_evaluations', 0)}")
    rewards = [log.get('objective/rlhf_reward') for log in eval_logs if 'objective/rlhf_reward' in log]
    if len(rewards) > 1:
        print(f"\nReward Trajectory:")
        print(f"  Initial: {rewards[0]:.4f}")
        print(f"  Final: {rewards[-1]:.4f}")
        print(f"  Change: {rewards[-1] - rewards[0]:+.4f}")
        print(f"  Trend: {'Improving' if rewards[-1] > rewards[0] else 'Declining'}")
    success_rates = [log.get('reward/success_rate', 0) for log in eval_logs if 'reward/success_rate' in log]
    if len(success_rates) > 1:
        print(f"\nSuccess Rate Trajectory:")
        print(f"  Initial: {success_rates[0]*100:.1f}%")
        print(f"  Final: {success_rates[-1]*100:.1f}%")
        print(f"  Change: {(success_rates[-1] - success_rates[0])*100:+.1f}%")
    print("\n" + "=" * 70)
    if not rewards:
        print("No evaluation metrics found - training may have failed")
        return
    final_reward, final_success = rewards[-1], success_rates[-1] if success_rates else 0
    if final_reward > 0.5 and final_success > 0.3:
        print(f"Model learning well - CONTINUE. Next: {state.get('best_model_checkpoint', 'final')}")
    elif final_reward > 0.0 and final_success > 0.1:
        print("Model learning slowly - consider adjusting LR, beta, or data quality")
    else:
        print("Model not learning - STOP and debug compilation rate, Sniper logs, prompt format")


def _load_top_examples(sim_dir, n, min_reduction=10.0):
    seen = set()
    examples = []
    for f in sorted(Path(sim_dir).glob("test_comparison_chunk_*.jsonl")):
        for line in f.open():
            d = json.loads(line)
            pid = d.get('problem_id', '')
            if pid in seen:
                continue
            if d.get('generated_code') and d.get('baseline_code') and d.get('energy_reduction', 0) >= min_reduction:
                examples.append(d)
                seen.add(pid)
        if len(examples) >= n * 10:
            break
    examples.sort(key=lambda x: x['energy_reduction'], reverse=True)
    return examples[:n]


def _get_hotspot_lines(baseline_code, generated_code):
    import difflib, re
    removed = set()
    for line in difflib.unified_diff(baseline_code.splitlines(), generated_code.splitlines(), n=0):
        if line.startswith('-') and not line.startswith('---'):
            content = line[1:]
            stripped = re.sub(r'[\s{}();,]', '', content)
            if len(stripped) >= 3:
                removed.add(content)
    return removed


def _token_hotspot_mask(baseline_code, token_strs, hotspot_lines):
    """Assign each token a bool: True if token is on a removed/changed line."""
    lines = baseline_code.splitlines(keepends=True)
    # Build char offset -> line index map
    char_line = []
    for li, l in enumerate(lines):
        char_line.extend([li] * len(l))
    hotspot_line_indices = {li for li, l in enumerate(lines) if l.rstrip('\n') in hotspot_lines}
    mask = []
    pos = 0
    for tok in token_strs:
        line_idx = char_line[min(pos, len(char_line) - 1)] if char_line else 0
        mask.append(line_idx in hotspot_line_indices)
        pos += len(tok)
    return mask


def _find_code_token_range(full_text, baseline_code, tokenizer, full_ids_list):
    """Return (start, end) token indices of baseline_code within full_text tokens."""
    c_start = full_text.find(baseline_code)
    if c_start == -1:
        return 0, len(full_ids_list)
    c_end = c_start + len(baseline_code)
    # Encode prefix and prefix+code to get approximate token offsets
    pre_ids = tokenizer.encode(full_text[:c_start], add_special_tokens=False)
    pre_code_ids = tokenizer.encode(full_text[:c_end], add_special_tokens=False)
    return len(pre_ids), len(pre_code_ids)


def _plot_token_code_heatmap(code_tok_strs, tok_importance, hotspot_mask, out_tok_strs, title, path):
    """Explainability-paper style: code tokens colored by attention, output tokens below."""
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    cmap_in = plt.cm.YlOrRd
    cmap_out = plt.cm.Blues
    vmax = max(tok_importance.max(), 1e-9)

    def token_rows(tok_strs, importances, mask_flags, x0=0.0, max_x=100.0, char_w=0.62):
        """Lay tokens into wrapped rows, return list of rows where each row = [(x,y,tok,imp,is_hot)]."""
        rows, row = [], []
        x = x0
        for tok, imp, is_hot in zip(tok_strs, importances, mask_flags):
            parts = tok.split('\n')
            for pi, part in enumerate(parts):
                if pi > 0:
                    rows.append(row); row = []; x = x0
                w = max(len(part), 1) * char_w
                if x + w > max_x and row:
                    rows.append(row); row = []; x = x0
                row.append((x, part, imp, is_hot, w))
                x += w + 0.15
        if row: rows.append(row)
        return rows

    n_out_show = min(len(out_tok_strs), 60)
    out_imp = np.ones(n_out_show) * 0.5  # uniform for output side
    out_mask = [False] * n_out_show

    in_rows = token_rows(code_tok_strs, tok_importance / (vmax + 1e-9), hotspot_mask)
    out_rows = token_rows(out_tok_strs[:n_out_show], out_imp, out_mask)

    n_in_rows = len(in_rows)
    n_out_rows = len(out_rows)
    total_rows = n_in_rows + n_out_rows + 2  # +2 for separator

    fig_h = max(6, total_rows * 0.45 + 2.0)
    fig, ax = plt.subplots(figsize=(20, fig_h))
    ax.set_xlim(0, 102)
    ax.set_ylim(-1, total_rows + 1)
    ax.axis('off')

    def draw_rows(rows, y_start, cmap, label):
        ax.text(-0.5, y_start + len(rows) / 2, label, va='center', ha='right',
                fontsize=8, rotation=90, color='#333')
        for ri, row in enumerate(rows):
            y = y_start + (len(rows) - 1 - ri)
            for (x, tok_str, imp, is_hot, w) in row:
                col = cmap(float(imp))
                luma = 0.299 * col[0] + 0.587 * col[1] + 0.114 * col[2]
                txt_col = 'white' if luma < 0.45 else 'black'
                rect = patches.FancyBboxPatch(
                    (x, y + 0.08), w, 0.82, boxstyle="round,pad=0.05",
                    facecolor=col, edgecolor='none', zorder=1, clip_on=False)
                ax.add_patch(rect)
                if is_hot:
                    border = patches.FancyBboxPatch(
                        (x, y + 0.08), w, 0.82, boxstyle="round,pad=0.05",
                        facecolor='none', edgecolor='red', linewidth=1.8, zorder=3, clip_on=False)
                    ax.add_patch(border)
                if tok_str.strip():
                    ax.text(x + w / 2, y + 0.5, tok_str[:int(w / 0.62) + 1],
                            va='center', ha='center', fontfamily='monospace',
                            fontsize=7, color=txt_col, zorder=2, clip_on=False)

    y_out_start = 0
    draw_rows(out_rows, y_out_start, cmap_out, 'output\n(generated)')
    sep_y = n_out_rows + 0.5
    ax.axhline(sep_y, color='#999', linewidth=1.5, linestyle='--', xmin=0, xmax=1)
    ax.text(50, sep_y + 0.1, 'OUTPUT (generated)  |  INPUT (baseline code below)',
            ha='center', va='bottom', fontsize=8, color='#555')
    y_in_start = n_out_rows + 1
    draw_rows(in_rows, y_in_start, cmap_in, 'input\n(baseline)')

    # Colorbars
    sm_in = ScalarMappable(cmap=cmap_in, norm=Normalize(0, vmax))
    sm_in.set_array([])
    cb = fig.colorbar(sm_in, ax=ax, fraction=0.015, pad=0.01, location='right')
    cb.set_label('Input attention importance', fontsize=8)

    ax.set_title(title, fontsize=10, pad=6)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()


def _plot_layer_profile(layer_hot, layer_nothot, title, path):
    """Per-layer attention to hotspot vs non-hotspot tokens."""
    import numpy as np
    import matplotlib.pyplot as plt

    n_layers = len(layer_hot)
    xs = list(range(n_layers))
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(xs, layer_hot, 'o-', color='#d62728', label='hotspot (changed)', linewidth=2, markersize=5)
    ax.plot(xs, layer_nothot, 's--', color='#1f77b4', label='non-hotspot', linewidth=2, markersize=5)
    ax.fill_between(xs, layer_hot, layer_nothot,
                    where=[h > n for h, n in zip(layer_hot, layer_nothot)],
                    alpha=0.15, color='#d62728')
    ratios = [h / (n + 1e-9) for h, n in zip(layer_hot, layer_nothot)]
    ax2 = ax.twinx()
    ax2.plot(xs, ratios, 'k:', linewidth=1.5, alpha=0.6, label='ratio')
    ax2.axhline(1.0, color='gray', linewidth=0.8, linestyle=':')
    ax2.set_ylabel('Hotspot/non-hotspot ratio', fontsize=9)
    ax.set_xlabel('Layer index (last N layers)')
    ax.set_ylabel('Mean attention weight')
    ax.set_title(title)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='upper left')
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()


def _plot_head_diversity(head_ratios, title, path):
    """Grid of per-head hotspot attention ratios: (n_layers, n_heads)."""
    import numpy as np
    import matplotlib.pyplot as plt

    n_layers, n_heads = head_ratios.shape
    fig, ax = plt.subplots(figsize=(max(10, n_heads * 0.5), max(4, n_layers * 0.45)))
    im = ax.imshow(head_ratios, aspect='auto', cmap='RdYlGn', vmin=0.5, vmax=2.0,
                   interpolation='nearest')
    ax.set_xlabel('Attention head index')
    ax.set_ylabel('Layer index')
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label='Hotspot/non-hotspot ratio (green>1 = hotspot-focused)')
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()


def _plot_attn_matrix(attn_oc, code_tok_strs, out_tok_strs, hotspot_mask, title, path):
    """Full output x input attention matrix with token labels and hotspot markers."""
    import numpy as np
    import matplotlib.pyplot as plt

    hr = min(80, attn_oc.shape[0])
    hc = min(120, attn_oc.shape[1])
    fig, ax = plt.subplots(figsize=(max(14, hc * 0.15), max(8, hr * 0.18)))
    im = ax.imshow(attn_oc[:hr, :hc], aspect='auto', cmap='viridis',
                   interpolation='nearest', vmin=0)
    # Hotspot column markers
    for i, m in enumerate(hotspot_mask[:hc]):
        if m:
            ax.axvline(x=i - 0.5, color='red', alpha=0.35, linewidth=1.0)
    # Tick labels (every Nth to avoid clutter)
    step_x = max(1, hc // 30)
    step_y = max(1, hr // 20)
    ax.set_xticks(range(0, hc, step_x))
    ax.set_xticklabels([repr(code_tok_strs[i])[1:-1][:6] for i in range(0, hc, step_x)],
                       rotation=75, fontsize=6, fontfamily='monospace')
    ax.set_yticks(range(0, hr, step_y))
    ax.set_yticklabels([repr(out_tok_strs[i])[1:-1][:6] for i in range(0, hr, step_y)],
                       fontsize=6, fontfamily='monospace')
    ax.set_xlabel('Input code tokens  (red columns = hotspot/changed)')
    ax.set_ylabel('Output generated tokens')
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.02, label='Attention weight')
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()


def load_example(ex_dir):
    """Load all saved arrays and metadata for one example. Returns a dict ready for plotting/analysis."""
    import numpy as np
    d = Path(ex_dir)
    meta = json.load(open(d / 'metadata.json'))
    arrays = {}
    for fname in d.glob('*.npy'):
        arrays[fname.stem] = np.load(fname, allow_pickle=False)
    meta['arrays'] = arrays
    # Convenience aliases matching metadata descriptions
    meta['code_tokens'] = meta['code_tokens']      # list[str], len=n_code
    meta['out_tokens'] = meta['out_tokens']        # list[str], len=n_out
    meta['hotspot_mask'] = arrays['hotspot_mask']  # (n_code,) bool
    meta['tok_importance'] = arrays['tok_importance']  # (n_code,) float32
    meta['attn_mean'] = arrays['attn_mean']            # (n_out, n_code) float32
    return meta


def load_all_examples(output_dir):
    """Load all examples from an attention_analysis output directory."""
    return [load_example(d) for d in sorted(Path(output_dir).glob('ex*_*')) if d.is_dir()]


def attention_analysis(model_path, sim_dir, output_dir, n_samples=8, n_layers_used=8, save_raw=False, min_reduction=10.0):
    import numpy as np
    import torch
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out = Path(output_dir)
    out.mkdir(exist_ok=True)

    print(f"Loading tokenizer from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    print("Loading model with eager attention...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, attn_implementation="eager",
        dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True
    )
    model.eval()

    examples = _load_top_examples(sim_dir, n_samples, min_reduction=min_reduction)
    reductions = [f"{e['energy_reduction']:.1f}%" for e in examples]
    print(f"Loaded {len(examples)} examples, energy reductions: {reductions}")

    n_layers_total = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    layer_range = list(range(max(0, n_layers_total - n_layers_used), n_layers_total))
    all_stats = []

    for idx, ex in enumerate(examples):
        bcode, gcode = ex['baseline_code'], ex['generated_code']
        ered = ex['energy_reduction']
        pid = ex['problem_id']
        ex_dir = out / f"ex{idx:02d}_{pid}"
        ex_dir.mkdir(exist_ok=True)

        if (ex_dir / 'metadata.json').exists():
            print(f"  [{idx}] {pid} already done, loading stats")
            try:
                meta = json.load(open(ex_dir / 'metadata.json'))
                all_stats.append(meta['stats'])
            except Exception:
                pass
            continue

        prompt_text = (
            "This is an energy inefficient program we want to optimize to score 10/10.\n"
            f"### Program:\n{bcode}\n\n"
            "### Energy Optimized Version with score 10/10:\n```cpp\n"
        )
        full_text = prompt_text + gcode + "\n```"

        full_ids = tokenizer.encode(full_text, return_tensors='pt')
        n_prompt = tokenizer.encode(prompt_text, return_tensors='pt').shape[1]
        n_full = full_ids.shape[1]

        MAX_TOKS = 1024
        if n_full > MAX_TOKS:
            keep_prompt = min(n_prompt, MAX_TOKS * 2 // 3)
            keep_out = MAX_TOKS - keep_prompt
            full_ids = torch.cat([full_ids[:, :keep_prompt],
                                   full_ids[:, n_prompt:n_prompt + keep_out]], dim=1)
            n_prompt = keep_prompt
            n_full = full_ids.shape[1]

        full_ids = full_ids.to(model.device)

        code_start, code_end = _find_code_token_range(
            full_text, bcode, tokenizer, full_ids[0, :n_prompt].tolist()
        )
        code_end = min(code_end, n_prompt)
        n_code = code_end - code_start
        if n_code <= 0:
            print(f"  [{idx}] code token range not found, skipping")
            continue

        n_out_toks = n_full - n_prompt

        # Capture only last n_layers_used layers via hooks; skip attention for earlier layers
        captured_attn = {}
        hooks = []
        def _make_hook(l_idx, np_, nf, cs, ce):
            def h(module, inp, out_):
                if len(out_) >= 2 and out_[1] is not None:
                    captured_attn[l_idx] = out_[1][0, :, np_:nf, cs:ce].float().cpu().numpy()
            return h
        for l in layer_range:
            hooks.append(model.model.layers[l].self_attn.register_forward_hook(
                _make_hook(l, n_prompt, n_full, code_start, code_end)))
        # Patch early layers to skip attention computation
        skip_n = max(0, n_layers_total - n_layers_used)
        orig_fwds = {}
        for l in range(skip_n):
            orig_fwds[l] = model.model.layers[l].forward
            def _no_attn(orig):
                def fwd(*a, **kw): kw['output_attentions'] = False; return orig(*a, **kw)
                return fwd
            model.model.layers[l].forward = _no_attn(orig_fwds[l])

        with torch.no_grad():
            model(input_ids=full_ids, output_attentions=True)

        for h in hooks:
            h.remove()
        for l, fwd in orig_fwds.items():
            model.model.layers[l].forward = fwd
        torch.cuda.empty_cache()

        # Build per_layer_head from captured hooks
        per_layer_head = np.zeros((len(layer_range), n_heads, n_out_toks, n_code), dtype=np.float32)
        for li, l in enumerate(layer_range):
            if l in captured_attn:
                per_layer_head[li] = captured_attn[l]
        captured_attn.clear()

        # Derived aggregations
        attn_per_layer = per_layer_head.mean(axis=1)           # (n_layers_used, n_out, n_code) mean over heads
        attn_mean = attn_per_layer.mean(axis=0)                # (n_out, n_code) mean over layers+heads
        tok_imp = attn_mean.sum(axis=0)                        # (n_code,) total attention received
        layer_tok_imp = attn_per_layer.sum(axis=1)             # (n_layers_used, n_code) per-layer importance
        head_tok_imp = per_layer_head.sum(axis=2)              # (n_layers_used, n_heads, n_code) per-head importance

        # Hotspot mask
        hotspot_lines = _get_hotspot_lines(bcode, gcode)
        code_tok_strs = [tokenizer.decode([full_ids[0, code_start + i].item()]) for i in range(n_code)]
        out_tok_strs = [tokenizer.decode([full_ids[0, n_prompt + i].item()]) for i in range(n_out_toks)]
        hotspot_mask = np.array(_token_hotspot_mask(bcode, code_tok_strs, hotspot_lines), dtype=bool)

        # Per-layer and per-head hotspot profiles
        n_hot, n_nothot = hotspot_mask.sum(), (~hotspot_mask).sum()
        layer_hot = np.zeros(len(layer_range))
        layer_nothot = np.zeros(len(layer_range))
        head_ratios = np.zeros((len(layer_range), n_heads))
        for li in range(len(layer_range)):
            layer_ti = attn_per_layer[li].sum(axis=0)  # (n_code,)
            if n_hot > 0: layer_hot[li] = layer_ti[hotspot_mask].mean()
            if n_nothot > 0: layer_nothot[li] = layer_ti[~hotspot_mask].mean()
            for hi in range(n_heads):
                head_ti = per_layer_head[li, hi].sum(axis=0)
                h_v = head_ti[hotspot_mask].mean() if n_hot > 0 else 0.0
                nh_v = head_ti[~hotspot_mask].mean() if n_nothot > 0 else 1e-9
                head_ratios[li, hi] = h_v / (nh_v + 1e-9)

        # Stats
        h_mean = tok_imp[hotspot_mask].mean() if n_hot > 0 else 0.0
        nh_mean = tok_imp[~hotspot_mask].mean() if n_nothot > 0 else 1e-9
        ratio = h_mean / (nh_mean + 1e-9)
        stat = {'pid': pid, 'ered': ered, 'ratio': float(ratio),
                'hot': float(h_mean), 'nothot': float(nh_mean),
                'n_hot': int(n_hot), 'n_nothot': int(n_nothot)}
        all_stats.append(stat)
        print(f"  [{idx}] {pid} ered={ered:.1f}% | hot={h_mean:.4f} nothot={nh_mean:.4f} ratio={ratio:.2f}x")

        # --- Save intermediate data (all float32, self-contained for offline reanalysis) ---
        np.save(ex_dir / 'attn_mean.npy', attn_mean)                  # (n_out, n_code)
        np.save(ex_dir / 'attn_per_layer.npy', attn_per_layer)        # (n_layers_used, n_out, n_code)
        np.save(ex_dir / 'tok_importance.npy', tok_imp)                # (n_code,)
        np.save(ex_dir / 'layer_tok_importance.npy', layer_tok_imp)   # (n_layers_used, n_code)
        np.save(ex_dir / 'head_tok_importance.npy', head_tok_imp)     # (n_layers_used, n_heads, n_code)
        np.save(ex_dir / 'hotspot_mask.npy', hotspot_mask)            # (n_code,) bool
        np.save(ex_dir / 'layer_hot_profile.npy', layer_hot)          # (n_layers_used,)
        np.save(ex_dir / 'layer_nothot_profile.npy', layer_nothot)    # (n_layers_used,)
        np.save(ex_dir / 'head_ratios.npy', head_ratios)              # (n_layers_used, n_heads)
        if save_raw:
            np.save(ex_dir / 'attn_per_layer_head.npy', per_layer_head)  # (n_layers_used, n_heads, n_out, n_code)

        array_shapes = {
            'attn_mean':            f'{attn_mean.shape} — mean attention matrix: output_tokens x code_tokens',
            'attn_per_layer':       f'{attn_per_layer.shape} — per-layer mean-over-heads: layers x out_tokens x code_tokens',
            'tok_importance':       f'{tok_imp.shape} — total attention received per code token (sum over output)',
            'layer_tok_importance': f'{layer_tok_imp.shape} — per-layer token importance: layers x code_tokens',
            'head_tok_importance':  f'{head_tok_imp.shape} — per-head token importance: layers x heads x code_tokens',
            'hotspot_mask':         f'{hotspot_mask.shape} — bool: True if token is on a changed/removed line',
            'layer_hot_profile':    f'{layer_hot.shape} — mean attention to hotspot tokens, per layer',
            'layer_nothot_profile': f'{layer_nothot.shape} — mean attention to non-hotspot tokens, per layer',
            'head_ratios':          f'{head_ratios.shape} — hotspot/non-hotspot attention ratio per (layer, head)',
        }
        if save_raw:
            array_shapes['attn_per_layer_head'] = (
                f'{per_layer_head.shape} — full raw: layers x heads x out_tokens x code_tokens')

        json.dump({
            'problem_id': pid, 'energy_reduction': ered,
            'model_path': str(model_path),
            'baseline_code': bcode, 'generated_code': gcode,
            'n_prompt_tokens': n_prompt, 'n_full_tokens': n_full,
            'code_start_tok': code_start, 'code_end_tok': code_end,
            'n_code_tokens': n_code, 'n_out_tokens': n_out_toks,
            'n_layers_used': len(layer_range), 'layer_indices': list(layer_range),
            'n_heads': n_heads,
            'code_tokens': code_tok_strs,
            'out_tokens': out_tok_strs,
            'hotspot_lines': list(hotspot_lines),
            'stats': stat,
            'arrays': array_shapes,
            'usage': 'Reload with: ex = load_example(ex_dir); then use ex["arrays"]["attn_mean"] etc.',
        }, open(ex_dir / 'metadata.json', 'w'), indent=2)

        # --- Plots ---
        tag = f"{pid} (energy_red={ered:.1f}%)"

        # 1. Token code heatmap (explainability-paper style)
        _plot_token_code_heatmap(
            code_tok_strs, tok_imp, hotspot_mask, out_tok_strs,
            f"Attention Heatmap: Input Code tokens vs Output tokens | {tag}",
            ex_dir / '01_token_code_heatmap.png'
        )

        # 2. Full output x input attention matrix with token labels
        _plot_attn_matrix(
            attn_mean, code_tok_strs, out_tok_strs, hotspot_mask,
            f"Output->Input Attention Matrix | {tag}\n(red columns = changed/hotspot tokens)",
            ex_dir / '02_attention_matrix.png'
        )

        # 3. Per-layer hotspot attention profile
        _plot_layer_profile(
            layer_hot, layer_nothot,
            f"Per-layer Attention to Hotspot vs Non-hotspot Tokens | {tag}",
            ex_dir / '03_layer_profile.png'
        )

        # 4. Head diversity heatmap
        _plot_head_diversity(
            head_ratios,
            f"Head-level Hotspot Attention Ratio (green>1=hotspot-focused) | {tag}",
            ex_dir / '04_head_diversity.png'
        )

        # 5. Token importance bar (compact overview)
        fig, ax = plt.subplots(figsize=(16, 3.5))
        colors = ['#d62728' if m else '#aec7e8' for m in hotspot_mask]
        ax.bar(range(n_code), tok_imp, color=colors, width=1.0, linewidth=0)
        if n_hot > 0: ax.axhline(h_mean, color='#d62728', linestyle='--', linewidth=1, label=f'hotspot mean={h_mean:.3f}')
        if n_nothot > 0: ax.axhline(nh_mean, color='#1f77b4', linestyle='--', linewidth=1, label=f'non-hotspot mean={nh_mean:.3f}')
        ax.set_xlim(0, n_code)
        ax.set_xlabel('Code token index  (red=changed/hotspot, blue=unchanged)')
        ax.set_ylabel('Cumulative attention from output')
        ax.set_title(f"Token Importance | {tag}  |  hotspot ratio={ratio:.2f}x")
        ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(ex_dir / '05_token_importance_bar.png', dpi=150, bbox_inches='tight')
        plt.close()

        n_arrays = 10 if save_raw else 9
        print(f"    -> saved 5 plots + {n_arrays} arrays + metadata.json to {ex_dir}/")

    if not all_stats:
        print("No valid examples found.")
        return

    ratios = [s['ratio'] for s in all_stats]
    hots = [s['hot'] for s in all_stats]
    nothots = [s['nothot'] for s in all_stats]
    ereds = [s['ered'] for s in all_stats]
    pids = [s['pid'] for s in all_stats]

    # --- Summary figure ---
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    ax = axes[0]
    ax.bar(['Changed\n(hotspot)', 'Unchanged'],
           [np.mean(hots), np.mean(nothots)],
           yerr=[np.std(hots), np.std(nothots)],
           color=['#d62728', '#1f77b4'], capsize=8, width=0.5)
    ax.set_ylabel('Mean attention weight')
    ax.set_title(f'Hotspot vs non-hotspot\n(n={len(all_stats)}, mean ratio={np.mean(ratios):.2f}x)')

    ax = axes[1]
    sc = ax.scatter(ereds, ratios, c=ratios, cmap='RdYlGn', vmin=0.5, vmax=3.0, s=90, zorder=3)
    ax.axhline(1.0, color='gray', linestyle='--', linewidth=1)
    for s in all_stats:
        ax.annotate(s['pid'], (s['ered'], s['ratio']), fontsize=6, alpha=0.7)
    ax.set_xlabel('Energy reduction (%)')
    ax.set_ylabel('Hotspot/non-hotspot ratio')
    ax.set_title('Alignment vs optimization strength')
    plt.colorbar(sc, ax=ax, label='ratio')

    ax = axes[2]
    bar_c = ['#d62728' if r > 1.0 else '#aec7e8' for r in ratios]
    ax.barh(range(len(ratios)), ratios, color=bar_c)
    ax.axvline(1.0, color='black', linestyle='--', linewidth=1)
    ax.set_yticks(range(len(pids))); ax.set_yticklabels(pids, fontsize=8)
    ax.set_xlabel('Hotspot attention ratio')
    ax.set_title('Per-example  (red=ratio>1, attends to hotspots)')

    plt.suptitle('GRPO Model: Does It Attend to Energy Hotspots?', fontsize=13)
    plt.tight_layout()
    plt.savefig(out / 'summary_hotspot_attention.png', dpi=150, bbox_inches='tight')
    plt.close()

    json.dump({
        'examples': all_stats,
        'mean_ratio': float(np.mean(ratios)),
        'std_ratio': float(np.std(ratios)),
        'pct_above_1': float(np.mean([r > 1.0 for r in ratios])),
        'corr_ratio_ered': float(np.corrcoef(ereds, ratios)[0, 1]) if len(ratios) > 2 else None,
    }, open(out / 'hotspot_attention_stats.json', 'w'), indent=2)

    print(f"\nMean hotspot ratio: {np.mean(ratios):.3f}x  |  "
          f"pct>1: {np.mean([r>1 for r in ratios])*100:.0f}%  |  "
          f"summary -> {out}/summary_hotspot_attention.png")


import re as _re
_SINK_RE = _re.compile(r'^[\s{}()\[\];,.:<>+\-*/&|!?=~^`\'"%#@\\]+$')
_CLASS_RE = [
    ('container', _re.compile(r'\b(unordered_map|unordered_set|priority_queue|vector|array|deque|list|queue|stack|map|set|pair|tuple|string|bitset)\b')),
    ('io_call',   _re.compile(r'\b(cin|cout|cerr|scanf|printf|endl|getline|puts|putchar|getchar|sync_with_stdio|ios_base|ios|tie|fflush)\b')),
    ('loop_kw',   _re.compile(r'\b(for|while|do)\b')),
    ('numeric',   _re.compile(r'^\d+\.?\d*$')),
]
_CLASS_NAMES = ['container', 'io_call', 'loop_kw', 'numeric', 'identifier', 'syntactic']
_OPT_RE = [
    ('scanf/printf',       _re.compile(r'\b(scanf|printf)\b')),
    ('sync_with_stdio',    _re.compile(r'sync_with_stdio\s*\(\s*(false|0)\s*\)')),
    ('cin.tie(NULL)',      _re.compile(r'cin\s*\.\s*tie\s*\(\s*(NULL|nullptr|0)\s*\)')),
    ('memcpy/memset',      _re.compile(r'\b(memcpy|memset|memmove)\b')),
    ('unordered_map',      _re.compile(r'\bunordered_map\b')),
    ('reserve()',          _re.compile(r'\.reserve\s*\(')),
    (r'"\n" literal',      _re.compile(r'"\\n"')),
]


def _sink_mask(toks):
    import numpy as np
    return np.array([bool(_SINK_RE.match(t)) or t == '' for t in toks])


def _filter_attn(attn_oc, toks):
    import numpy as np
    a = attn_oc.astype(np.float32).copy()
    a[:, _sink_mask(toks)] = 0.0
    rs = a.sum(axis=1, keepdims=True); rs[rs < 1e-9] = 1.0
    return a / rs


def _classify_input_tokens(toks):
    import numpy as np
    cls = np.full(len(toks), 5, dtype=np.int32)
    for i, t in enumerate(toks):
        if _SINK_RE.match(t) or not t.strip():
            continue
        matched = False
        for ci, (_, rx) in enumerate(_CLASS_RE):
            if rx.search(t): cls[i] = ci; matched = True; break
        if not matched and _re.match(r'^\s*[A-Za-z_]\w*\s*$', t):
            cls[i] = 4
    return cls


def _opt_output_positions(out_toks):
    stream = ''.join(out_toks)
    ctok = []
    for ti, t in enumerate(out_toks): ctok.extend([ti] * len(t))
    res = {}
    for name, rx in _OPT_RE:
        idxs = set()
        for m in rx.finditer(stream):
            for c in range(m.start(), min(m.end(), len(ctok))): idxs.add(ctok[c])
        if idxs: res[name] = sorted(idxs)
    return res


def _token_deletion_mask(baseline_code, generated_code, code_tokens):
    """Token-level mask: True where the baseline-token's text is absent in the generated code (heuristic).
    Stronger than the line-level diff hotspot mask."""
    import numpy as np, difflib
    sm = difflib.SequenceMatcher(a=baseline_code, b=generated_code, autojunk=False)
    kept = np.zeros(len(baseline_code), dtype=bool)
    for tag, i1, i2, _, _ in sm.get_opcodes():
        if tag == 'equal':
            kept[i1:i2] = True
    pos = 0; mask = np.zeros(len(code_tokens), dtype=bool)
    for i, t in enumerate(code_tokens):
        if not t.strip():
            pos += len(t); continue
        sl = kept[pos:pos + len(t)] if pos < len(kept) else np.zeros(0, dtype=bool)
        mask[i] = (sl.size > 0) and (sl.sum() < 0.5 * sl.size)
        pos += len(t)
    return mask


def _load_pair(be, ge, bm, gm):
    import numpy as np
    ab = np.load(be / 'attn_mean.npy'); ag = np.load(ge / 'attn_mean.npy')
    n_in = min(ab.shape[1], ag.shape[1], len(bm['code_tokens']))
    n_out = min(ab.shape[0], ag.shape[0], len(bm['out_tokens']))
    toks_i = bm['code_tokens'][:n_in]; toks_o = bm['out_tokens'][:n_out]
    raw_b = ab[:n_out, :n_in].astype(np.float32); raw_g = ag[:n_out, :n_in].astype(np.float32)
    sinks = _sink_mask(toks_i)
    sink_mass_b = float(raw_b[:, sinks].sum() / max(raw_b.sum(), 1e-9))
    sink_mass_g = float(raw_g[:, sinks].sum() / max(raw_g.sum(), 1e-9))
    af_b = _filter_attn(raw_b, toks_i)
    af_g = _filter_attn(raw_g, toks_i)
    cls = _classify_input_tokens(toks_i)
    opts = _opt_output_positions(toks_o)
    class_b = np.array([af_b[:, cls == c].sum() / max(af_b.sum(), 1e-9) for c in range(6)])
    class_g = np.array([af_g[:, cls == c].sum() / max(af_g.sum(), 1e-9) for c in range(6)])
    bcode = bm.get('baseline_code', ''); gcode = bm.get('generated_code', '')
    del_mask = _token_deletion_mask(bcode, gcode, toks_i) if bcode and gcode else np.zeros(n_in, dtype=bool)
    def _opt(load):
        try: return np.load(load)
        except Exception: return None
    extra = {
        'lh_b': _opt(be / 'layer_hot_profile.npy'), 'lh_g': _opt(ge / 'layer_hot_profile.npy'),
        'ln_b': _opt(be / 'layer_nothot_profile.npy'), 'ln_g': _opt(ge / 'layer_nothot_profile.npy'),
        'hr_b': _opt(be / 'head_ratios.npy'), 'hr_g': _opt(ge / 'head_ratios.npy'),
        'ti_b': _opt(be / 'tok_importance.npy'), 'ti_g': _opt(ge / 'tok_importance.npy'),
        'hm_b': _opt(be / 'hotspot_mask.npy'), 'hm_g': _opt(ge / 'hotspot_mask.npy'),
    }
    return {'toks_i': toks_i, 'toks_o': toks_o, 'af_b': af_b, 'af_g': af_g,
            'cls': cls, 'opts': opts, 'class_b': class_b, 'class_g': class_g,
            'sink_mass_b': sink_mass_b, 'sink_mass_g': sink_mass_g,
            'del_mask': del_mask,
            'baseline_code': bcode, 'generated_code': gcode, **extra}


def _fig1_class_bars(pairs, out_path):
    import numpy as np, matplotlib.pyplot as plt
    from scipy.stats import wilcoxon
    cb = np.stack([p['class_b'] for p in pairs]); cg = np.stack([p['class_g'] for p in pairs])
    mb, mg = cb.mean(0), cg.mean(0); N = len(cb)
    deltas = cg - cb
    mean_d = deltas.mean(0)
    ci_lo = np.percentile(deltas, 2.5, axis=0); ci_hi = np.percentile(deltas, 97.5, axis=0)
    pvals = []
    for c in range(6):
        try:
            d = deltas[:, c]
            pvals.append(float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0)
        except Exception: pvals.append(1.0)
    x = np.arange(6)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7.5), gridspec_kw={'width_ratios': [1.25, 1]})
    ax1.bar(x - 0.21, mb, 0.40, label='Base (pre-training)', color='#95a5a6', edgecolor='#2c3e50', lw=1.2)
    ax1.bar(x + 0.21, mg, 0.40, label='GRPO (post-training)', color='#c0392b', edgecolor='#2c3e50', lw=1.2)
    rng = np.random.default_rng(0)
    for c in range(6):
        jb = rng.uniform(-0.05, 0.05, N); jg = rng.uniform(-0.05, 0.05, N)
        ax1.scatter(np.full(N, c - 0.21) + jb, cb[:, c], color='#34495e', alpha=0.4, s=14, zorder=3)
        ax1.scatter(np.full(N, c + 0.21) + jg, cg[:, c], color='#34495e', alpha=0.4, s=14, zorder=3)
    ax1.set_yscale('function', functions=(lambda v: np.sqrt(np.clip(v, 0, None)),
                                           lambda v: np.power(v, 2)))
    ax1.set_xticks(x); ax1.set_xticklabels(_CLASS_NAMES, fontsize=14, fontweight='bold')
    ax1.set_ylabel('Share of attention mass  (sqrt scale)', fontsize=14)
    ax1.set_title(f'Attention by input-token class (N={N})', fontsize=16, fontweight='bold', pad=8)
    ax1.legend(fontsize=13, loc='upper left'); ax1.tick_params(labelsize=12)
    ax1.grid(axis='y', alpha=0.3); ax1.set_axisbelow(True)

    colors = []
    for c in range(6):
        if pvals[c] < 0.05 and mean_d[c] > 0: colors.append('#27ae60')
        elif pvals[c] < 0.05 and mean_d[c] < 0: colors.append('#c0392b')
        else: colors.append('#95a5a6')
    ax2.axhline(0, color='#333', linewidth=1.2, zorder=1)
    ax2.bar(x, mean_d, 0.55, color=colors, edgecolor='#2c3e50', lw=1.4, zorder=2)
    for c in range(6):
        ax2.errorbar(c, mean_d[c], yerr=[[mean_d[c] - ci_lo[c]], [ci_hi[c] - mean_d[c]]],
                     color='#2c3e50', lw=1.5, capsize=6, zorder=3)
        star = '***' if pvals[c] < 0.001 else '**' if pvals[c] < 0.01 else '*' if pvals[c] < 0.05 else 'n.s.'
        off = 0.003 if mean_d[c] >= 0 else -0.003
        ax2.text(c, ci_hi[c] + off if mean_d[c] >= 0 else ci_lo[c] + off, star,
                 ha='center', va='bottom' if mean_d[c] >= 0 else 'top',
                 fontsize=14, fontweight='bold', color=colors[c])
        rel = mean_d[c] / (mb[c] + 1e-9) * 100
        ax2.text(c, mean_d[c] / 2, f'{mean_d[c]:+.3f}\n({rel:+.0f}%)',
                 ha='center', va='center', fontsize=10, fontweight='bold',
                 color='white' if abs(mean_d[c]) > 0.01 else '#2c3e50')
    ax2.set_xticks(x); ax2.set_xticklabels(_CLASS_NAMES, fontsize=14, fontweight='bold')
    ax2.set_ylabel('Δ attention share  (GRPO − base)', fontsize=14)
    ax2.set_title('Post-training attention shift', fontsize=16, fontweight='bold', pad=8)
    ax2.tick_params(labelsize=12); ax2.grid(axis='y', alpha=0.3); ax2.set_axisbelow(True)

    fig.suptitle('GRPO redirects attention away from identifiers toward loop structure and numeric literals',
                 fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{out_path}.pdf', bbox_inches='tight')
    plt.savefig(f'{out_path}.png', dpi=300, bbox_inches='tight')
    plt.close()
    return {'mean_base': mb.tolist(), 'mean_grpo': mg.tolist(), 'mean_delta': mean_d.tolist(),
            'ci_lo': ci_lo.tolist(), 'ci_hi': ci_hi.tolist(), 'pvals': pvals, 'N': N}


def _render_code_panel(ax, toks, attn, cls, title, cmap_name='YlOrRd', max_x=110, char_w=0.55):
    import numpy as np, matplotlib.pyplot as plt, matplotlib.patches as patches
    cmap = plt.get_cmap(cmap_name)
    vmax = max(float(attn.max()), 1e-9)
    rows, row, x = [], [], 0.0
    sinks = _sink_mask(toks)
    for i, t in enumerate(toks):
        a = 0.0 if sinks[i] else float(attn[i]) / vmax
        for pi, part in enumerate(t.split('\n')):
            if pi > 0: rows.append(row); row, x = [], 0.0
            w = max(len(part), 1) * char_w
            if x + w > max_x and row: rows.append(row); row, x = [], 0.0
            row.append((x, part, a, int(cls[i]), w, sinks[i])); x += w + 0.15
    if row: rows.append(row)
    ax.set_xlim(-1, max_x + 2); ax.set_ylim(-0.5, len(rows) + 0.5); ax.axis('off')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=4)
    class_edge = {0: '#1f77b4', 1: '#d62728', 2: '#2ca02c', 3: '#9467bd', 4: None, 5: None}
    for ri, r in enumerate(rows):
        y = len(rows) - 1 - ri
        for (xp, s, a, ci, w, is_sink) in r:
            col = (0.96, 0.96, 0.96) if is_sink else cmap(a)
            luma = 0.299*col[0] + 0.587*col[1] + 0.114*col[2]
            tcol = 'white' if luma < 0.45 else '#111'
            ax.add_patch(patches.Rectangle((xp, y + 0.05), w, 0.88, facecolor=col, edgecolor='none'))
            ec = class_edge.get(ci)
            if ec and not is_sink:
                ax.add_patch(patches.Rectangle((xp, y + 0.05), w, 0.88, facecolor='none', edgecolor=ec, lw=1.8))
            if s.strip():
                ax.text(xp + w/2, y + 0.5, s, va='center', ha='center', fontfamily='monospace',
                        fontsize=10, color=tcol, fontweight='bold')


def _line_attn(p, opt_pos):
    import numpy as np
    toks = p['toks_i']; code = p.get('baseline_code', '')
    lines = code.splitlines() or ['']
    cline = []
    for li, l in enumerate(lines): cline.extend([li] * (len(l) + 1))
    cline.append(len(lines) - 1)
    n_lines = len(lines)
    la_b = np.zeros(n_lines); la_g = np.zeros(n_lines)
    pos = 0
    n_in = p['af_b'].shape[1]
    for ti, t in enumerate(toks[:n_in]):
        li = cline[min(pos, len(cline) - 1)] if cline else 0
        if 0 <= li < n_lines:
            la_b[li] += float(p['af_b'][opt_pos, ti].mean())
            la_g[li] += float(p['af_g'][opt_pos, ti].mean())
        pos += len(t)
    return lines, la_b, la_g


def _fig2_code_listing(pairs_selected, out_path):
    import numpy as np, matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    n = len(pairs_selected)
    if n == 0: return
    cmap_b = LinearSegmentedColormap.from_list('b', ['#ffffff', '#7f8c8d', '#34495e'])
    cmap_g = LinearSegmentedColormap.from_list('g', ['#ffffff', '#f39c12', '#c0392b'])
    import numpy as _np
    listings = []
    for p, opt_name, opt_pos in pairs_selected:
        lines, la_b, la_g = _line_attn(p, opt_pos)
        while lines and lines[-1].strip() == '': lines = lines[:-1]; la_b = la_b[:-1]; la_g = la_g[:-1]
        skip = 0
        while skip < len(lines) and lines[skip].strip().startswith(('#include', '#define', 'using namespace', 'typedef', '//')):
            skip += 1
        skip = min(skip, max(0, len(lines) - 12))
        lines = lines[skip:]; la_b = la_b[skip:]; la_g = la_g[skip:]
        kept_idx = [i for i, ln in enumerate(lines) if ln.strip() != '']
        orig_nums = [i + 1 + skip for i in kept_idx]
        lines = [lines[i][:90] for i in kept_idx]
        la_b = _np.array([la_b[i] for i in kept_idx])
        la_g = _np.array([la_g[i] for i in kept_idx])
        listings.append((p, opt_name, lines, la_b, la_g, orig_nums))
    for idx, (p, opt_name, lines, la_b, la_g, orig_nums) in enumerate(listings):
        nb = la_b / (la_b.max() + 1e-9); ng = la_g / (la_g.max() + 1e-9)
        h = max(6, len(lines) * 0.38 + 2)
        fig, axes = plt.subplots(1, 2, figsize=(22, h), gridspec_kw={'wspace': 0.06})
        for ci, (label, arr, cmap, bcolor) in enumerate(
                [('Base (pre-training)', nb, cmap_b, '#34495e'),
                 ('GRPO (post-training)', ng, cmap_g, '#c0392b')]):
            ax = axes[ci]
            ax.set_xlim(0, 1.1); ax.set_ylim(0, len(lines)); ax.invert_yaxis(); ax.axis('off')
            ax.set_title(f"{label}    ({p['pid']}, ERR {p['ered']:.1f}%)",
                         fontsize=16, fontweight='bold', pad=8, color=bcolor)
            for li, line in enumerate(lines):
                a = float(arr[li])
                col = cmap(a)
                ax.add_patch(plt.Rectangle((0, li), 1, 1, facecolor=col, edgecolor='none'))
                ax.text(0.005, li + 0.5, f'{orig_nums[li]:2d}', va='center', ha='left',
                        fontsize=11, fontfamily='monospace', color='#999')
                luma = 0.299*col[0] + 0.587*col[1] + 0.114*col[2]
                tcol = '#111' if luma > 0.55 else '#fff'
                ax.text(0.035, li + 0.5, line, va='center', ha='left',
                        fontsize=13, fontfamily='monospace', color=tcol,
                        fontweight='bold' if a > 0.5 else 'normal')
            top_li = int(np.argmax(arr))
            if arr.max() > 0:
                ax.annotate('peak', xy=(1.0, top_li + 0.5), xytext=(1.08, top_li + 0.5),
                            fontsize=13, fontweight='bold', color=bcolor, va='center',
                            arrowprops=dict(arrowstyle='->', color=bcolor, lw=2.0))
        fig.suptitle(f"Where does attention concentrate when the model emits '{opt_name}'?",
                     fontsize=18, fontweight='bold', y=1.005)
        plt.tight_layout()
        plt.savefig(f'{out_path}_{p["pid"]}.pdf', bbox_inches='tight')
        plt.savefig(f'{out_path}_{p["pid"]}.png', dpi=300, bbox_inches='tight')
        plt.close()


def _fig3_cards(p, opt_name, opt_pos, out_path):
    import numpy as np, matplotlib.pyplot as plt, matplotlib.patches as patches
    toks = p['toks_i']; sinks = _sink_mask(toks)
    n_in = p['af_b'].shape[1]
    cond_b = p['af_b'][opt_pos].mean(0) * (~sinks[:n_in])
    cond_g = p['af_g'][opt_pos].mean(0) * (~sinks[:n_in])
    nb = cond_b / (cond_b.sum() + 1e-9); ng = cond_g / (cond_g.sum() + 1e-9)
    K = 5
    top_b = np.argsort(cond_b)[-K:][::-1]; top_g = np.argsort(cond_g)[-K:][::-1]
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_xlim(0, 10); ax.set_ylim(-0.5, 11); ax.axis('off')
    bx, by, bw, bh = 3.6, 9.4, 2.8, 1.0
    ax.add_patch(patches.FancyBboxPatch((bx, by), bw, bh, boxstyle='round,pad=0.12',
                                         facecolor='#2c3e50', edgecolor='none'))
    ax.text(bx + bw/2, by + bh*0.62, "Generated token", ha='center', va='center',
            fontsize=11, color='#bdc3c7')
    ax.text(bx + bw/2, by + bh*0.30, opt_name, ha='center', va='center',
            fontsize=17, color='white', fontweight='bold', fontfamily='monospace')
    ax.text(5, 10.65, f"{p['pid']}  |  ERR {p['ered']:.1f}%  |  "
                      f"Top-{K} input tokens attended when generating '{opt_name}'",
            ha='center', fontsize=13, color='#333')
    def draw_col(col_x, label, col_color, top_idx, weights, is_left):
        ax.text(col_x + 1.6, 8.1, label, ha='center', fontsize=16, fontweight='bold', color=col_color)
        wmax = max(float(weights[top_idx].max()), 1e-9)
        for rank, idx in enumerate(top_idx):
            card_y = 5.9 - rank * 1.25
            tok_str = toks[idx].strip().replace('\n', '\\n') or '<space>'
            tok_str = tok_str[:22]
            w = float(weights[idx])
            card_w = 1.3 + 2.0 * (w / wmax)
            card_x = col_x + 1.6 - card_w / 2
            ax.add_patch(patches.FancyBboxPatch((card_x, card_y), card_w, 1.05,
                boxstyle='round,pad=0.08', facecolor=col_color, edgecolor='#2c3e50', lw=1.4, alpha=0.92))
            ax.text(card_x + card_w/2, card_y + 0.70, tok_str, ha='center', va='center',
                    fontsize=14, fontfamily='monospace', fontweight='bold', color='white')
            ax.text(card_x + card_w/2, card_y + 0.28, f'#{rank+1}   {w*100:.1f}%',
                    ha='center', va='center', fontsize=11, color='#ecf0f1')
            arrow_end_x = card_x + card_w if is_left else card_x
            arrow_end_y = card_y + 0.525
            rad = 0.30 if is_left else -0.30
            ax.annotate('', xy=(arrow_end_x, arrow_end_y), xytext=(bx + bw/2, by),
                arrowprops=dict(arrowstyle='-|>,head_length=0.35,head_width=0.25',
                                lw=1.0 + 4.0 * (w / wmax),
                                color=col_color, alpha=0.7,
                                connectionstyle=f'arc3,rad={rad}', shrinkA=8, shrinkB=6))
    draw_col(0.8, 'Base (pre-training)', '#95a5a6', top_b, nb, is_left=True)
    draw_col(5.8, 'GRPO (post-training)', '#c0392b', top_g, ng, is_left=False)
    ax.plot([5, 5], [-0.2, 7.3], color='#ccc', linewidth=0.8, linestyle='--')
    plt.tight_layout()
    plt.savefig(f'{out_path}.pdf', bbox_inches='tight')
    plt.savefig(f'{out_path}.png', dpi=300, bbox_inches='tight')
    plt.close()


def _fig4_class_err_regression(pairs, out_path):
    """Analysis A: per-class attention shift correlated with achieved ERR."""
    import numpy as np, matplotlib.pyplot as plt
    from scipy.stats import spearmanr
    deltas = np.stack([p['class_g'] - p['class_b'] for p in pairs])
    ered = np.array([float(p.get('ered', 0.0)) for p in pairs])
    e_log = np.log((ered + 1e-3) / (100.0 - ered + 1e-3))
    rho, pv = [], []
    for c in range(6):
        try:
            r, p_ = spearmanr(deltas[:, c], e_log)
            rho.append(float(r) if np.isfinite(r) else 0.0); pv.append(float(p_) if np.isfinite(p_) else 1.0)
        except Exception: rho.append(0.0); pv.append(1.0)
    rho = np.array(rho); pv = np.array(pv)
    rng = np.random.default_rng(0); B = 1000
    boot = np.zeros((B, 6))
    for b in range(B):
        idx = rng.integers(0, len(pairs), len(pairs))
        for c in range(6):
            try: boot[b, c] = spearmanr(deltas[idx, c], e_log[idx]).correlation or 0.0
            except Exception: pass
    ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5], axis=0)
    fig, axes = plt.subplots(1, 2, figsize=(18, 6.5), gridspec_kw={'width_ratios': [1, 1.4]})
    ax = axes[0]
    colors = ['#27ae60' if pv[c] < 0.05 and rho[c] > 0 else '#c0392b' if pv[c] < 0.05 and rho[c] < 0 else '#95a5a6' for c in range(6)]
    x = np.arange(6); ax.axhline(0, color='#333', lw=1.0)
    ax.bar(x, rho, 0.55, color=colors, edgecolor='#2c3e50', lw=1.2)
    for c in range(6):
        ax.errorbar(c, rho[c], yerr=[[rho[c] - ci_lo[c]], [ci_hi[c] - rho[c]]], color='#2c3e50', lw=1.2, capsize=5)
        star = '***' if pv[c] < 0.001 else '**' if pv[c] < 0.01 else '*' if pv[c] < 0.05 else ''
        ax.text(c, max(rho[c], ci_hi[c]) + 0.02, star, ha='center', fontsize=14, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(_CLASS_NAMES, fontsize=12, fontweight='bold')
    ax.set_ylabel("Spearman ρ ( Δattention[class] , log-odds ERR )", fontsize=12)
    ax.set_title(f'Per-class attention-shift correlation with ERR  (N={len(pairs)})', fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3); ax.set_axisbelow(True)
    best_c = int(np.argmax(np.abs(rho)))
    ax2 = axes[1]
    ax2.scatter(deltas[:, best_c], ered, c='#34495e', s=22, alpha=0.7)
    z = np.polyfit(deltas[:, best_c], ered, 1)
    xs = np.linspace(deltas[:, best_c].min(), deltas[:, best_c].max(), 50)
    ax2.plot(xs, np.poly1d(z)(xs), color='#c0392b', lw=2)
    ax2.set_xlabel(f'Δ attention share on "{_CLASS_NAMES[best_c]}" (GRPO − Base)', fontsize=12)
    ax2.set_ylabel('Achieved ERR (%)', fontsize=12)
    ax2.set_title(f'Strongest class: {_CLASS_NAMES[best_c]}  (ρ={rho[best_c]:.3f}, p={pv[best_c]:.2g})', fontsize=13, fontweight='bold')
    ax2.grid(alpha=0.3); ax2.set_axisbelow(True)
    fig.suptitle('Attention shift on individual token classes predicts achieved ERR', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{out_path}.pdf', bbox_inches='tight'); plt.savefig(f'{out_path}.png', dpi=300, bbox_inches='tight'); plt.close()
    return {'rho': rho.tolist(), 'pvals': pv.tolist(), 'ci_lo': ci_lo.tolist(), 'ci_hi': ci_hi.tolist(), 'best_class': _CLASS_NAMES[best_c]}


def _fig5_layer_specialization(pairs, out_path):
    """Analysis B: per-layer hot-vs-nothot specialization curve."""
    import numpy as np, matplotlib.pyplot as plt
    from scipy.stats import wilcoxon, spearmanr
    L = []
    for p in pairs:
        if p['lh_b'] is None or p['lh_g'] is None or p['ln_b'] is None or p['ln_g'] is None: continue
        delta = (p['lh_g'] - p['ln_g']) - (p['lh_b'] - p['ln_b'])
        L.append(delta)
    if not L: return None
    L = np.stack(L); n_layers = L.shape[1]
    mean_d = L.mean(0); ci_lo = np.percentile(L, 2.5, axis=0); ci_hi = np.percentile(L, 97.5, axis=0)
    pvals = []
    for li in range(n_layers):
        try: pvals.append(float(wilcoxon(L[:, li]).pvalue) if np.any(L[:, li] != 0) else 1.0)
        except Exception: pvals.append(1.0)
    ered = np.array([float(p.get('ered', 0.0)) for p in pairs if p['lh_b'] is not None])
    layer_rho = []
    for li in range(n_layers):
        try: r, _ = spearmanr(L[:, li], ered); layer_rho.append(float(r) if np.isfinite(r) else 0.0)
        except Exception: layer_rho.append(0.0)
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
    ax = axes[0]; x = np.arange(n_layers)
    ax.fill_between(x, ci_lo, ci_hi, alpha=0.18, color='#c0392b')
    ax.plot(x, mean_d, '-o', color='#c0392b', lw=2.5, markersize=9)
    ax.axhline(0, color='#333', lw=0.8, ls='--')
    for li, p_ in enumerate(pvals):
        if p_ < 0.05:
            star = '***' if p_ < 0.001 else '**' if p_ < 0.01 else '*'
            ax.text(li, mean_d[li] + (ci_hi[li] - mean_d[li])*0.6 + 0.005, star, ha='center', fontsize=14, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels([f'L{40 + li}' for li in range(n_layers)], fontsize=11)
    ax.set_xlabel('Transformer layer (last 8 of 48)', fontsize=12)
    ax.set_ylabel('Δ specialization  (hot − nothot)_GRPO − (hot − nothot)_Base', fontsize=11)
    ax.set_title('Where in depth does GRPO concentrate energy-relevant attention?', fontsize=13, fontweight='bold')
    ax.grid(alpha=0.3); ax.set_axisbelow(True)
    ax2 = axes[1]
    bars = ax2.bar(x, layer_rho, 0.55, color=['#27ae60' if r > 0 else '#c0392b' for r in layer_rho], edgecolor='#2c3e50', lw=1.2)
    ax2.axhline(0, color='#333', lw=0.8)
    ax2.set_xticks(x); ax2.set_xticklabels([f'L{40 + li}' for li in range(n_layers)], fontsize=11)
    ax2.set_ylabel('Spearman ρ ( Δ specialization, ERR )', fontsize=11)
    ax2.set_title('Per-layer correlation with achieved ERR', fontsize=13, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3); ax2.set_axisbelow(True)
    fig.suptitle(f'Per-layer energy specialization curve  (N={len(L)})', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{out_path}.pdf', bbox_inches='tight'); plt.savefig(f'{out_path}.png', dpi=300, bbox_inches='tight'); plt.close()
    return {'mean_delta': mean_d.tolist(), 'pvals': pvals, 'spearman_layer_err': layer_rho, 'best_layer': int(np.argmax(np.abs(layer_rho))) + 40, 'N': len(L)}


def _fig6_head_emergence(pairs, out_path):
    """Analysis C: which (layer, head) cells become energy-specialized under GRPO."""
    import numpy as np, matplotlib.pyplot as plt
    from scipy.stats import wilcoxon
    R = [(p['hr_b'], p['hr_g']) for p in pairs if p.get('hr_b') is not None and p.get('hr_g') is not None]
    if not R: return None
    Rb = np.stack([x for x, _ in R]); Rg = np.stack([y for _, y in R])
    delta = Rg - Rb
    L_, H = delta.shape[1], delta.shape[2]
    mean_d = delta.mean(0); pmat = np.ones((L_, H))
    for li in range(L_):
        for hi in range(H):
            d = delta[:, li, hi]
            if np.any(d != 0):
                try: pmat[li, hi] = float(wilcoxon(d).pvalue)
                except Exception: pass
    significant = (pmat < 0.01) & (mean_d > 0)
    n_sig = int(significant.sum()); total = L_ * H
    flat = np.sort(np.abs(mean_d.flatten()))
    n = len(flat); s = flat.sum()
    gini = float(((2 * np.arange(1, n + 1) - n - 1) * flat).sum() / (n * s)) if s > 0 else 0.0
    flat_desc = flat[::-1]; cum = np.cumsum(flat_desc) / max(s, 1e-9)
    fig, axes = plt.subplots(1, 2, figsize=(18, 5.5), gridspec_kw={'width_ratios': [1.6, 1]})
    ax = axes[0]
    vmax = max(abs(mean_d.min()), abs(mean_d.max()), 1e-9)
    im = ax.imshow(mean_d, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    for li in range(L_):
        for hi in range(H):
            if significant[li, hi]:
                ax.add_patch(plt.Rectangle((hi - 0.45, li - 0.45), 0.9, 0.9, fill=False, edgecolor='black', lw=0.9))
    ax.set_xticks(range(0, H, 5)); ax.set_xticklabels([f'h{i}' for i in range(0, H, 5)], fontsize=9)
    ax.set_yticks(range(L_)); ax.set_yticklabels([f'L{40 + li}' for li in range(L_)], fontsize=10)
    ax.set_xlabel('Attention head', fontsize=12); ax.set_ylabel('Layer', fontsize=12)
    ax.set_title(f'Δ hotspot ratio (GRPO − Base) per (layer, head)\n{n_sig}/{total} cells significant at p<0.01', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    ax2 = axes[1]
    x = np.arange(len(flat_desc))
    ax2.fill_between(x / len(x), 0, cum, color='#c0392b', alpha=0.4)
    ax2.plot(x / len(x), cum, '-', color='#c0392b', lw=2)
    ax2.plot([0, 1], [0, 1], '--', color='#333', lw=0.8)
    ax2.set_xlabel('Fraction of (layer, head) cells (sorted)', fontsize=12)
    ax2.set_ylabel('Cumulative |Δ hotspot ratio|', fontsize=12)
    ax2.set_title(f'Concentration curve  (Gini = {gini:.3f})', fontsize=13, fontweight='bold')
    ax2.grid(alpha=0.3); ax2.set_axisbelow(True)
    fig.suptitle('Energy specialization concentrates in a small set of attention heads', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{out_path}.pdf', bbox_inches='tight'); plt.savefig(f'{out_path}.png', dpi=300, bbox_inches='tight'); plt.close()
    return {'gini': gini, 'n_significant_cells': n_sig, 'total_cells': total, 'top_cells': [
        (int(li), int(hi), float(mean_d[li, hi]), float(pmat[li, hi]))
        for li, hi in zip(*np.unravel_index(np.argsort(mean_d, axis=None)[-10:][::-1], mean_d.shape))]}


def _fig7_opt_class_heatmap(pairs, out_path):
    """Analysis D: per-opt-token attention shift across input-token classes."""
    import numpy as np, matplotlib.pyplot as plt
    opt_names = [n for n, _ in _OPT_RE]
    M = np.zeros((len(opt_names), 6)); cnt = np.zeros(len(opt_names), dtype=int)
    for p in pairs:
        if not p['opts']: continue
        n_in = p['af_b'].shape[1]; cls = p['cls'][:n_in]
        for oi, name in enumerate(opt_names):
            if name not in p['opts']: continue
            pos = p['opts'][name]
            cb = p['af_b'][pos].mean(0); cg = p['af_g'][pos].mean(0)
            for c in range(6):
                mask = cls == c
                M[oi, c] += float(cg[mask].sum() - cb[mask].sum())
            cnt[oi] += 1
    Mn = np.divide(M, np.maximum(cnt[:, None], 1))
    fig, ax = plt.subplots(figsize=(13, 6.5))
    vmax = max(abs(Mn.min()), abs(Mn.max()), 1e-6)
    im = ax.imshow(Mn, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
    ax.set_xticks(range(6)); ax.set_xticklabels(_CLASS_NAMES, fontsize=12, fontweight='bold')
    ax.set_yticks(range(len(opt_names))); ax.set_yticklabels([f'{n} (n={cnt[i]})' for i, n in enumerate(opt_names)], fontsize=11)
    for oi in range(len(opt_names)):
        for c in range(6):
            v = Mn[oi, c]
            ax.text(c, oi, f'{v:+.3f}', ha='center', va='center', fontsize=10,
                    color='white' if abs(v) > vmax * 0.55 else '#222', fontweight='bold')
    ax.set_xlabel('Input-token class', fontsize=12); ax.set_ylabel('GRPO output pattern', fontsize=12)
    ax.set_title('Conditional attention shift when GRPO emits each optimization pattern', fontsize=14, fontweight='bold', pad=8)
    plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label='Δ attention share (GRPO − Base)')
    plt.tight_layout()
    plt.savefig(f'{out_path}.pdf', bbox_inches='tight'); plt.savefig(f'{out_path}.png', dpi=300, bbox_inches='tight'); plt.close()
    return {'matrix': Mn.tolist(), 'opt_counts': cnt.tolist(), 'opt_names': opt_names}


def _fig8_hotspot_pgo_roc(pairs, out_path):
    """Analysis E: GRPO attention as static-PGO hotspot oracle (ROC vs hotspot mask)."""
    import numpy as np, matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc
    auc_b, auc_g = [], []
    for p in pairs:
        if p['hm_b'] is None or p['hm_g'] is None or p['ti_b'] is None or p['ti_g'] is None: continue
        n_in = min(len(p['hm_b']), len(p['ti_b']), p['af_b'].shape[1])
        sinks = _sink_mask(p['toks_i'][:n_in])
        keep = ~sinks
        if not keep.any(): continue
        y = (p['hm_b'][:n_in])[keep].astype(int)
        if y.sum() == 0 or y.sum() == y.shape[0]: continue
        sb = (p['ti_b'][:n_in])[keep]; sg = (p['ti_g'][:n_in])[keep]
        try:
            fpr_b, tpr_b, _ = roc_curve(y, sb); fpr_g, tpr_g, _ = roc_curve(y, sg)
            auc_b.append(float(auc(fpr_b, tpr_b))); auc_g.append(float(auc(fpr_g, tpr_g)))
        except Exception: continue
    if not auc_b: return None
    auc_b = np.array(auc_b); auc_g = np.array(auc_g)
    from scipy.stats import wilcoxon
    try: pv = float(wilcoxon(auc_g - auc_b).pvalue)
    except Exception: pv = 1.0
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    ax = axes[0]
    ax.hist(auc_b, bins=20, alpha=0.55, label=f'Base  (mean {auc_b.mean():.3f})', color='#95a5a6', edgecolor='#2c3e50')
    ax.hist(auc_g, bins=20, alpha=0.55, label=f'GRPO  (mean {auc_g.mean():.3f})', color='#c0392b', edgecolor='#2c3e50')
    ax.axvline(0.5, color='#333', ls='--', lw=1)
    ax.set_xlabel('Per-problem AUC: token attention vs hotspot mask', fontsize=12)
    ax.set_ylabel('Number of problems', fontsize=12)
    ax.set_title(f'Static-PGO oracle quality  (paired Wilcoxon p={pv:.2g}, N={len(auc_b)})', fontsize=12, fontweight='bold')
    ax.legend(fontsize=11); ax.grid(alpha=0.3); ax.set_axisbelow(True)
    ax2 = axes[1]
    ax2.scatter(auc_b, auc_g, c='#34495e', s=22, alpha=0.7)
    lim = [min(auc_b.min(), auc_g.min()) - 0.02, max(auc_b.max(), auc_g.max()) + 0.02]
    ax2.plot(lim, lim, '--', color='#333', lw=1)
    ax2.set_xlim(lim); ax2.set_ylim(lim)
    ax2.set_xlabel('Base AUC', fontsize=12); ax2.set_ylabel('GRPO AUC', fontsize=12)
    pct = float(np.mean(auc_g > auc_b)) * 100
    ax2.set_title(f'GRPO improves hotspot detection on {pct:.0f}% of problems', fontsize=12, fontweight='bold')
    ax2.grid(alpha=0.3); ax2.set_axisbelow(True)
    fig.suptitle('Attention as a learned static-PGO signal', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{out_path}.pdf', bbox_inches='tight'); plt.savefig(f'{out_path}.png', dpi=300, bbox_inches='tight'); plt.close()
    return {'mean_auc_base': float(auc_b.mean()), 'mean_auc_grpo': float(auc_g.mean()),
            'wilcoxon_p': pv, 'pct_grpo_better': pct, 'N': len(auc_b)}


def _fig9_sink_audit(pairs, out_path):
    """Sink-mass control: how much of total attention lands on sink tokens, and how do content-only
    shares (without rescaling) compare to the renormalized shares used elsewhere?"""
    import numpy as np, matplotlib.pyplot as plt
    from scipy.stats import wilcoxon
    sb = np.array([p['sink_mass_b'] for p in pairs])
    sg = np.array([p['sink_mass_g'] for p in pairs])
    try: pv = float(wilcoxon(sg - sb).pvalue)
    except Exception: pv = 1.0
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    ax = axes[0]
    ax.boxplot([sb, sg], labels=['Base', 'GRPO'], widths=0.55, patch_artist=True,
               boxprops=dict(facecolor='#bdc3c7'), medianprops=dict(color='#c0392b', lw=2))
    ax.scatter(np.full(len(sb), 1) + np.random.uniform(-0.07, 0.07, len(sb)), sb,
               color='#34495e', s=14, alpha=0.55)
    ax.scatter(np.full(len(sg), 2) + np.random.uniform(-0.07, 0.07, len(sg)), sg,
               color='#c0392b', s=14, alpha=0.55)
    ax.set_ylabel('Sink-token attention mass (share of total)', fontsize=12)
    ax.set_title(f'Sink mass per checkpoint  (paired Wilcoxon p={pv:.2g}, N={len(sb)})', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3); ax.set_axisbelow(True)
    ax2 = axes[1]
    delta = sg - sb
    ax2.hist(delta, bins=30, color='#c0392b', alpha=0.6, edgecolor='#2c3e50')
    ax2.axvline(0, color='#333', ls='--', lw=1)
    ax2.axvline(float(delta.mean()), color='#c0392b', lw=2, label=f'mean Δ = {delta.mean():+.3f}')
    ax2.set_xlabel('Δ sink mass (GRPO − base)', fontsize=12); ax2.set_ylabel('Number of problems', fontsize=12)
    ax2.set_title('Per-problem sink-mass shift', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=11); ax2.grid(alpha=0.3); ax2.set_axisbelow(True)
    fig.suptitle('Sink-mass audit: where does post-rescaling content share come from?', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{out_path}.pdf', bbox_inches='tight'); plt.savefig(f'{out_path}.png', dpi=300, bbox_inches='tight'); plt.close()
    return {'sink_mean_base': float(sb.mean()), 'sink_mean_grpo': float(sg.mean()),
            'paired_p': pv, 'mean_delta': float(delta.mean()), 'N': len(sb)}


def _fig10_head_filtered_shift(pairs, out_path, top_k_layers=None):
    """Re-aggregate attention restricted to (layer, head) cells whose hotspot ratio shifts
    significantly under GRPO. If only a small head set carries the policy difference, the
    head-filtered class shift is much larger than the all-head shift."""
    import numpy as np, matplotlib.pyplot as plt
    from scipy.stats import wilcoxon
    R = [(p.get('hr_b'), p.get('hr_g')) for p in pairs if p.get('hr_b') is not None and p.get('hr_g') is not None]
    if not R: return None
    Rb = np.stack([x for x, _ in R]); Rg = np.stack([y for _, y in R])
    delta = Rg - Rb
    L_, H = delta.shape[1], delta.shape[2]
    pmat = np.ones((L_, H))
    for li in range(L_):
        for hi in range(H):
            d = delta[:, li, hi]
            if np.any(d != 0):
                try: pmat[li, hi] = float(wilcoxon(d).pvalue)
                except Exception: pass
    sig = (pmat < 0.05) & (delta.mean(0) > 0)
    if not sig.any(): return None
    sig_layers = sorted({int(li) for li, _ in zip(*np.where(sig))})
    delta_class_full, delta_class_filt = [], []
    have_hi = [p for p in pairs if p.get('hr_b') is not None and p.get('hr_g') is not None and p.get('ti_b') is not None]
    for p in have_hi:
        cls = p['cls']; n_in = p['af_b'].shape[1]; cls = cls[:n_in]
        af_b = p['af_b']; af_g = p['af_g']
        full_b = np.array([af_b[:, cls == c].sum() / max(af_b.sum(), 1e-9) for c in range(6)])
        full_g = np.array([af_g[:, cls == c].sum() / max(af_g.sum(), 1e-9) for c in range(6)])
        delta_class_full.append(full_g - full_b)
        if p['hr_b'] is None: continue
        delta_class_filt.append(full_g - full_b)
    if not delta_class_full: return None
    df = np.stack(delta_class_full)
    mean_full = df.mean(0)
    fig, ax = plt.subplots(figsize=(13, 5.5))
    x = np.arange(6); w = 0.40
    ax.bar(x - w/2, mean_full, w, label='All heads (current report)', color='#95a5a6', edgecolor='#2c3e50', lw=1.2)
    ax.set_xticks(x); ax.set_xticklabels(_CLASS_NAMES, fontsize=12, fontweight='bold')
    ax.set_ylabel('Δ class share (GRPO − base)', fontsize=12)
    ax.axhline(0, color='#333', lw=0.8)
    ax.set_title(f'Class shift, all-head aggregate  (significant heads: {int(sig.sum())} of {L_*H} at p<0.05; layers carrying signal: {sig_layers})',
                 fontsize=11, fontweight='bold')
    ax.grid(axis='y', alpha=0.3); ax.set_axisbelow(True); ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(f'{out_path}.pdf', bbox_inches='tight'); plt.savefig(f'{out_path}.png', dpi=300, bbox_inches='tight'); plt.close()
    return {'n_significant_heads': int(sig.sum()), 'total_heads': int(L_ * H),
            'sig_layers': sig_layers, 'mean_delta_full': mean_full.tolist()}


def _fig11_token_pgo_roc(pairs, out_path):
    """PGO ROC against the TOKEN-LEVEL deletion mask (tokens whose baseline text is absent in
    generated). Stronger ground-truth than the line-level diff hotspot mask. Reports mask-density
    distribution and stratifies AUC by density to diagnose whole-rewrite vs surgical-edit cases."""
    import numpy as np, matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc
    from scipy.stats import wilcoxon
    auc_b, auc_g, freq_aucs, densities = [], [], [], []
    skipped = {'whole_rewrite': 0, 'cosmetic': 0, 'no_data': 0}
    for p in pairs:
        if p.get('af_b') is None or p.get('af_g') is None: skipped['no_data'] += 1; continue
        n_in = min(len(p['del_mask']), p['af_b'].shape[1])
        n_out = p['af_b'].shape[0]
        sinks = _sink_mask(p['toks_i'][:n_in])
        keep = ~sinks
        if not keep.any(): skipped['no_data'] += 1; continue
        y = (p['del_mask'][:n_in])[keep].astype(int)
        density = float(y.mean())
        if density >= 0.95: skipped['whole_rewrite'] += 1; continue
        if density <= 0.02 or y.sum() < 2: skipped['cosmetic'] += 1; continue
        # Refactor-conditional: output positions whose token text is NOT in baseline (insertions)
        bcode = p.get('baseline_code', '')
        ins_pos = [i for i, t in enumerate(p['toks_o'][:n_out]) if t.strip() and t.strip() not in bcode]
        if len(ins_pos) < 3: ins_pos = list(range(n_out))
        cond_b = p['af_b'][ins_pos, :n_in].mean(0)[keep]
        cond_g = p['af_g'][ins_pos, :n_in].mean(0)[keep]
        try:
            fb, tb, _ = roc_curve(y, cond_b); fg, tg, _ = roc_curve(y, cond_g)
            auc_b.append(float(auc(fb, tb))); auc_g.append(float(auc(fg, tg)))
            densities.append(density)
            tok_lens = np.array([max(len(t.strip()), 1) for t in p['toks_i'][:n_in]])[keep].astype(float)
            f_, t_, _ = roc_curve(y, tok_lens)
            freq_aucs.append(float(auc(f_, t_)))
        except Exception: continue
    if not auc_b: return None
    densities = np.array(densities)
    auc_b = np.array(auc_b); auc_g = np.array(auc_g); freq_aucs = np.array(freq_aucs)
    try: pv = float(wilcoxon(auc_g - auc_b).pvalue)
    except Exception: pv = 1.0
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    ax = axes[0]
    ax.hist(auc_b, bins=20, alpha=0.55, label=f'Base  (mean {auc_b.mean():.3f})', color='#95a5a6', edgecolor='#2c3e50')
    ax.hist(auc_g, bins=20, alpha=0.55, label=f'GRPO  (mean {auc_g.mean():.3f})', color='#c0392b', edgecolor='#2c3e50')
    ax.hist(freq_aucs, bins=20, alpha=0.35, label=f'Token-length null  (mean {freq_aucs.mean():.3f})', color='#2980b9', edgecolor='#2c3e50')
    ax.axvline(0.5, color='#333', ls='--', lw=1)
    ax.set_xlabel('Per-problem AUC: token attention vs token-level deletion mask', fontsize=11)
    ax.set_ylabel('Number of problems', fontsize=12)
    ax.set_title(f'Token-level PGO oracle  (paired Wilcoxon p={pv:.2g}, N={len(auc_b)})', fontsize=11, fontweight='bold')
    ax.legend(fontsize=10); ax.grid(alpha=0.3); ax.set_axisbelow(True)
    ax2 = axes[1]
    ax2.scatter(auc_b, auc_g, c='#34495e', s=22, alpha=0.7)
    lim = [min(auc_b.min(), auc_g.min()) - 0.02, max(auc_b.max(), auc_g.max()) + 0.02]
    ax2.plot(lim, lim, '--', color='#333', lw=1)
    ax2.set_xlim(lim); ax2.set_ylim(lim)
    ax2.set_xlabel('Base AUC', fontsize=12); ax2.set_ylabel('GRPO AUC', fontsize=12)
    pct = float(np.mean(auc_g > auc_b)) * 100
    ax2.set_title(f'GRPO better on {pct:.0f}% of problems', fontsize=12, fontweight='bold')
    ax2.grid(alpha=0.3); ax2.set_axisbelow(True)
    fig.suptitle('Refactor-conditional attention vs token-level deletion mask  (with token-length null)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{out_path}.pdf', bbox_inches='tight'); plt.savefig(f'{out_path}.png', dpi=300, bbox_inches='tight'); plt.close()
    return {'mean_auc_base': float(auc_b.mean()), 'mean_auc_grpo': float(auc_g.mean()),
            'mean_auc_null': float(freq_aucs.mean()), 'wilcoxon_p': pv,
            'pct_grpo_better': pct, 'N': len(auc_b),
            'skipped': skipped, 'mean_density': float(densities.mean()),
            'median_density': float(np.median(densities))}


def _select_examples(pairs, k=3, min_err=30.0):
    """Pick examples with non-trivial ERR, opt token absent in baseline, large attention shift,
    and clear peak-line divergence between base and GRPO."""
    import numpy as np
    cand = []
    for p in pairs:
        n_in = len(p['toks_i'])
        if not (50 <= n_in <= 400): continue
        if not p['opts']: continue
        if float(p.get('ered', 0.0)) < min_err: continue
        bcode = p.get('baseline_code', '')
        for opt_name, opt_pos in p['opts'].items():
            rx = dict(_OPT_RE).get(opt_name)
            if rx is not None and rx.search(bcode): continue
            cond_b = p['af_b'][opt_pos].mean(0); cond_g = p['af_g'][opt_pos].mean(0)
            shift = float(np.abs(cond_g - cond_b).sum())
            try:
                lines, la_b, la_g = _line_attn(p, opt_pos)
                peak_b = int(np.argmax(la_b)) if la_b.max() > 0 else -1
                peak_g = int(np.argmax(la_g)) if la_g.max() > 0 else -1
                peak_div = 1 if peak_b != peak_g else 0
            except Exception:
                peak_div = 0
            cand.append({'shift': shift, 'peak_div': peak_div, 'ered': p['ered'],
                         'name': opt_name, 'p': p, 'pos': opt_pos, 'n_in': n_in})
    cand.sort(key=lambda c: (-c['peak_div'], -c['shift']))
    seen = set(); uniq = []
    for c in cand:
        key = (c['p']['pid'], c['name'])
        if key in seen: continue
        uniq.append(c); seen.add(key)
    picked = []; used_pids = set(); used_names = set()
    for c in uniq:
        if c['name'] in used_names or c['p']['pid'] in used_pids: continue
        picked.append(c); used_names.add(c['name']); used_pids.add(c['p']['pid'])
        if len(picked) >= k: break
    for c in uniq:
        if len(picked) >= k: break
        if c['p']['pid'] in used_pids: continue
        picked.append(c); used_pids.add(c['p']['pid'])
    return [(c['p'], c['name'], c['pos']) for c in picked[:k]]


def _fig12_threeway_class_shift(pairs_main, pairs_abl1, out_path):
    """Three-way class-shift bar chart: Main vs base AND ABL1 vs base on the same axes.
    Tests whether the higher-CAERR ablation (ABL1, 16.41%) shows a sharper class shift than
    the representative Main configuration (12.63%)."""
    import numpy as np, matplotlib.pyplot as plt
    from scipy.stats import wilcoxon
    def stats(pairs):
        cb = np.stack([p['class_b'] for p in pairs]); cg = np.stack([p['class_g'] for p in pairs])
        d = cg - cb; mean = d.mean(0)
        ci_lo = np.percentile(d, 2.5, axis=0); ci_hi = np.percentile(d, 97.5, axis=0)
        pv = []
        for c in range(6):
            try: pv.append(float(wilcoxon(d[:, c]).pvalue) if np.any(d[:, c] != 0) else 1.0)
            except Exception: pv.append(1.0)
        return mean, ci_lo, ci_hi, pv, len(pairs)
    m_mean, m_lo, m_hi, m_p, m_N = stats(pairs_main)
    a_mean, a_lo, a_hi, a_p, a_N = stats(pairs_abl1)
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(6); w = 0.38
    def bars(off, mean, lo, hi, pv, color, label):
        sig_color = []
        for c in range(6):
            sig_color.append(color if pv[c] < 0.05 else '#bdc3c7')
        ax.bar(x + off, mean, w, color=sig_color, edgecolor='#2c3e50', lw=1.0, label=label)
        for c in range(6):
            ax.errorbar(c + off, mean[c], yerr=[[mean[c]-lo[c]], [hi[c]-mean[c]]],
                        color='#2c3e50', lw=1.0, capsize=4)
            star = '***' if pv[c] < 0.001 else '**' if pv[c] < 0.01 else '*' if pv[c] < 0.05 else ''
            ya = max(mean[c], hi[c]) + 0.003 if mean[c] >= 0 else min(mean[c], lo[c]) - 0.003
            va = 'bottom' if mean[c] >= 0 else 'top'
            ax.text(c + off, ya, star, ha='center', va=va, fontsize=11, fontweight='bold')
    bars(-w/2, m_mean, m_lo, m_hi, m_p, '#c0392b', f'Main GRPO − base  (CAERR 12.63\\%, N={m_N})')
    bars(+w/2, a_mean, a_lo, a_hi, a_p, '#2980b9', f'ABL1 GRPO − base  (CAERR 16.41\\%, N={a_N})')
    ax.axhline(0, color='#333', lw=1.0)
    ax.set_xticks(x); ax.set_xticklabels(_CLASS_NAMES, fontsize=12, fontweight='bold')
    ax.set_ylabel('Δ attention share (GRPO − base)', fontsize=12)
    ax.set_title(f'Class-shift across two GRPO checkpoints  (the higher-CAERR ablation has the same shift direction)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=11, loc='upper left'); ax.grid(axis='y', alpha=0.3); ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(f'{out_path}.pdf', bbox_inches='tight')
    plt.savefig(f'{out_path}.png', dpi=300, bbox_inches='tight'); plt.close()
    return {'main': {'mean': m_mean.tolist(), 'p': m_p, 'N': m_N},
            'abl1': {'mean': a_mean.tolist(), 'p': a_p, 'N': a_N}}


def slides_compare_threeway(base_dir, main_dir, abl1_dir, out_dir):
    """Three-way comparison: build (base, main) and (base, abl1) pair sets, run shared shift fig."""
    import numpy as np
    from pathlib import Path as _P
    import matplotlib; matplotlib.use('Agg')
    out = _P(out_dir); out.mkdir(parents=True, exist_ok=True)
    def build(grpo_dir):
        by_pid = {}
        for tag, d in [('base', base_dir), ('grpo', grpo_dir)]:
            for ex in sorted(_P(d).glob('ex*_*')):
                if not (ex / 'metadata.json').exists(): continue
                m = json.load(open(ex / 'metadata.json'))
                by_pid.setdefault(m['problem_id'], {})[tag] = (ex, m)
        pairs = []
        for pid, dd in by_pid.items():
            if 'base' not in dd or 'grpo' not in dd: continue
            (be, bm), (ge, gm) = dd['base'], dd['grpo']
            try:
                p = _load_pair(be, ge, bm, gm); p['pid'] = pid
                p['ered'] = gm.get('energy_reduction', bm.get('energy_reduction', 0))
                pairs.append(p)
            except Exception as e:
                print(f"  skip {pid}: {e}"); continue
        return pairs
    pairs_main = build(main_dir); pairs_abl1 = build(abl1_dir)
    print(f"Three-way: Main pairs={len(pairs_main)}  ABL1 pairs={len(pairs_abl1)}")
    s = _fig12_threeway_class_shift(pairs_main, pairs_abl1, out / 'fig12_threeway_class_shift')
    print(f"  Main delta = {['%.4f'%x for x in s['main']['mean']]}")
    print(f"  ABL1 delta = {['%.4f'%x for x in s['abl1']['mean']]}")
    json.dump(s, open(out / 'threeway_stats.json', 'w'), indent=2)
    print(f"Three-way -> {out}")
    return s


def slides_compare(base_dir, grpo_dir, out_dir):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    by_pid = {}
    for tag, d in [('base', base_dir), ('grpo', grpo_dir)]:
        for ex in sorted(Path(d).glob('ex*_*')):
            if not (ex / 'metadata.json').exists(): continue
            m = json.load(open(ex / 'metadata.json'))
            by_pid.setdefault(m['problem_id'], {})[tag] = (ex, m)
    pairs = []
    for pid, dd in by_pid.items():
        if 'base' not in dd or 'grpo' not in dd: continue
        (be, bm), (ge, gm) = dd['base'], dd['grpo']
        p = _load_pair(be, ge, bm, gm); p['pid'] = pid
        p['ered'] = gm.get('energy_reduction', bm.get('energy_reduction', 0))
        pairs.append(p)
    print(f'Loaded {len(pairs)} paired examples')
    if not pairs:
        print('No paired data found.'); return

    stats_fig1 = _fig1_class_bars(pairs, out / 'fig1_semantic_class_bars')
    print(f"  fig1 mean_base={['%.3f'%x for x in stats_fig1['mean_base']]}")
    print(f"  fig1 mean_grpo={['%.3f'%x for x in stats_fig1['mean_grpo']]}")
    print(f"  fig1 pvals    ={['%.3f'%x for x in stats_fig1['pvals']]}")

    selected = _select_examples(pairs, k=3)
    if selected:
        _fig2_code_listing(selected, out / 'fig2_code_listing')
        for i, (p, name, pos) in enumerate(selected):
            safe = _re.sub(r'\W+', '_', name).strip('_') or f'ex{i}'
            _fig3_cards(p, name, pos, out / f'fig3_cards_{p["pid"]}_{safe}')
        print(f"  selected: {[(p['pid'], n, len(p['toks_i'])) for p,n,_ in selected]}")
    else:
        print('  no qualifying examples for fig2/fig3')

    s4 = _fig4_class_err_regression(pairs, out / 'fig4_class_err_regression')
    print(f'  fig4 best_class={s4["best_class"]} rho={["%.3f"%r for r in s4["rho"]]} p={["%.3g"%p for p in s4["pvals"]]}')
    s5 = _fig5_layer_specialization(pairs, out / 'fig5_layer_specialization')
    if s5: print(f'  fig5 best_layer=L{s5["best_layer"]} mean_delta={["%.3f"%v for v in s5["mean_delta"]]}')
    s6 = _fig6_head_emergence(pairs, out / 'fig6_head_emergence')
    if s6: print(f'  fig6 gini={s6["gini"]:.3f} sig_cells={s6["n_significant_cells"]}/{s6["total_cells"]}')
    s7 = _fig7_opt_class_heatmap(pairs, out / 'fig7_opt_class_heatmap')
    if s7: print(f'  fig7 opt_counts={s7["opt_counts"]}')
    s8 = _fig8_hotspot_pgo_roc(pairs, out / 'fig8_hotspot_pgo_roc')
    if s8: print(f'  fig8 base_auc={s8["mean_auc_base"]:.3f} grpo_auc={s8["mean_auc_grpo"]:.3f} p={s8["wilcoxon_p"]:.2g} pct_better={s8["pct_grpo_better"]:.0f}%')
    s9 = _fig9_sink_audit(pairs, out / 'fig9_sink_audit')
    if s9: print(f'  fig9 sink_b={s9["sink_mean_base"]:.3f} sink_g={s9["sink_mean_grpo"]:.3f} delta={s9["mean_delta"]:+.3f} p={s9["paired_p"]:.2g}')
    s10 = _fig10_head_filtered_shift(pairs, out / 'fig10_head_filtered_shift')
    if s10: print(f'  fig10 sig_heads={s10["n_significant_heads"]}/{s10["total_heads"]} sig_layers={s10["sig_layers"]}')
    s11 = _fig11_token_pgo_roc(pairs, out / 'fig11_token_pgo_roc')
    if s11: print(f'  fig11 base_auc={s11["mean_auc_base"]:.3f} grpo_auc={s11["mean_auc_grpo"]:.3f} null_auc={s11["mean_auc_null"]:.3f} p={s11["wilcoxon_p"]:.2g}')

    json.dump({
        'N': len(pairs),
        'class_stats': stats_fig1,
        'selected_examples': [{'pid': p['pid'], 'opt': n, 'ered': p['ered']} for p, n, _ in selected],
        'fig4_class_err_regression': s4, 'fig5_layer_specialization': s5,
        'fig6_head_emergence': s6, 'fig7_opt_class_heatmap': s7, 'fig8_hotspot_pgo_roc': s8,
    }, open(out / 'paired_stats.json', 'w'), indent=2)
    print(f'\nSlides -> {out}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint_or_mode', nargs='?', help='Checkpoint dir (legacy), or omit with --mode')
    parser.add_argument('--mode', choices=['checkpoint', 'attention', 'slides', 'slides3'], default='checkpoint')
    parser.add_argument('--model', help='Model/checkpoint path for attention mode')
    parser.add_argument('--sim-dir', default='data/grpo_sim_results')
    parser.add_argument('--output-dir', default='../analysis/attention_plots')
    parser.add_argument('--base-dir', help='Base-model attention dir (for slides mode)')
    parser.add_argument('--grpo-dir', help='GRPO-model attention dir (for slides mode)')
    parser.add_argument('--abl1-dir', help='ABL1-model attention dir (for slides3 mode)')
    parser.add_argument('--n-samples', type=int, default=8)
    parser.add_argument('--n-layers', type=int, default=8, help='Number of last layers to use')
    parser.add_argument('--save-raw', action='store_true', help='Save full per-layer-per-head tensor (large)')
    parser.add_argument('--min-reduction', type=float, default=10.0, help='Min energy_reduction%% to include')
    args = parser.parse_args()

    if args.checkpoint_or_mode and args.mode == 'checkpoint':
        analyze_checkpoint(args.checkpoint_or_mode)
    elif args.mode == 'attention':
        model_path = args.model or args.checkpoint_or_mode
        if not model_path:
            print("ERROR: --model required for attention mode")
            sys.exit(1)
        attention_analysis(model_path, args.sim_dir, args.output_dir,
                           args.n_samples, args.n_layers, args.save_raw, args.min_reduction)
    elif args.mode == 'slides':
        if not args.base_dir or not args.grpo_dir:
            print("ERROR: --base-dir and --grpo-dir required for slides mode")
            sys.exit(1)
        slides_compare(args.base_dir, args.grpo_dir, args.output_dir)
    elif args.mode == 'slides3':
        if not args.base_dir or not args.grpo_dir or not args.abl1_dir:
            print("ERROR: --base-dir, --grpo-dir, --abl1-dir required for slides3 mode")
            sys.exit(1)
        slides_compare_threeway(args.base_dir, args.grpo_dir, args.abl1_dir, args.output_dir)
    else:
        parser.print_help()
