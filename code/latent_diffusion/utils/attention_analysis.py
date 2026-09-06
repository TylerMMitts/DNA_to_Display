# Helpers for reading the UNet's cross-attention weights.
#
# Reshapes stored attention into spatial grids, turns those into deviation
# maps against the batch mean, and scores how similar two maps are. Shared by
# every script in analysis/.

import os

import torch
import numpy as np
import matplotlib

if os.name != 'nt' and not os.environ.get('DISPLAY'):
    matplotlib.use('Agg')

import matplotlib.pyplot as plt


def _to_numpy(t):
    # Cross-attention weights are usually still on the GPU and inside autograd,
    # so detach and move them before handing anything to matplotlib/numpy.
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy()
    return np.asarray(t)


def grid_size_from_attention(attention_weights):
    # Attention is [B, N, M] with N = H*W spatial positions. The UNet feature
    # maps are square, so the side length is just sqrt(N).
    N = attention_weights.shape[-2]
    side = int(round(np.sqrt(N)))
    if side * side != N:
        raise ValueError(f"{N} spatial positions is not a square grid")
    return side


def attention_to_grids(attention_weights):
    # [B, N, M] -> [B, H, W, M], one spatial heatmap per SNP token.
    attn = _to_numpy(attention_weights)
    if attn.ndim != 3:
        raise ValueError(f"expected [B, N, M] attention, got shape {attn.shape}")
    B, N, M = attn.shape
    side = grid_size_from_attention(attn)
    return attn.reshape(B, side, side, M)


def attention_to_grid(attention_weights, batch_idx=0):
    # Single batch item: [H, W, M]
    return attention_to_grids(attention_weights)[batch_idx]


def deviation_maps(grids):
    # Subtracts the mean over the batch (genotype) axis.
    # Raw attention is dominated by structure that is identical for every
    # genotype (the UNet attends to the same latent regions regardless of which
    # SNP vector it is conditioned on), which makes raw heatmaps look nearly
    # identical side by side. Removing the across-genotype mean leaves only the
    # part of the attention that the SNP vector is actually responsible for.
    grids = np.asarray(grids)
    return grids - grids.mean(axis=0, keepdims=True)


def resize_map(heatmap, size):
    # Nearest-neighbour resize of a 2D map, used to bring 8x8 / 16x16 attention
    # up to the resolution of a decoded image for overlays.
    heatmap = np.asarray(heatmap)
    h, w = heatmap.shape
    rows = (np.arange(size) * h // size).clip(0, h - 1)
    cols = (np.arange(size) * w // size).clip(0, w - 1)
    return heatmap[rows][:, cols]


def map_similarity(map_a, map_b):
    # Cosine similarity and Pearson correlation between two heatmaps.
    # On raw attention maps cosine similarity is always near 1 (everything is
    # positive and shares the same structure), so the informative number is the
    # one computed on deviation maps, where the shared component is gone.
    a = np.asarray(map_a, dtype=np.float64).ravel()
    b = np.asarray(map_b, dtype=np.float64).ravel()

    denom = np.linalg.norm(a) * np.linalg.norm(b)
    cosine = float(a @ b / denom) if denom > 0 else 0.0

    a_c, b_c = a - a.mean(), b - b.mean()
    denom_c = np.linalg.norm(a_c) * np.linalg.norm(b_c)
    pearson = float(a_c @ b_c / denom_c) if denom_c > 0 else 0.0

    return {'cosine': cosine, 'pearson': pearson}


def token_entropy(attention_weights, normalize=True):
    # Mean entropy of the token distribution at each spatial position.
    # Low entropy means each region of the image is driven by a small number of
    # SNP tokens (localised conditioning); entropy near 1.0 (normalised) means
    # the model spreads attention evenly and the conditioning is not spatially
    # specific at all
    attn = _to_numpy(attention_weights).astype(np.float64)
    M = attn.shape[-1]
    ent = -(attn * np.log(attn + 1e-12)).sum(axis=-1)
    if normalize:
        ent = ent / np.log(M)
    return ent.mean(axis=-1)


def get_top_tokens(attention_weights, top_k=5):
    # Get the top-k SNP tokens for each spatial position based on attention weights.
    top_values, top_indices = torch.topk(attention_weights, top_k, dim=-1)
    return top_indices, top_values


def aggregate_attention_by_region(attention_weights, grid_size=4):
    # Reshapes the attention weights to a grid and aggregates them by spatial region.
    attn = attention_weights[0]
    return attn.view(grid_size, grid_size, -1)


def plot_attention_heatmap(attention_weights, token_idx, save_path=None, show=False):
    # Plots a heatmap of attention weights for a specific SNP token across
    # spatial positions, for the first batch item.
    heatmap = attention_to_grid(attention_weights, batch_idx=0)[:, :, token_idx]

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(heatmap, cmap='hot', interpolation='nearest')
    fig.colorbar(im, ax=ax, label='Attention Weight')
    ax.set_title(f'Attention to SNP Token {token_idx}')
    ax.set_xlabel('Width')
    ax.set_ylabel('Height')

    _finish(fig, save_path, show)
    return heatmap


def visualize_attention_weights(attention_weights, snp_tokens=None, save_path=None,
                                max_regions=4, show=False):
    # Bar chart of the token distribution per spatial region.
    # The attention grids are 8x8 or 16x16, so one subplot per position would be
    # 64-256 panels. Positions are pooled into at most max_regions x max_regions
    # blocks to keep the figure readable.
    grid = attention_to_grid(attention_weights, batch_idx=0)  # [H, W, M]
    H, W, M = grid.shape

    regions = min(max_regions, H, W)
    row_edges = np.linspace(0, H, regions + 1).astype(int)
    col_edges = np.linspace(0, W, regions + 1).astype(int)

    fig, axes = plt.subplots(regions, regions, figsize=(3 * regions, 3 * regions),
                             squeeze=False)

    for i in range(regions):
        for j in range(regions):
            ax = axes[i][j]
            block = grid[row_edges[i]:row_edges[i + 1], col_edges[j]:col_edges[j + 1]]
            ax.bar(range(M), block.reshape(-1, M).mean(axis=0))
            ax.set_title(f'Region ({i},{j})', fontsize=9)
            ax.set_xticks(range(M))
            if snp_tokens is not None:
                ax.set_xticklabels(snp_tokens, rotation=45, fontsize=7)
            ax.set_ylim(0, 1)

    fig.suptitle('SNP token attention by image region')
    _finish(fig, save_path, show)


def save_overlay_figure(image, heatmap, save_path, title=None, cmap='inferno', alpha=0.5):
    # Overlays one attention map on a decoded image, upsampled to image size.
    image = np.asarray(image)
    size = image.shape[0]
    upsampled = resize_map(heatmap, size)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(image)
    axes[0].set_title('Generated image')
    axes[0].axis('off')

    axes[1].imshow(image)
    im = axes[1].imshow(upsampled, cmap=cmap, alpha=alpha, interpolation='bilinear')
    axes[1].set_title('SNP attention overlay')
    axes[1].axis('off')
    fig.colorbar(im, ax=axes[1], fraction=0.046)

    if title:
        fig.suptitle(title)
    _finish(fig, save_path, show=False)


def save_genotype_token_grid(grids, labels, save_path, title=None,
                             deviation=False, sublabels=None):
    # Rows are genotypes, columns are SNP tokens.
    # Every panel shares one colour scale, otherwise matplotlib rescales each
    # panel independently and two genotypes with very different attention
    # magnitudes look identical.
    grids = np.asarray(grids)
    n_rows, _, _, M = grids.shape

    if deviation:
        limit = float(np.abs(grids).max()) or 1.0
        vmin, vmax, cmap = -limit, limit, 'RdBu_r'
    else:
        vmin, vmax, cmap = float(grids.min()), float(grids.max()), 'inferno'

    fig, axes = plt.subplots(n_rows, M,
                             figsize=(1.6 * M + 2.5, 1.6 * n_rows + 1.2),
                             squeeze=False)

    for r in range(n_rows):
        for m in range(M):
            ax = axes[r][m]
            im = ax.imshow(grids[r, :, :, m], cmap=cmap, vmin=vmin, vmax=vmax,
                           interpolation='nearest')
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(f'token {m}', fontsize=9)
            if m == 0:
                label = labels[r]
                if sublabels is not None:
                    label = f'{label}\n{sublabels[r]}'
                ax.set_ylabel(label, fontsize=8, rotation=0,
                              ha='right', va='center', labelpad=38)

    fig.colorbar(im, ax=axes, fraction=0.02,
                 label='attention deviation' if deviation else 'attention weight')

    if title:
        fig.suptitle(title, fontsize=12)
    _finish(fig, save_path, show=False, tight=False)


def save_dominant_token_figure(grids, labels, save_path, title=None):
    # For each spatial position, which SNP token wins the softmax.
    grids = np.asarray(grids)
    n_rows, _, _, M = grids.shape

    cmap = plt.get_cmap('tab10', M)
    fig, axes = plt.subplots(1, n_rows, figsize=(2.4 * n_rows + 2.0, 3.2), squeeze=False)

    for r in range(n_rows):
        ax = axes[0][r]
        im = ax.imshow(grids[r].argmax(axis=-1), cmap=cmap, vmin=-0.5, vmax=M - 0.5,
                       interpolation='nearest')
        ax.set_title(labels[r], fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    cbar = fig.colorbar(im, ax=axes, fraction=0.02, ticks=range(M))
    cbar.set_label('dominant SNP token')

    if title:
        fig.suptitle(title, fontsize=12)
    _finish(fig, save_path, show=False, tight=False)


def save_similarity_scatter(genetic_sim, attention_sim, save_path, title=None,
                            labels=None, xlabel='genetic similarity'):
    # Genetic similarity vs attention-map similarity, one point per genotype pair.
    genetic_sim = np.asarray(genetic_sim, dtype=float)
    attention_sim = np.asarray(attention_sim, dtype=float)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(genetic_sim, attention_sim, s=28, alpha=0.75, edgecolor='none')

    if len(genetic_sim) >= 2 and np.ptp(genetic_sim) > 0:
        slope, intercept = np.polyfit(genetic_sim, attention_sim, 1)
        xs = np.linspace(genetic_sim.min(), genetic_sim.max(), 50)
        ax.plot(xs, slope * xs + intercept, color='crimson', lw=1.5,
                label=f'fit: slope={slope:.3f}')
        ax.legend()

    if labels is not None:
        for x, y, lab in zip(genetic_sim, attention_sim, labels):
            ax.annotate(lab, (x, y), fontsize=6, alpha=0.6,
                        xytext=(2, 2), textcoords='offset points')

    ax.set_xlabel(xlabel)
    ax.set_ylabel('attention map similarity (cosine, deviation maps)')
    ax.axhline(0.0, color='grey', lw=0.8, ls='--')
    if title:
        ax.set_title(title)

    _finish(fig, save_path, show=False)


def _finish(fig, save_path, show, tight=True):
    if tight:
        fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)) or '.', exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)
