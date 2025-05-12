import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import seaborn as sns
import argparse
import os

np.seterr("raise")

action_name_mapping = {
    "normal": "Normal",
    "skip": "Deletion",
    "backtrack": "Backtrack",
    "disfl": "Filled Pause",
    "form_sub": "Form-based Sub.",
    "sem_sub": "Semantic Sub.",
    "insert": "Insertion",
    "morph_sub": "Morph. Sub.",
}

pal = {
    "Normal": "green",
    "Deletion": "gray",
    "Backtrack": "brown",
    "Filled Pause": "red",
    "Form-based Sub.": "magenta",
    "Semantic Sub.": "blue",
    "Morph. Sub.": "indigo",
    "Insertion": "orange",
}

pal2 = {
    "sub_error": "magenta",
    "insert_error": "gray",
}


def action_prob_plot(output_dir: str, sent_id: str) -> None:
    """Make plot of action probabilities (posterior)

    Args:
        output_dir (str): name of output directory
        sent_id (str): sentence ID
    """

    # KDE Plots of Action Probabilities
    plt.clf()
    fig, ax = plt.subplots(figsize=(8, 2.5))

    probs = pd.read_csv(os.path.join(output_dir, sent_id, "action_probs.csv"))
    probs["particle"] = list(range(len(probs)))
    probs = pd.melt(
        probs, id_vars=["particle"], var_name="action", value_name="probability"
    )
    probs["action"] = probs["action"].replace(action_name_mapping)

    # MAP estimates
    probs_map = probs.groupby("action")["probability"].agg(Mean="mean", Std="std")
    probs_map.to_csv(os.path.join(output_dir, sent_id, "action_probs_map.csv"))

    try:
        sns.kdeplot(
            data=probs,
            x="probability",
            hue="action",
            alpha=0.2,
            fill=True,
            palette=pal,
        )

        plt.xlabel("Action Prior", fontsize=18)
        plt.ylabel("Density", fontsize=18)
        plt.savefig(
            os.path.join(output_dir, sent_id, "action_density.png"),
            dpi=300,
            bbox_inches="tight",
        )
    except FloatingPointError as e:
        print(e)

    # Point Plot of Action probability estimates with 95% interval
    plt.clf()
    fig, ax = plt.subplots(figsize=(12, 9))

    sns.pointplot(
        data=probs,
        x="probability",
        y="action",
        hue="action",
        palette=pal,
        estimator=np.mean,
        errorbar=lambda x: (np.percentile(x, 2.5), np.percentile(x, 97.5)),
    )

    plt.xlabel("Inferred Value", fontsize=18)
    plt.ylabel("Action", fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig(
        os.path.join(output_dir, sent_id, "action_estimates.png"),
        dpi=300,
        bbox_inches="tight",
    )


def rejuv_plot(output_dir: str, sent_id: str) -> None:
    """Make plot of rejuvenations targeting each word in the sentence.

    Args:
        output_dir (str): name of output directory
        sent_id (str): sentence ID
    """

    agg_type = "mean"

    df = pd.read_csv(os.path.join(output_dir, sent_id, "trace_results.csv"))
    df["word_pos"] = df["t"] - 1
    num_words = len(df.word_pos.unique())
    num_particles = len(df) // num_words
    df["particle"] = np.repeat(list(range(num_particles)), num_words)

    # observed words for axis labels
    observed_words = [
        grp.obs_word.unique().item() for key, grp in df.groupby("word_pos")
    ]
    N = len(observed_words)

    plt.clf()

    df = pd.read_csv(os.path.join(output_dir, sent_id, "acceptances_by_t.csv"))
    if "move" not in df.columns:
        # this means that no rejuvenations took place
        return
    df = df[df.move.isin(pal2.keys())]

    fig, ax = plt.subplots(figsize=(1.5 * N, 2.5))
    df.t -= 1
    df.t_prime -= 1
    sns.pointplot(
        data=df,
        x="t_prime",
        y="accepted",
        hue="move",
        palette=pal2,
        estimator="mean",
        markersize=10,
        alpha=0.5,
    )
    plt.xlabel("Word", fontsize=18)
    plt.ylabel("Acceptance\nRate", fontsize=18)
    plt.xticks(
        ticks=list(range(N)),
        labels=observed_words,
        fontsize=15,
    )
    plt.yticks(fontsize=14)
    plt.savefig(
        os.path.join(output_dir, sent_id, "rejuvenations.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.clf()
    plt.close()

    # substitutions
    if "sub_error" in df.move.unique():
        fig, ax = plt.subplots(figsize=(12, 9))
        all_combs = pd.DataFrame(
            [(a, b) for a in range(N) for b in range(N)],
            columns=["t", "t_prime"],
        )
        df_plot = df[df.move.isin(["sub_error"])].copy()
        df_plot = all_combs.merge(df_plot, on=["t", "t_prime"], how="left").fillna(
            {"accepted": 0}
        )
        df_plot = df_plot[["t", "t_prime", "accepted"]].pivot_table(
            index="t_prime",
            columns="t",
            values="accepted",
            aggfunc=agg_type,
            fill_value=0,
        )
        ax = sns.heatmap(df_plot, annot=True, fmt=".3f", cmap="Blues")
        plt.xlabel("Rejuvenation Proposed At", fontsize=18)
        plt.ylabel("Rejuvenation Target Word", fontsize=18)
        plt.xticks(
            ticks=[x + 0.5 for x in list(range(N))],
            labels=observed_words,
            fontsize=15,
            ha="center",
        )
        plt.yticks(
            ticks=[x + 0.5 for x in list(range(N))],
            labels=observed_words,
            fontsize=15,
            va="center",
            rotation=0,
        )
        ax.invert_yaxis()
        plt.savefig(
            os.path.join(output_dir, sent_id, "rejuvenations_sub_grid.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    df_agg = (
        df.groupby(["t", "t_prime"]).agg({"accepted": ["count", "mean"]}).reset_index()
    )

    # Adjust bar width and position so bases tile the x-y plane
    x = np.array(df_agg.t)
    y = np.array(df_agg.t_prime)
    z = np.zeros_like(x)

    # Set bar base dimensions to 1 to remove gaps
    dx = dy = 1.0
    dz = np.array(df_agg[("accepted", "count")])

    means = np.array(df_agg[("accepted", "mean")])

    # Normalize mean values to [0, 1]
    norm = Normalize(vmin=means.min(), vmax=means.max())
    palette = sns.cubehelix_palette(
        start=0.5, rot=-0.5, dark=0.3, light=0.95, reverse=False, as_cmap=True
    )
    colors = palette(norm(means))

    # Add an alpha value (e.g., 0.6) to each RGB color from the colormap
    alpha_value = 0.25
    colors[:, 3] = np.full((len(colors)), alpha_value)

    # Create updated plot
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111, projection="3d")
    ax.bar3d(x, y, z, dx, dy, dz, color=colors, shade=True)

    ax.set_xlabel("Rejuvenation Proposed At")
    ax.set_ylabel("Rejuvenation Target")
    ax.set_zlabel("Rejuvenation Count")

    ax.set_xticks(np.array(list(range(N))))
    ax.set_xticklabels(observed_words)

    ax.set_yticks(np.array(list(range(N))))
    ax.set_yticklabels(observed_words)

    ax.xaxis.labelpad = 10
    ax.yaxis.labelpad = 20
    ax.zaxis.labelpad = 10

    ax.view_init(elev=50, azim=-110)

    for tick in ax.xaxis.get_ticklabels():
        tick.set_horizontalalignment("left")
        tick.set_verticalalignment("bottom")

    for tick in ax.yaxis.get_ticklabels():
        tick.set_horizontalalignment("right")
        tick.set_verticalalignment("bottom")

    # Add colorbar
    mappable = ScalarMappable(cmap=palette, norm=norm)
    mappable.set_array([])
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.6)
    cbar.set_label("Acceptance Rate")

    plt.savefig(
        os.path.join(output_dir, sent_id, "rejuvenations_grid_3d.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # SKIPS and INSERTIONS
    if "insert_error" in df.move.unique():
        fig, ax = plt.subplots(figsize=(12, 9))
        all_combs = pd.DataFrame(
            [(a, b) for a in range(N) for b in range(N)],
            columns=["t", "t_prime"],
        )
        df_plot = df[df.move.isin(["insert_error"])].copy()
        df_plot = all_combs.merge(df_plot, on=["t", "t_prime"], how="left").fillna(
            {"accepted": 0}
        )
        df_plot = df_plot[["t", "t_prime", "accepted"]].pivot_table(
            index="t_prime",
            columns="t",
            values="accepted",
            aggfunc=agg_type,
            fill_value=0,
        )
        ax = sns.heatmap(df_plot, annot=True, fmt=".3f", cmap="Blues")
        plt.xlabel("Rejuvenation Proposed At", fontsize=18)
        plt.ylabel("Rejuvenation Target Word", fontsize=18)
        plt.xticks(
            ticks=[x + 0.5 for x in list(range(len(observed_words)))],
            labels=observed_words,
            fontsize=15,
            ha="center",
        )
        plt.yticks(
            ticks=[x + 0.5 for x in list(range(len(observed_words)))],
            labels=observed_words,
            fontsize=15,
            va="center",
            rotation=0,
        )
        ax.invert_yaxis()
        plt.savefig(
            os.path.join(output_dir, sent_id, "rejuvenations_insert_grid.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()


def action_hist_plot(output_dir: str, sent_id: str) -> None:
    """make a histogram of inferred actions at each word in the sentence (a posterior distribution once all words are processed)

    Args:
        output_dir (str): name of output directory
        sent_id (str): sentence ID
    """
    df = pd.read_csv(os.path.join(output_dir, sent_id, "trace_results.csv"))
    df = df.rename(columns={"t": "word_pos"})
    df["inferred_action"] = df["inferred_action"].replace(action_name_mapping)
    df.word_pos -= 1
    num_words = len(df.word_pos.unique())
    num_particles = len(df) // num_words
    df["particle"] = np.repeat(list(range(num_particles)), num_words)

    # observed words for axis labels
    observed_words = [
        grp.obs_word.unique().item() for key, grp in df.groupby("word_pos")
    ]

    # Inferred Actions Stacked Histograms
    fig, ax = plt.subplots(figsize=(1.5 * len(observed_words), 2.5))
    sns.histplot(
        df,
        x="word_pos",
        hue="inferred_action",
        alpha=0.25,
        multiple="stack",
        discrete=True,
        legend=True,
        palette=pal,
        shrink=0.75,
        stat="count",
    )
    plt.xticks(list(range(len(observed_words))), labels=observed_words, fontsize=13)
    plt.xlabel("Word", fontsize=18)
    plt.ylabel("#Particles\nby Action", fontsize=18)
    plt.savefig(
        os.path.join(f"{output_dir}/{sent_id}", "action_hist.png"),
        dpi=300,
        bbox_inches="tight",
    )


def surprisal_plot(
    output_dir: str, sent_id: str, wts_filename: str, img_out_filename: str
) -> None:
    """Make plots showing noisy-channel and baseline surprisal for words in the sentence.

    Args:
        output_dir (str): name of output directory.
        sent_id (str): sentence ID
        wts_filename (str): path to particle weights file
        img_out_filename (str): path where image should be saved
    """

    # wts_melted = process_weights(output_dir, sent_id, normalize=True)
    wts_melted = process_weights(
        os.path.join(output_dir, sent_id, wts_filename), normalize=False
    )

    df = pd.read_csv(os.path.join(output_dir, sent_id, "trace_results.csv"))
    df = df.rename(columns={"t": "word_pos"})
    df.word_pos -= 1
    num_words = len(df.word_pos.unique())
    num_particles = len(df) // num_words
    df["particle"] = np.repeat(list(range(num_particles)), num_words)

    # observed words for axis labels
    observed_words = [
        grp.obs_word.unique().item() for key, grp in df.groupby("word_pos")
    ]

    # inferred_actions
    actions = pd.read_csv(os.path.join(output_dir, sent_id, "inferred_actions.csv"))
    actions["particle"] = list(range(len(actions)))
    actions = pd.melt(
        actions, id_vars=["particle"], var_name="word_pos", value_name="action"
    )
    actions["word_pos"] = actions["word_pos"].apply(lambda x: x[1:]).astype(int) - 1

    surps = actions.merge(wts_melted, on=["particle", "word_pos"])

    surps["logprob"] = surps["weight"]
    surps["prob"] = np.exp(surps["logprob"])
    surps["surprisal"] = -surps["logprob"]
    surps.surprisal *= np.log2(np.e)

    surps["action"] = surps["action"].replace(action_name_mapping)

    # Surprisal by Particle Plot
    plt.clf()
    fig, ax = plt.subplots(figsize=(1.5 * len(observed_words), 9))
    colormap = sns.color_palette("blend:#ffe8e6,#e84631", as_cmap=True)
    lp_ax = sns.stripplot(
        data=surps,
        x="word_pos",
        y="surprisal",
        hue="action",
        jitter=0.1,
        legend=True,
        alpha=0.25,
        size=5,
        palette=pal,
    )

    surps_agg = surps.groupby("word_pos").mean(numeric_only=True)

    # convert back to surprisal in bits
    surps_agg["surprisal_weighted"] = -np.log2(surps_agg["prob"])

    sns.pointplot(
        data=surps_agg,
        x="word_pos",
        y="surprisal_weighted",
        ax=lp_ax,
        label="Noisy-Channel Surp.",
        color="black",
        linestyles="dashed",
        alpha=0.5,
        markersize=15,
        linewidth=5,
    )

    # BASELINE LM SURPRISAL
    df_lm_only = pd.read_csv(os.path.join(output_dir, sent_id, "lm_only_surps.csv"))
    df_lm_only.surprisal *= np.log2(np.e)  # convert surprisal from nats to bits
    df_lm_only["index"] = df_lm_only["index"] - 1
    pp_ax = sns.pointplot(
        data=df_lm_only,
        x="index",
        y="surprisal",
        ax=lp_ax,
        label="Baseline Surp.",
        color="gray",
        linestyles="dashed",
        alpha=0.5,
        markersize=15,
        linewidth=5,
    )
    plt.setp(pp_ax.lines, zorder=100)
    plt.setp(pp_ax.collections, zorder=100)

    plt.xticks(
        ticks=list(range(len(observed_words))),
        labels=observed_words,
        fontsize=16,
    )
    plt.yticks(fontsize=16)
    plt.xlabel("Word", fontsize=24)
    plt.ylabel("Surprisal (bits)", fontsize=24)
    plt.legend(fontsize=16)
    plt.savefig(
        os.path.join(output_dir, sent_id, img_out_filename),
        dpi=300,
        bbox_inches="tight",
    )


def process_weights(weights_file: str, normalize: bool = True) -> pd.DataFrame:
    """Read the particle weights from a file, optionally normalize, and index by particle number and word position.

    Args:
        weights_file (str): path to particle weights file
        normalize (bool, optional): whether to normalize. Defaults to True.

    Returns:
        pd.DataFrame: DataFrame storing weights in long format
    """

    # particle weights
    wts = pd.read_csv(weights_file)

    # normalize across particles for a given time step and convert logs to probabilities
    if normalize:
        wts = wts.apply(lambda x: x - np.max(x), axis=0)
        wts = wts.apply(lambda x: np.exp(x), axis=0)
        wts = wts.apply(lambda x: x / np.sum(x), axis=0)

    wts["particle"] = list(range(len(wts)))
    wts_melted = pd.melt(
        wts, id_vars=["particle"], var_name="word_pos", value_name="weight"
    )
    wts_melted["word_pos"] = (
        wts_melted["word_pos"].apply(lambda x: x[1:]).astype(int) - 1
    )
    return wts_melted


def inferred_sent_plot(output_dir: str, sent_id: str):
    """Make plot showing posterior distribution over inferred intended sentences.

    Args:
        output_dir (str): name of output directory
        sent_id (str): sentence_id
    """
    df = pd.read_csv(os.path.join(output_dir, sent_id, "inferred_sents.csv"))
    df["proportion"] = 1
    df = df.groupby("inferred_sent").agg({"proportion": "sum"}).reset_index()
    df.proportion = df.proportion / df.proportion.sum()
    df = df.sort_values("proportion", ascending=False).iloc[: min(5, len(df))]

    plt.clf()
    plt.subplots(figsize=(4, 4))
    sns.barplot(data=df, x="proportion", y="inferred_sent")
    plt.ylabel("Intended\nSentence", fontsize=36)
    plt.xlabel("Proportion", fontsize=36)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=24)
    plt.savefig(
        os.path.join(output_dir, sent_id, "inferred_sents.png"), bbox_inches="tight"
    )


def make_sent_inference_plots(output_dir, sent_id):
    action_hist_plot(output_dir, sent_id)
    surprisal_plot(output_dir, sent_id, "log_weights.csv", "surprisals.png")
    inferred_sent_plot(output_dir, sent_id)
    rejuv_plot(output_dir, sent_id)
    action_prob_plot(output_dir, sent_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", default="..")
    parser.add_argument("--output_dir", default="log/debug")
    parser.add_argument("--sent_id", default="")
    args = parser.parse_args()

    make_sent_inference_plots(
        os.path.join(args.base_dir, args.output_dir), args.sent_id
    )
