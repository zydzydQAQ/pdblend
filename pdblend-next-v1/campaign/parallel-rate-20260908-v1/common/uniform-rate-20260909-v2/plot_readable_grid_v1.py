"""Keep every fixed-step tick, with readable labels on dense scientific plots."""
from report import c, cap_rate, stop_trigger


def plot(result, rows, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.ticker import MultipleLocator, FuncFormatter
    from matplotlib.lines import Line2D
    panels = [('energy_j', 'Energy (kJ)', .001), ('slo_attainment', 'SLO attainment (%)', 100),
              ('ttft_avg_s', 'Avg TTFT (ms)', 1000), ('tpot_avg_s', 'Avg TPOT (ms)', 1000),
              ('generated_token_throughput_tps', 'Output throughput (tokens/s)', 1),
              ('gpu_util', 'GPU utilization (%)', 100)]
    colors = dict(zip(c.SYSTEMS, ['#c43c32', '#285f9e', '#3e8a61', '#9865ad', '#db9628']))
    for group in result['groups']:
        model, dataset = group['model'], group['dataset']
        cap = cap_rate(group['decision'])
        grid = [x for x in group['rate_grid'] if cap is None or x <= cap]
        figure, axes = plt.subplots(2, 3, figsize=(13, 7.2), constrained_layout=True)
        label = 'complete' if group['complete'] else 'partial; boundary pending' if cap is None else 'partial; cap confirmed'
        figure.suptitle(f'{model.upper()} / {dataset} — {label}', fontsize=14)
        for ax, (metric, ylabel, scale) in zip(axes.flat, panels):
            for system in c.SYSTEMS:
                series = {r['rate_rps']: r for r in rows if (r['model'], r['dataset'], r['system']) == (model, dataset, system)}
                mean = [series[x].get(metric + '_mean') if x in series else None for x in grid]
                low = [series[x].get(metric + '_min') if x in series else None for x in grid]
                high = [series[x].get(metric + '_max') if x in series else None for x in grid]
                values = np.array([np.nan if v is None else v * scale for v in mean])
                lo = np.array([np.nan if v is None else v * scale for v in low])
                hi = np.array([np.nan if v is None else v * scale for v in high])
                ax.plot(grid, values, 'o-', color=colors[system], label=system, markersize=3.5, linewidth=1.3)
                ax.fill_between(grid, lo, hi, color=colors[system], alpha=.12)
                partial = [x for x in grid if x in series and series[x].get('incomplete_work_repeats')]
                if partial:
                    ax.scatter(partial, [series[x].get(metric + '_mean', float('nan')) * scale
                        if series[x].get(metric + '_mean') is not None else float('nan') for x in partial],
                        marker='x', s=34, color=colors[system], zorder=4)
            if metric == 'slo_attainment':
                ax.axhline(90, color='#555555', linestyle='--', linewidth=.9)
                ax.set_ylim(-2, 103)
                if cap is not None:
                    trigger = stop_trigger(result['observations'], model, dataset, cap)
                    if trigger:
                        ax.scatter([cap], [trigger['slo_attainment'] * 100], marker='*',
                                   s=115, color=colors['pdblend'], zorder=5)
            ax.set_xlabel('Offered rate (requests/s)')
            ax.set_ylabel(ylabel)
            ax.xaxis.set_major_locator(MultipleLocator(group['rate_step_rps']))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda value, position: f'{value:g}'))
            if len(grid) > 8:
                ax.tick_params(axis='x', labelrotation=45, labelsize=9)
                for tick_label in ax.get_xticklabels():
                    tick_label.set_horizontalalignment('right')
            ax.set_xlim(0, max(grid) + group['rate_step_rps'] * .25)
            ax.grid(alpha=.2)
            ax.spines[['top', 'right']].set_visible(False)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        if cap is not None and stop_trigger(result['observations'], model, dataset, cap):
            handles.append(Line2D([], [], color=colors['pdblend'], marker='*', linestyle='None', markersize=10))
            labels.append('First PDB repeat <90%')
        figure.legend(handles, labels, loc='outside lower center', ncol=6, frameon=False,
                      title='x: incomplete work; shaded range: observed repeats')
        for extension in ('png', 'pdf'):
            path = out / f'{model}-{dataset}.{extension}'
            temporary = path.with_suffix('.tmp.' + extension)
            figure.savefig(temporary, dpi=170, bbox_inches='tight')
            temporary.replace(path)
        plt.close(figure)
