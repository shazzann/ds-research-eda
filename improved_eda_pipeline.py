import argparse
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid')

MONTH_ORDER = [
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December'
]

ID_COLS = ['province', 'district', 'year', 'month']
AIR_COLS = ['pm2.5_ug_m3', 'so2_ug_m3', 'no2_ug_m3']
HEALTH_COLS = [
    'bronchitis_live_discharges', 'bronchitis_deaths', 'bronchitis_cfr',
    'asthma_live_discharges', 'asthma_deaths', 'asthma_cfr'
]
FOREST_COLS = [
    'tc_loss_ha', 'carbon_gross_emissions_yearly', 'Net_C_Flux_yr-1',
    'Gross Emissions_yr-1', 'Gross_C_Removals_yr-1'
]
FIRE_COLS = ['no_fire_types', 'frp_mean', 'frp_median', 'frp_total', 'brightness', 'bright_t31']
VEG_COLS = ['vim', 'viq', 'vim_anomaly', 'vim_climatology', 'vim_min', 'vim_max']
POP_COLS = ['total_population_1k', 'male_population_1k', 'female_population_1k']

TARGET_COLS = [
    'asthma_live_discharges',
    'bronchitis_live_discharges',
    'asthma_deaths',
    'bronchitis_deaths',
    'asthma_cfr',
    'bronchitis_cfr'
]

NOTABLE_FEATURES = [
    'pm2.5_ug_m3', 'no2_ug_m3', 'so2_ug_m3',
    'frp_total', 'frp_mean', 'brightness', 'bright_t31',
    'tc_loss_ha', 'carbon_gross_emissions_yearly', 'Net_C_Flux_yr-1',
    'vim', 'vim_anomaly', 'vim_climatology',
    'total_population_1k', 'male_population_1k', 'female_population_1k'
]


def safe_cols(df, cols):
    return [c for c in cols if c in df.columns]


def savefig(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    if 'month' in df.columns:
        df['month'] = pd.Categorical(df['month'], categories=MONTH_ORDER, ordered=True)
        df['month_num'] = df['month'].cat.codes + 1

    if 'year' in df.columns:
        df['year'] = pd.to_numeric(df['year'], errors='coerce').astype('Int64')

    return df


def validate_dataset(df: pd.DataFrame) -> dict:
    checks = {}
    checks['row_count'] = int(len(df))
    checks['column_count'] = int(df.shape[1])
    checks['district_count'] = int(df['district'].nunique()) if 'district' in df.columns else 0
    checks['province_count'] = int(df['province'].nunique()) if 'province' in df.columns else 0
    checks['year_min'] = int(df['year'].min()) if 'year' in df.columns else None
    checks['year_max'] = int(df['year'].max()) if 'year' in df.columns else None
    checks['missing_by_column'] = df.isna().sum().to_dict()

    if all(c in df.columns for c in ['district', 'year', 'month']):
        panel = df.groupby(['district', 'year']).size().unstack(fill_value=0)
        checks['district_year_min_records'] = int(panel.min().min())
        checks['district_year_max_records'] = int(panel.max().max())
        checks['complete_12_month_district_years'] = int((panel == 12).sum().sum())
        checks['incomplete_district_years'] = int((panel != 12).sum().sum())

    return checks


def dataset_snapshot(df: pd.DataFrame, outdir: Path):
    snapshot_dir = outdir / 'L0_snapshot'
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.DataFrame({
        'column': df.columns,
        'dtype': [str(df[c].dtype) for c in df.columns],
        'nulls': [int(df[c].isna().sum()) for c in df.columns],
        'unique_values': [int(df[c].nunique(dropna=True)) for c in df.columns],
        'sample_value': [df[c].dropna().iloc[0] if df[c].dropna().shape[0] else None for c in df.columns],
    })
    summary.to_csv(snapshot_dir / 'column_profile.csv', index=False)

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if numeric_cols:
        describe_df = df[numeric_cols].describe().T.reset_index().rename(columns={'index': 'metric'})
        describe_df.to_csv(snapshot_dir / 'numeric_summary.csv', index=False)

    if all(c in df.columns for c in ['district', 'year']):
        completeness = df.groupby(['district', 'year']).size().unstack(fill_value=0)
        completeness.to_csv(snapshot_dir / 'panel_completeness.csv')

        fig, ax = plt.subplots(figsize=(12, 7))
        sns.heatmap(completeness, cmap='Blues', annot=True, fmt='d', cbar=False, linewidths=.3, ax=ax)
        ax.set_title('Records per District-Year')
        savefig(fig, snapshot_dir / 'panel_completeness_heatmap.png')

    corr_cols = safe_cols(df, TARGET_COLS + NOTABLE_FEATURES)
    corr_cols = [c for c in corr_cols if pd.api.types.is_numeric_dtype(df[c])]
    corr_cols = [c for c in corr_cols if df[c].nunique(dropna=True) > 1]

    if len(corr_cols) >= 2:
        corr = df[corr_cols].corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(16, 12))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, cmap='coolwarm', center=0, linewidths=.2, ax=ax)
        ax.set_title('Global Correlation Matrix: Targets + Notable Features')
        savefig(fig, snapshot_dir / 'global_target_feature_correlation_heatmap.png')


def compute_trend_slope(series_df: pd.DataFrame, xcol: str, ycol: str) -> float:
    tmp = series_df[[xcol, ycol]].dropna()
    if len(tmp) < 2:
        return np.nan
    if tmp[xcol].nunique() < 2 or tmp[ycol].nunique() < 2:
        return 0.0
    x = tmp[xcol].astype(float).values
    y = tmp[ycol].astype(float).values
    slope = np.polyfit(x, y, 1)[0]
    return float(slope)


def build_target_rankings(df: pd.DataFrame) -> pd.DataFrame:
    district_target_means = (
        df.groupby(['province', 'district'])[safe_cols(df, TARGET_COLS)]
          .mean(numeric_only=True)
          .reset_index()
    )

    all_rankings = []
    for target in safe_cols(district_target_means, TARGET_COLS):
        temp = district_target_means[['province', 'district', target]].copy()
        temp = temp.sort_values(target, ascending=False).reset_index(drop=True)
        temp['rank'] = np.arange(1, len(temp) + 1)
        temp['target'] = target
        temp = temp.rename(columns={target: 'value'})
        all_rankings.append(temp[['target', 'rank', 'province', 'district', 'value']])

    if all_rankings:
        return pd.concat(all_rankings, ignore_index=True)
    return pd.DataFrame()


def build_target_trend_rankings(df: pd.DataFrame) -> pd.DataFrame:
    results = []

    for district in sorted(df['district'].dropna().unique()):
        ddf = df[df['district'] == district]
        province = ddf['province'].iloc[0] if 'province' in ddf.columns and not ddf.empty else None

        yearly = ddf.groupby('year')[safe_cols(ddf, TARGET_COLS)].mean(numeric_only=True).reset_index()
        for target in safe_cols(yearly, TARGET_COLS):
            slope = compute_trend_slope(yearly, 'year', target)
            results.append({
                'province': province,
                'district': district,
                'target': target,
                'yearly_slope': slope
            })

    trend_df = pd.DataFrame(results)
    if trend_df.empty:
        return trend_df

    ranked = []
    for target in trend_df['target'].dropna().unique():
        temp = trend_df[trend_df['target'] == target].copy()
        temp = temp.sort_values('yearly_slope', ascending=False).reset_index(drop=True)
        temp['rank_worsening'] = np.arange(1, len(temp) + 1)
        ranked.append(temp)

    return pd.concat(ranked, ignore_index=True) if ranked else pd.DataFrame()


def top_level_target_overview(df: pd.DataFrame, outdir: Path):
    level_dir = outdir / 'L1_target_overview'
    level_dir.mkdir(parents=True, exist_ok=True)

    target_cols = safe_cols(df, TARGET_COLS)

    # National KPI summary for targets
    kpi = {
        'districts': int(df['district'].nunique()),
        'provinces': int(df['province'].nunique()),
        'years': f"{int(df['year'].min())}-{int(df['year'].max())}",
        'rows': int(len(df)),
    }
    for col in target_cols:
        kpi[f'avg_{col}'] = float(df[col].mean())
        kpi[f'max_{col}'] = float(df[col].max())
        kpi[f'min_{col}'] = float(df[col].min())
    pd.DataFrame([kpi]).to_csv(level_dir / 'target_kpis.csv', index=False)

    # National yearly trend for targets
    if target_cols:
        yearly_targets = df.groupby('year')[target_cols].mean(numeric_only=True)
        yearly_targets.to_csv(level_dir / 'national_target_yearly_trends.csv')

        fig, axes = plt.subplots(len(target_cols), 1, figsize=(13, 3.2 * len(target_cols)), sharex=True)
        axes = np.atleast_1d(axes)
        for ax, target in zip(axes, target_cols):
            ax.plot(yearly_targets.index, yearly_targets[target], marker='o', linewidth=2)
            ax.set_title(f'National yearly trend: {target}')
            ax.set_ylabel(target)
        axes[-1].set_xlabel('Year')
        plt.tight_layout()
        savefig(fig, level_dir / 'national_target_yearly_trends.png')

    # National monthly pattern for targets
    if target_cols:
        monthly_targets = df.groupby('month')[target_cols].mean(numeric_only=True).reindex(MONTH_ORDER)
        monthly_targets.to_csv(level_dir / 'national_target_monthly_patterns.csv')

        fig, axes = plt.subplots(len(target_cols), 1, figsize=(13, 3.0 * len(target_cols)), sharex=True)
        axes = np.atleast_1d(axes)
        xs = np.arange(len(monthly_targets.index))
        for ax, target in zip(axes, target_cols):
            ax.bar(xs, monthly_targets[target].values)
            ax.set_title(f'National monthly pattern: {target}')
            ax.set_ylabel(target)
        axes[-1].set_xticks(xs)
        axes[-1].set_xticklabels([m[:3] for m in monthly_targets.index], rotation=45)
        plt.tight_layout()
        savefig(fig, level_dir / 'national_target_monthly_patterns.png')

    # District mean ranking tables
    ranking_df = build_target_rankings(df)
    ranking_df.to_csv(level_dir / 'district_target_rankings.csv', index=False)

    # Rank charts by target
    if not ranking_df.empty:
        for target in ranking_df['target'].unique():
            temp = ranking_df[ranking_df['target'] == target].copy()
            temp = temp.sort_values('value', ascending=True)

            fig, ax = plt.subplots(figsize=(11, 7))
            ax.barh(temp['district'], temp['value'])
            ax.set_title(f'District ranking by average {target} (all months + all years)')
            ax.set_xlabel(target)
            savefig(fig, level_dir / f'district_rank_{target}.png')

    # Top-bottom district table
    summary_rows = []
    district_target_mean = (
        df.groupby(['province', 'district'])[target_cols]
          .mean(numeric_only=True)
          .reset_index()
    )
    for target in target_cols:
        desc = district_target_mean.sort_values(target, ascending=False)
        asc = district_target_mean.sort_values(target, ascending=True)
        summary_rows.append({
            'target': target,
            'highest_district': desc.iloc[0]['district'],
            'highest_province': desc.iloc[0]['province'],
            'highest_value': desc.iloc[0][target],
            'lowest_district': asc.iloc[0]['district'],
            'lowest_province': asc.iloc[0]['province'],
            'lowest_value': asc.iloc[0][target]
        })
    pd.DataFrame(summary_rows).to_csv(level_dir / 'top_bottom_target_districts.csv', index=False)

    # Trend ranking by slope
    trend_rank_df = build_target_trend_rankings(df)
    trend_rank_df.to_csv(level_dir / 'district_target_trend_rankings.csv', index=False)

    if not trend_rank_df.empty:
        for target in trend_rank_df['target'].unique():
            temp = trend_rank_df[trend_rank_df['target'] == target].copy()
            temp = temp.sort_values('yearly_slope', ascending=True)

            fig, ax = plt.subplots(figsize=(11, 7))
            ax.barh(temp['district'], temp['yearly_slope'])
            ax.axvline(0, color='black', linewidth=1)
            ax.set_title(f'District trend slope ranking: {target}')
            ax.set_xlabel('Yearly slope (positive = increasing)')
            savefig(fig, level_dir / f'district_trend_slope_{target}.png')

    # Province-level target heatmap
    province_targets = df.groupby('province')[target_cols].mean(numeric_only=True)
    if not province_targets.empty:
        province_targets.to_csv(level_dir / 'province_target_summary.csv')
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(province_targets, cmap='YlOrRd', annot=True, fmt='.2f', linewidths=.3, ax=ax)
        ax.set_title('Province-level target comparison')
        savefig(fig, level_dir / 'province_target_heatmap.png')


def district_target_report(df: pd.DataFrame, district: str, outdir: Path):
    ddf = df[df['district'].astype(str).str.lower() == district.lower()].copy()
    if ddf.empty:
        return None

    district_name = ddf['district'].iloc[0]
    district_slug = district_name.replace(' ', '_')
    district_dir = outdir / 'L2_district_targets' / district_slug
    district_dir.mkdir(parents=True, exist_ok=True)

    target_cols = safe_cols(ddf, TARGET_COLS)
    feature_cols = safe_cols(ddf, NOTABLE_FEATURES)

    summary = {
        'district': district_name,
        'province': ddf['province'].iloc[0] if 'province' in ddf.columns else None,
        'records': int(len(ddf)),
        'years': f"{int(ddf['year'].min())}-{int(ddf['year'].max())}",
    }
    for col in target_cols:
        summary[f'avg_{col}'] = float(ddf[col].mean())
    pd.DataFrame([summary]).to_csv(district_dir / 'district_target_summary.csv', index=False)

    # Yearly trend for each target
    if target_cols:
        yearly_targets = ddf.groupby('year')[target_cols].mean(numeric_only=True)
        yearly_targets.to_csv(district_dir / 'yearly_target_profile.csv')

        fig, axes = plt.subplots(len(target_cols), 1, figsize=(13, 3.2 * len(target_cols)), sharex=True)
        axes = np.atleast_1d(axes)
        for ax, target in zip(axes, target_cols):
            ax.plot(yearly_targets.index, yearly_targets[target], marker='o', linewidth=2)
            ax.set_title(f'{district_name} yearly target trend: {target}')
            ax.set_ylabel(target)
        axes[-1].set_xlabel('Year')
        plt.tight_layout()
        savefig(fig, district_dir / 'yearly_target_profile.png')

    # Monthly pattern for each target
    if target_cols:
        monthly_targets = ddf.groupby('month')[target_cols].mean(numeric_only=True).reindex(MONTH_ORDER)
        monthly_targets.to_csv(district_dir / 'monthly_target_profile.csv')

        fig, axes = plt.subplots(len(target_cols), 1, figsize=(13, 3.0 * len(target_cols)), sharex=True)
        axes = np.atleast_1d(axes)
        xs = np.arange(len(monthly_targets.index))
        for ax, target in zip(axes, target_cols):
            ax.bar(xs, monthly_targets[target].values)
            ax.set_title(f'{district_name} monthly target pattern: {target}')
            ax.set_ylabel(target)
        axes[-1].set_xticks(xs)
        axes[-1].set_xticklabels([m[:3] for m in monthly_targets.index], rotation=45)
        plt.tight_layout()
        savefig(fig, district_dir / 'monthly_target_profile.png')

    # Year-month heatmap for each target
    for target in target_cols:
        pivot = ddf.pivot_table(index='year', columns='month', values=target, aggfunc='mean').reindex(columns=MONTH_ORDER)
        if pivot.notna().sum().sum() == 0:
            continue
        pivot.to_csv(district_dir / f'heatmap_{target}.csv')

        fig, ax = plt.subplots(figsize=(14, 4.5))
        sns.heatmap(pivot, cmap='YlOrRd', annot=True, fmt='.2f', linewidths=.3, ax=ax)
        ax.set_title(f'{district_name}: {target} by year and month')
        savefig(fig, district_dir / f'heatmap_{target}.png')

    # District vs national comparison for targets
    compare_base = df.groupby('district')[target_cols].mean(numeric_only=True)
    if district_name in compare_base.index:
        district_vs_all = pd.DataFrame({
            'district_value': compare_base.loc[district_name],
            'national_avg': compare_base.mean(),
        })
        district_vs_all['difference'] = district_vs_all['district_value'] - district_vs_all['national_avg']
        district_vs_all.reset_index(names='target').to_csv(district_dir / 'district_vs_national_targets.csv', index=False)

    # Correlation heatmap: targets + notable features
    corr_cols = target_cols + feature_cols
    corr_cols = [c for c in corr_cols if pd.api.types.is_numeric_dtype(ddf[c])]
    corr_cols = [c for c in corr_cols if ddf[c].nunique(dropna=True) > 1]

    if len(corr_cols) >= 2:
        corr_df = ddf[corr_cols].corr(method='pearson')
        corr_df.to_csv(district_dir / f'{district_slug}_target_feature_correlation_matrix.csv')

        fig, ax = plt.subplots(figsize=(14, 11))
        sns.heatmap(
            corr_df,
            cmap='coolwarm',
            center=0,
            annot=True,
            fmt='.2f',
            linewidths=.3,
            ax=ax,
            cbar_kws={'shrink': 0.8}
        )
        ax.set_title(f'{district_name} — Target and Feature Correlation Heatmap')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        savefig(fig, district_dir / 'target_feature_correlation_heatmap.png')

    # Target-feature association tables and charts
    assoc_rows = []
    for target in target_cols:
        corr_map = {}
        for feat in feature_cols:
            tmp = ddf[[target, feat]].dropna()
            if len(tmp) >= 3 and tmp[target].nunique() > 1 and tmp[feat].nunique() > 1:
                corr_map[feat] = tmp[target].corr(tmp[feat])
            else:
                corr_map[feat] = np.nan

        corr_series = pd.Series(corr_map).dropna().sort_values()

        if not corr_series.empty:
            fig, ax = plt.subplots(figsize=(9, 6))
            colors = ['#d62728' if v < 0 else '#1f77b4' for v in corr_series.values]
            ax.barh(corr_series.index, corr_series.values, color=colors, edgecolor='white')
            ax.axvline(0, color='black', linewidth=1)
            ax.set_xlim(-1, 1)
            ax.set_xlabel('Pearson correlation')
            ax.set_title(f'{district_name} — Association with {target}')
            plt.tight_layout()
            savefig(fig, district_dir / f'association_with_{target}.png')

            for feat, corr_val in corr_series.items():
                assoc_rows.append({
                    'district': district_name,
                    'target': target,
                    'feature': feat,
                    'correlation': corr_val,
                    'abs_correlation': abs(corr_val)
                })

    assoc_df = pd.DataFrame(assoc_rows)
    if not assoc_df.empty:
        assoc_df = assoc_df.sort_values(['target', 'abs_correlation'], ascending=[True, False])
        assoc_df.to_csv(district_dir / f'{district_slug}_target_feature_associations.csv', index=False)

        top_assoc = (
            assoc_df.groupby('target')
                    .head(10)
                    .reset_index(drop=True)
        )
        top_assoc.to_csv(district_dir / f'{district_slug}_top10_associations_per_target.csv', index=False)

    # Scatter plots for strongest associations
    if not assoc_df.empty:
        for target in target_cols:
            target_assoc = assoc_df[assoc_df['target'] == target].copy().sort_values('abs_correlation', ascending=False)
            top_feats = target_assoc['feature'].head(3).tolist()

            for feat in top_feats:
                tmp = ddf[[feat, target]].dropna()
                if len(tmp) < 3:
                    continue

                fig, ax = plt.subplots(figsize=(7, 5))
                sns.regplot(data=tmp, x=feat, y=target, scatter_kws={'alpha': 0.5, 's': 35}, line_kws={'linewidth': 2}, ax=ax)
                corr_val = tmp[feat].corr(tmp[target])
                ax.set_title(f'{district_name}: {feat} vs {target} (r={corr_val:.2f})')
                plt.tight_layout()
                savefig(fig, district_dir / f'scatter_{feat}_vs_{target}.png')

    return summary


def write_readme(outdir: Path, validation: dict, district_summaries: list):
    readme = outdir / 'README.md'
    lines = [
        '# Target-driven EDA output',
        '',
        '## Dataset validation',
        f"- Rows: {validation.get('row_count')}",
        f"- Columns: {validation.get('column_count')}",
        f"- Districts: {validation.get('district_count')}",
        f"- Provinces: {validation.get('province_count')}",
        f"- Year range: {validation.get('year_min')} to {validation.get('year_max')}",
        f"- Complete district-year panels with 12 months: {validation.get('complete_12_month_district_years')}",
        f"- Incomplete district-years: {validation.get('incomplete_district_years')}",
        '',
        '## Target variables',
        '- asthma_live_discharges',
        '- bronchitis_live_discharges',
        '- asthma_deaths',
        '- bronchitis_deaths',
        '- asthma_cfr',
        '- bronchitis_cfr',
        '',
        '## Output structure',
        '- `L0_snapshot/`: data quality and global checks',
        '- `L1_target_overview/`: top-level target trends and district rankings',
        '- `L2_district_targets/<district>/`: district-level target trends and target-feature associations',
        '',
        '## Districts generated',
    ]

    for item in district_summaries:
        if not item:
            continue
        lines.append(f"- {item['district']} ({item['province']}) | records={item['records']} | years={item['years']}")

    readme.write_text('\n'.join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description='Target-driven EDA pipeline for district-wise monthly dataset')
    parser.add_argument('--data', type=str, required=True, help='Path to CSV dataset')
    parser.add_argument('--outdir', type=str, default='eda_target_output', help='Output directory')
    parser.add_argument('--district', type=str, default=None, help='Optional single district name')
    parser.add_argument('--skip_districts', action='store_true', help='Skip district-level outputs')
    args = parser.parse_args()

    data_path = Path(args.data)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_data(data_path)

    validation = validate_dataset(df)
    (outdir / 'validation_summary.json').write_text(json.dumps(validation, indent=2), encoding='utf-8')

    dataset_snapshot(df, outdir)
    top_level_target_overview(df, outdir)

    district_summaries = []
    if not args.skip_districts:
        districts = [args.district] if args.district else sorted(df['district'].dropna().unique())
        for district in districts:
            district_summaries.append(district_target_report(df, district, outdir))

    write_readme(outdir, validation, district_summaries)
    print(f'Target-driven EDA completed. Output saved to: {outdir.resolve()}')


if __name__ == '__main__':
    main()